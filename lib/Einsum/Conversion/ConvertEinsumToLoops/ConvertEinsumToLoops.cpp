//
// Created by felix on 30.09.24.
//

#include<stdio.h>
#include "Einsum/Dialect/Einsum/EinsumOps.h"
#include "Einsum/Passes.h"
#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Dialect/Xsmm/XsmmEnum.h"
#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <iostream>
using namespace mlir;

namespace mlir {
namespace einsum {
#define GEN_PASS_DEF_CONVERTEINSUMTOLOOPS
#include "Einsum/Passes.h.inc"
} // namespace einsum
} // namespace mlir

namespace {

struct LoopWrapper {
  scf::ParallelOp parallelOp = nullptr;
  scf::ForOp forOp = nullptr;

  LoopWrapper(scf::ForOp fo){
    forOp = fo;
  };

  LoopWrapper(scf::ParallelOp po){
    parallelOp = po;
  };

  Block* getBody() {
    return forOp != nullptr ? forOp.getBody() : parallelOp != nullptr ? parallelOp.getBody() : nullptr;
  }

  ResultRange getResults() {
    return forOp != nullptr ? forOp.getResults() : parallelOp != nullptr ? parallelOp.getResults() : ResultRange(nullptr);
  }
};

struct DimensionData {
  std::string name;
  std::string type;
  int64_t size;  
  int64_t pos_left;
  int64_t pos_right;
  int64_t pos_out;
  int64_t stride_left;
  int64_t stride_right;
  int64_t stride_out;
  std::string parallel_type;
};

static int64_t get_position(ArrayAttr dims, StringAttr name) {
  int64_t pos = 0;
  for(auto it = dims.begin(); it != dims.end(); it++){
    StringAttr current_dim = ::llvm::dyn_cast<StringAttr>(*it);
    if (current_dim.compare(name) == 0) {
      return pos;
    }
    pos++;
  }
  return -1;
}

struct DimensionDatas {
  SmallVector<DimensionData> data;
  xsmm::DataType dtype;
  xsmm::UnaryKind prim_first_touch;
  std::string prim_main;
  xsmm::UnaryKind prim_last_touch;

  void add(DimensionData d) {
    data.push_back(d);
  };

  std::vector<int64_t> get_kernel_data() {
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    int64_t strideA;
    int64_t strideB;
    
    for (auto it = data.begin(); it != data.end(); it++) {
      DimensionData d = *it;
      if (d.parallel_type.compare("prim") == 0) {
        if (d.type.compare("m") == 0) {
          m = d.size;
        }
        if (d.type.compare("n") == 0) {
          n = d.size;  
        }
        if (d.type.compare("k") == 0) {
          k = d.size;
        }
      }
    }
    lda = ldc = m;
    ldb = k;
    if (prim_main.compare("BRGEMM") == 0) {
      auto it = data.begin();
      while (it != data.end() && (*it).type.compare("k") != 0) {
        it++;
      }
      strideA = (*it).stride_left;
      strideB = (*it).stride_right;
      return {m, n, k, lda, ldb, ldc, strideA, strideB};
    } else if (prim_main.compare("GEMM") == 0) {
      return {m, n, k, lda, ldb, ldc};
    } else {
      return {};
    }
  }

  int64_t get_batch_size() {
    if (prim_main.compare("BRGEMM") == 0) {
      auto it = data.begin();
      while (it != data.end() && (*it).type.compare("k") != 0) {
        it++;
      }
      return (*it).size;
    } else {
      return 0;
    }
  }

  int64_t get_pos_last_non_prim_k() {
    int64_t pos = data.size() - 1;

    for (auto it = data.end()-1; it != data.begin(); it--) {
      if((*it).parallel_type.compare("prim") != 0 && (*it).type.compare("k") == 0) {
        break;
      }
      pos--;
    }
    return pos;
  }
};

static xsmm::DataType getDType(DictionaryAttr config) {
  xsmm::DataType result;
  if( config.contains("dtype")){
    result = xsmm::symbolizeDataType(::llvm::dyn_cast<StringAttr>(config.get("dtype"))).value_or(xsmm::DataType::F32);
  } 
  else {
    result = xsmm::DataType::F32;
  }
  return result;
}

static xsmm::UnaryKind getUnary(DictionaryAttr config, std::string key){
  xsmm::UnaryKind result;
  if(config.contains(key)) {
      result = xsmm::symbolizeUnaryKind(::llvm::dyn_cast<StringAttr>(config.get(key))).value_or(xsmm::UnaryKind::NONE);
  } else {
    result = xsmm::UnaryKind::NONE;
  }
  return result;
}

static DimensionDatas create_dimension_datas(einsum::BinaryContractionOp binaryContractionOp) {
  DimensionDatas result;
    
  DictionaryAttr config = binaryContractionOp.getConfig();
  
  result.dtype = getDType(config);
  result.prim_first_touch = getUnary(config, "prim_first_touch");
  result.prim_main = config.contains("prim_main") ? ::llvm::dyn_cast<StringAttr>(config.get("prim_main")).str() : "None";
  result.prim_last_touch = getUnary(config, "prim_last_touch");
    
  auto dim_names =  ::llvm::dyn_cast<ArrayAttr>(config.get("dim_names"));
  auto dim_types =  ::llvm::dyn_cast<ArrayAttr>(config.get("dim_types"));
  auto dim_sizes =  ::llvm::dyn_cast<ArrayAttr>(config.get("dim_sizes"));
  auto dims_left =  ::llvm::dyn_cast<ArrayAttr>(config.get("dims_left"));
  auto dims_right =  ::llvm::dyn_cast<ArrayAttr>(config.get("dims_right"));
  auto dims_out =  ::llvm::dyn_cast<ArrayAttr>(config.get("dims_out"));  
  auto strides_left =  ::llvm::dyn_cast<ArrayAttr>(config.get("strides_left"));
  auto strides_right =  ::llvm::dyn_cast<ArrayAttr>(config.get("strides_right"));
  auto strides_out =  ::llvm::dyn_cast<ArrayAttr>(config.get("strides_out"));
  auto parallel_types =  ::llvm::dyn_cast<ArrayAttr>(config.get("parallel_types"));
  
  size_t pos = 0;
  for(auto it = dim_names.begin(); it != dim_names.end(); it++){
    StringAttr name = ::llvm::dyn_cast<StringAttr>(*it);
    DimensionData dim_data;
    dim_data.name = name.str();
    dim_data.type = ::llvm::dyn_cast<StringAttr>(dim_types[pos]).str();
    dim_data.size = ::llvm::dyn_cast<IntegerAttr>(dim_sizes[pos]).getInt();
    dim_data.parallel_type = ::llvm::dyn_cast<StringAttr>(parallel_types[pos]).str();
    dim_data.pos_left = get_position(dims_left, name);
    dim_data.pos_right = get_position(dims_right, name);
    dim_data.pos_out = get_position(dims_out, name);
    dim_data.stride_left = ::llvm::dyn_cast<IntegerAttr>(strides_left[pos]).getInt();
    dim_data.stride_right = ::llvm::dyn_cast<IntegerAttr>(strides_right[pos]).getInt();
    dim_data.stride_out = ::llvm::dyn_cast<IntegerAttr>(strides_out[pos]).getInt();
    result.add(dim_data);
    pos++;
  } 

  return result;
}

static LogicalResult validateConfig(DictionaryAttr config) {
  bool validationResult = true;
  validationResult &= config.contains("dim_names");
  validationResult &= config.contains("dim_sizes");
  validationResult &= config.contains("dim_types");
  validationResult &= config.contains("dims_left");
  validationResult &= config.contains("dims_right");
  validationResult &= config.contains("dims_out");
  validationResult &= config.contains("strides_left");
  validationResult &= config.contains("strides_right");
  validationResult &= config.contains("strides_out");
  validationResult &= config.contains("parallel_types");


  return success(validationResult);
}

static void createUnary(RewriterBase &rewriter, Location loc,
                        ArrayRef<Value> operands, xsmm::UnaryInfo unaryInfo,
                        ArrayAttr flags, xsmm::UnaryKindAttr kind) {
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
      rewriter.getContext(), ArrayRef<int64_t>{unaryInfo.m, unaryInfo.n,
                                               unaryInfo.ldi, unaryInfo.ldo});
  auto dtype = xsmm::utils::getDataType(rewriter, operands.back().getType());
  Value dispatched = rewriter.create<xsmm::UnaryDispatchOp>(
      loc, integer64, kind, dims, flags, dtype);
  SmallVector<Value> invokeOperands;
  invokeOperands.push_back(dispatched);
  invokeOperands.append(operands.begin(), operands.end());
  rewriter.create<xsmm::UnaryOp>(loc, dtype, kind,
                                             invokeOperands);
}

static scf::ValueVector bodyBuilder(RewriterBase &rewriter, Location loc, DimensionDatas data, Value leftBuffer, 
  Value rightBuffer, Value outBuffer) {
  
  xsmm::GemmFlagsAttr gemmFlags =
  xsmm::GemmFlagsAttr::get(rewriter.getContext(), xsmm::GemmFlags::NONE);

  auto dtype =
    xsmm::DataTypeAttr::get(rewriter.getContext(), data.dtype);
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  
  DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
      rewriter.getContext(), ArrayRef<int64_t>(data.get_kernel_data()));

  auto flags = rewriter.getArrayAttr(gemmFlags);

  if (data.prim_main.compare("BRGEMM") == 0) {
    Value dispatched = rewriter.create<xsmm::BrgemmDispatchOp>(
        loc, integer64, dims, flags, dtype);
    
    Value batchDim = rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, data.get_batch_size()));
    
    SmallVector<Value> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.push_back(leftBuffer);
    invokeOperands.push_back(rightBuffer);
    invokeOperands.push_back(outBuffer);
    invokeOperands.push_back(batchDim);
    
    rewriter.create<xsmm::BrgemmOp>(loc, dtype, invokeOperands);
  } else if (data.prim_main.compare("GEMM") == 0) {
     Value dispatched = rewriter.create<xsmm::GemmDispatchOp>(
      loc, integer64, dims, flags, dtype);
  
  
    SmallVector<Value> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.push_back(leftBuffer);
    invokeOperands.push_back(rightBuffer);
    invokeOperands.push_back(outBuffer);
    
    rewriter.create<xsmm::GemmOp>(loc, dtype, invokeOperands);
  }
  return scf::ValueVector();
}

static scf::ValueVector bodyBuilder(RewriterBase &rewriter, Location loc, DimensionDatas data, TypedValue<MemRefType> leftBuffer, 
    TypedValue<MemRefType> rightBuffer, TypedValue<MemRefType> outBuffer, ValueRange localIvs) {
 
  size_t left_rank = cast<MemRefType>(leftBuffer.getType()).getRank();
  size_t right_rank = cast<MemRefType>(rightBuffer.getType()).getRank();
  size_t out_rank = cast<MemRefType>(outBuffer.getType()).getRank();

  OpFoldResult zero = rewriter.createOrFold<mlir::arith::ConstantIndexOp>(loc, 0);
  OpFoldResult one = rewriter.getI64IntegerAttr(1);

  SmallVector<OpFoldResult> left_offsets(left_rank, zero);
  SmallVector<OpFoldResult> left_sizes(left_rank, one);
  SmallVector<OpFoldResult> left_strides(left_rank, one);
  SmallVector<int64_t> left_reduced_shape;
  SmallVector<OpFoldResult> right_offsets(right_rank, zero);
  SmallVector<OpFoldResult> right_sizes(right_rank, one);
  SmallVector<OpFoldResult> right_strides(right_rank, one);
  SmallVector<int64_t> right_reduced_shape;
  SmallVector<OpFoldResult> out_offsets(out_rank, zero);
  SmallVector<OpFoldResult> out_sizes(out_rank, one);
  SmallVector<OpFoldResult> out_strides(out_rank, one);
  SmallVector<int64_t> out_reduced_shape;

  int64_t pos_left = 0;
  int64_t pos_right = 0;
  int64_t pos_out = 0;

  for(auto it = data.data.begin(); it != data.data.end(); it++) {
    DimensionData dim = *it;
    if (dim.pos_left != -1) {
      if (dim.parallel_type.compare("prim") == 0) {
        left_sizes[dim.pos_left] = rewriter.getI64IntegerAttr(dim.size);
        left_reduced_shape.push_back(dim.size);
      } else {
        left_offsets[dim.pos_left] = localIvs[pos_left];
      }
      pos_left++;
    }
    if (dim.pos_right != -1) {
      if (dim.parallel_type.compare("prim") == 0) {
        right_sizes[dim.pos_right] = rewriter.getI64IntegerAttr(dim.size);
        right_reduced_shape.push_back(dim.size);
       } else {
        right_offsets[dim.pos_right] = localIvs[pos_right];
      }
      pos_right++;
    }
    if (dim.pos_out != -1) {
      if (dim.parallel_type.compare("prim") == 0) {
        out_sizes[dim.pos_out] = rewriter.getI64IntegerAttr(dim.size);
        out_reduced_shape.push_back(dim.size);
      } else {
        out_offsets[dim.pos_out] = localIvs[pos_out];
      }
      pos_out++;
    }
  }

  auto left_subview = rewriter.create<mlir::memref::SubViewOp>(loc, leftBuffer, left_offsets, left_sizes, left_strides);
  auto right_subview = rewriter.create<mlir::memref::SubViewOp>(loc, rightBuffer, right_offsets, right_sizes, right_strides);
  auto out_subview = rewriter.create<mlir::memref::SubViewOp>(loc, outBuffer, out_offsets, out_sizes, out_strides);

  auto left_reduced_subview = memref::SubViewOp::rankReduceIfNeeded(rewriter, loc, left_subview, ArrayRef<int64_t>(left_reduced_shape));
  auto right_reduced_subview = memref::SubViewOp::rankReduceIfNeeded(rewriter, loc, right_subview, ArrayRef<int64_t>(right_reduced_shape));
  auto out_reduced_subview = memref::SubViewOp::rankReduceIfNeeded(rewriter, loc, out_subview, ArrayRef<int64_t>(out_reduced_shape));

  int64_t last_k_pos = data.get_pos_last_non_prim_k();
  auto iter = localIvs[last_k_pos];

  // first touch: k iterator == zero
  if (last_k_pos >= 0 && data.prim_first_touch != xsmm::UnaryKind::NONE) {
    Value iterEqZero =
     rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, iter, rewriter.createOrFold<mlir::arith::ConstantIndexOp>(loc, 0));
     rewriter.create<scf::IfOp>(
      loc, iterEqZero, 
            [&](OpBuilder &b, Location loc) {
               auto unaryInfo = xsmm::utils::getUnaryInfo(*out_reduced_subview, *out_reduced_subview,
                                               xsmm::UnaryFlags::BCAST_SCALAR);

              auto flags = b.getArrayAttr(xsmm::UnaryFlagsAttr::get(
                  b.getContext(), xsmm::UnaryFlags::BCAST_SCALAR));
              xsmm::UnaryKindAttr kind =
                  xsmm::UnaryKindAttr::get(b.getContext(), data.prim_first_touch);
              createUnary(rewriter, loc, ArrayRef<Value>({*out_reduced_subview, *out_reduced_subview}), *unaryInfo,
                                              flags, kind);
              b.create<scf::YieldOp>(loc, scf::ValueVector());
            }, nullptr);
  }
  

  scf::ValueVector result = bodyBuilder(rewriter, loc, data, *left_reduced_subview, *right_reduced_subview, *out_reduced_subview);

  // last touch: k iterator == |K|
  if (last_k_pos >= 0 && data.prim_last_touch != xsmm::UnaryKind::NONE) {
    Value iterEqMax =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, iter, rewriter.createOrFold<mlir::arith::ConstantIndexOp>(loc,data.data[last_k_pos].size));
      rewriter.create<scf::IfOp>(
      loc, iterEqMax, 
            [&](OpBuilder &b, Location loc) {
              auto unaryInfo = xsmm::utils::getUnaryInfo(*out_reduced_subview, *out_reduced_subview,
                                               xsmm::UnaryFlags::BCAST_SCALAR);
            
              auto flags = b.getArrayAttr(xsmm::UnaryFlagsAttr::get(
                  b.getContext(), xsmm::UnaryFlags::BCAST_SCALAR));
              xsmm::UnaryKindAttr kind =
                  xsmm::UnaryKindAttr::get(b.getContext(), data.prim_last_touch);
              createUnary(rewriter, loc,  ArrayRef<Value>({*out_reduced_subview, *out_reduced_subview}), *unaryInfo,
                                              flags, kind);
              b.create<scf::YieldOp>(loc, scf::ValueVector());
            }, nullptr);
  }

  return result;
}

struct ConvertBinaryContractionOp
    : public OpRewritePattern<einsum::BinaryContractionOp> {
  using OpRewritePattern<einsum::BinaryContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(einsum::BinaryContractionOp binaryContractionOp,
                                PatternRewriter &rewriter) const override {
    Location loc = binaryContractionOp.getLoc();
    DictionaryAttr config = binaryContractionOp.getConfig();
    if (validateConfig(config).failed()) {
      return failure();
    }

    DimensionDatas dim_data = create_dimension_datas(binaryContractionOp);
  
    Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
   
    auto leftBuffer = binaryContractionOp.getLeft();
    auto rightBuffer = binaryContractionOp.getRight();
    
    auto outBuffer = binaryContractionOp.getOut();

     
    size_t left_rank = cast<MemRefType>(leftBuffer.getType()).getRank();
    size_t right_rank = cast<MemRefType>(rightBuffer.getType()).getRank();
    size_t out_rank = cast<MemRefType>(outBuffer.getType()).getRank();
    if ((left_rank == 2 && right_rank == 2 && out_rank == 2 && data.prim_main.compare("GEMM") == 0) 
      || (left_rank == 3 && right_rank == 3 && out_rank == 2 && data.prim_main.compare("BRGEMM") == 0)) {
      scf::ValueVector results = bodyBuilder(rewriter, loc, dim_data, leftBuffer, rightBuffer, outBuffer);
    } else {
      SmallVector<LoopWrapper, 4> loops;
      SmallVector<Value, 4> ivs;
      ValueRange currentIterArgs = std::nullopt;
      Location currentLoc = loc;

      auto it = dim_data.data.begin();  
      while((*it).parallel_type.compare("prim") != 0 && it != dim_data.data.end()) {
        DimensionData d = (*it);
        Value ubs = rewriter.create<mlir::arith::ConstantIndexOp>(loc, d.size);
        if(d.parallel_type.compare("shared") == 0) {
          auto loop = rewriter.create<scf::ParallelOp>(
          currentLoc, zero, ubs, one, currentIterArgs,
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange iv,
              ValueRange args) {
            ivs.append(iv.begin(), iv.end());
            currentIterArgs = args;
            currentLoc = nestedLoc;
          });
          rewriter.setInsertionPointToStart(loop.getBody());
          loops.push_back(LoopWrapper(loop));
        } else {
          auto loop = rewriter.create<scf::ForOp>(
          currentLoc, zero, ubs, one, currentIterArgs,
          [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
              ValueRange args) {
            ivs.push_back(iv);
            currentIterArgs = args;
            currentLoc = nestedLoc;
          });
          rewriter.setInsertionPointToStart(loop.getBody());
          loops.push_back(LoopWrapper(loop));
        }
        it++;
      }

      if (loops.size() > 1) {
        SmallVector<LoopWrapper, 4> rewinded_loops;
        //TODO better rewind
        for (unsigned i = loops.size() - 1; i > 0; --i) {
          rewinded_loops.push_back(loops[i - 1]);
        }

        for (unsigned i = 0, e = rewinded_loops.size() - 1; i < e; ++i) {
          rewriter.setInsertionPointToEnd(rewinded_loops[i].getBody());
          rewriter.create<scf::YieldOp>(loc, rewinded_loops[i + 1].getResults());
        }
      }
      rewriter.setInsertionPointToStart(loops.back().getBody());
      scf::ValueVector results = bodyBuilder(rewriter, currentLoc, dim_data, leftBuffer, rightBuffer, outBuffer, ivs);
      rewriter.setInsertionPointToEnd(loops.back().getBody());
    
      rewriter.create<scf::YieldOp>(loc, results);
    }
    SmallVector<Value> result({outBuffer});
    rewriter.replaceOp(binaryContractionOp, result);

    return success();
  }
};

void populateEinsumToLoopsPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertBinaryContractionOp>(
      patterns.getContext());
}

struct ConvertEinsumToLoops
    : public einsum::impl::ConvertEinsumToLoopsBase<ConvertEinsumToLoops> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateEinsumToLoopsPatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};


} // namespace
