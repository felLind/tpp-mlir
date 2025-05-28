//
// Created by felix on 30.09.24.
//

#include<stdio.h>
#include "Einsum/Dialect/Einsum/EinsumOps.h"
#include "Einsum/Passes.h"
#include "TPP/Dialect/Xsmm/XsmmOps.h"
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
      lda = ldc = m;
      ldb = k;
    }
    return {m, n, k, lda, ldb, ldc};
  }
};

static DimensionDatas create_dimension_datas(einsum::BinaryContractionOp binaryContractionOp) {
  DimensionDatas result;
    
  DictionaryAttr config = binaryContractionOp.getConfig();
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
  int pos = 0;
  for(auto it = dim_names.getValue().begin(); it != dim_names.getValue().end(); it++){
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
  validationResult &= config.contains("primitive_types");

  return success(validationResult);
}

static scf::ValueVector bodyBuilder(DimensionDatas data, TypedValue<MemRefType> leftBuffer, TypedValue<MemRefType> rightBuffer, TypedValue<MemRefType> outBuffer,
                               OpBuilder &b, Location loc, ValueRange localIvs, xsmm::DataTypeAttr dtype, Value dispatched) {
 
  size_t left_rank = cast<MemRefType>(leftBuffer.getType()).getRank();
  size_t right_rank = cast<MemRefType>(rightBuffer.getType()).getRank();
  size_t out_rank = cast<MemRefType>(outBuffer.getType()).getRank();

  SmallVector<int64_t> left_offsets(left_rank, 0);
  SmallVector<int64_t> left_sizes(left_rank, 1);
  SmallVector<int64_t> left_strides(left_rank, 1);
  SmallVector<int64_t> left_reduced_shape;
  SmallVector<int64_t> right_offsets(right_rank, 0);
  SmallVector<int64_t> right_sizes(right_rank, 1);
  SmallVector<int64_t> right_strides(right_rank, 1);
  SmallVector<int64_t> right_reduced_shape;
  SmallVector<int64_t> out_offsets(out_rank, 0);
  SmallVector<int64_t> out_sizes(out_rank, 1);
  SmallVector<int64_t> out_strides(out_rank, 1);
  SmallVector<int64_t> out_reduced_shape;

  for(auto it = data.data.begin(); it != data.data.end(); it++) {
    DimensionData dim = *it;
    if (dim.pos_left != -1) {
      if (dim.parallel_type.compare("prim") == 0) {
        left_sizes[dim.pos_left] = dim.size; 
        left_reduced_shape.push_back(dim.size);
      } else {
        left_offsets[dim.pos_left] = dim.size - 1;
      }
    }
    if (dim.pos_right != -1) {
      if (dim.parallel_type.compare("prim") == 0) {
        right_sizes[dim.pos_right] = dim.size;
        right_reduced_shape.push_back(dim.size);
       } else {
        right_offsets[dim.pos_right] = dim.size - 1;
      }
    }
    if (dim.pos_out != -1) {
      if (dim.parallel_type.compare("prim") == 0) {
        out_sizes[dim.pos_out] = dim.size;
        out_reduced_shape.push_back(dim.size);
      } else {
        out_offsets[dim.pos_out] = dim.size - 1;
      }
    }
  }
  auto left_subview = b.create<mlir::memref::SubViewOp>(loc, leftBuffer, left_offsets, left_sizes, left_strides);
  auto right_subview = b.create<mlir::memref::SubViewOp>(loc, rightBuffer, right_offsets, right_sizes, right_strides);
  auto out_subview = b.create<mlir::memref::SubViewOp>(loc, outBuffer, out_offsets, out_sizes, out_strides);

  auto left_reduced_subview = memref::SubViewOp::rankReduceIfNeeded(b, loc, left_subview, ArrayRef<int64_t>(left_reduced_shape));
  auto right_reduced_subview = memref::SubViewOp::rankReduceIfNeeded(b, loc, right_subview, ArrayRef<int64_t>(right_reduced_shape));
  auto out_reduced_subview = memref::SubViewOp::rankReduceIfNeeded(b, loc, out_subview, ArrayRef<int64_t>(out_reduced_shape));

  SmallVector<Value> invokeOperands;
  invokeOperands.push_back(dispatched);
  invokeOperands.push_back(*left_reduced_subview);
  invokeOperands.push_back(*right_reduced_subview);
  invokeOperands.push_back(*out_reduced_subview);
  b.create<xsmm::GemmOp>(loc, dtype, invokeOperands);
  return scf::ValueVector();
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
  
    auto outMemrefType = binaryContractionOp.getOut().getType();
  
    auto leftBuffer = binaryContractionOp.getLeft();
    auto rightBuffer = binaryContractionOp.getRight();
    
    auto outBuffer = binaryContractionOp.getOut();
    auto allocBuffer = rewriter.create<mlir::memref::AllocOp>(loc, outMemrefType);
    rewriter.create<mlir::memref::CopyOp>(loc, outBuffer, allocBuffer);

    xsmm::GemmFlagsAttr gemmFlags =
    xsmm::GemmFlagsAttr::get(rewriter.getContext(), xsmm::GemmFlags::NONE);
    //TODO: bf16! 
    auto dtype =
      xsmm::DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::F32);
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
    
    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(), ArrayRef<int64_t>(dim_data.get_kernel_data()));

    auto flags = rewriter.getArrayAttr(gemmFlags);

    Value dispatched = rewriter.create<xsmm::GemmDispatchOp>(
        loc, integer64, dims, flags, dtype);
  

    SmallVector<LoopWrapper, 4> loops;
    SmallVector<Value, 4> ivs;
    ValueRange currentIterArgs = std::nullopt;
    Location currentLoc = loc;

    auto it = dim_data.data.begin();  
    while((*it).parallel_type.compare("prim") != 0 && it != dim_data.data.end()) {
      DimensionData d = (*it);
      Value ubs = rewriter.create<mlir::arith::ConstantIndexOp>(loc, d.size);
      if(d.parallel_type.compare("omp") == 0) {
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

    SmallVector<LoopWrapper, 4> rewinded_loops;

    //TODO better rewind
    for (unsigned i = loops.size() - 1; i > 0; --i) {
      rewinded_loops.push_back(loops[i - 1]);
    }

    for (unsigned i = 0, e = rewinded_loops.size() - 1; i < e; ++i) {
      rewriter.setInsertionPointToEnd(rewinded_loops[i].getBody());
      rewriter.create<scf::YieldOp>(loc, rewinded_loops[i + 1].getResults());
    }
    
    rewriter.setInsertionPointToStart(loops.back().getBody());
    scf::ValueVector results = bodyBuilder(dim_data, leftBuffer, rightBuffer, outBuffer, rewriter, currentLoc, ivs, dtype, dispatched);
    rewriter.setInsertionPointToEnd(loops.back().getBody());
   
    rewriter.create<scf::YieldOp>(loc, results);

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
