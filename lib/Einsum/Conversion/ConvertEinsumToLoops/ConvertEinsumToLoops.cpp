//
// Created by felix on 30.09.24.
//
#include "Einsum/Dialect/Einsum/EinsumOps.h"
#include "Einsum/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace einsum {
#define GEN_PASS_DEF_CONVERTEINSUMTOLOOPS
#include "Einsum/Passes.h.inc"
} // namespace einsum
} // namespace mlir

namespace {

static bool validateConfig(DictionaryAttr config) {
  bool validationResult = true;
  validationResult &= config.contains("tree");
  validationResult &= config.contains("dim_sizes");
  return validationResult;
}

struct ConvertBinaryContractionOp
    : public OpRewritePattern<einsum::BinaryContractionOp> {
  using OpRewritePattern<einsum::BinaryContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(einsum::BinaryContractionOp binaryContractionOp,
                                PatternRewriter &rewriter) const override {
    Location loc = binaryContractionOp.getLoc();
   
    DictionaryAttr config = binaryContractionOp.getConfig();
    
    if (!validateConfig(config)) {
      return failure();
    }

    Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
  
    auto leftTensorType = llvm::cast<ShapedType>(binaryContractionOp.getLeft().getType());
    auto leftMemrefType =
      MemRefType::get(leftTensorType.getShape(), leftTensorType.getElementType());

    auto rightTensorType = llvm::cast<ShapedType>(binaryContractionOp.getRight().getType());
    auto rightMemrefType =
     MemRefType::get(rightTensorType.getShape(), rightTensorType.getElementType());  

    auto outTensorType = llvm::cast<ShapedType>(binaryContractionOp.getOut().getType());
    auto outMemrefType =
      MemRefType::get(outTensorType.getShape(), outTensorType.getElementType());  

    auto leftBuffer = rewriter.create<mlir::bufferization::ToMemrefOp>(loc, leftMemrefType, binaryContractionOp.getLeft());
    auto rightBuffer = rewriter.create<mlir::bufferization::ToMemrefOp>(loc, rightMemrefType, binaryContractionOp.getRight());
    auto outBuffer = rewriter.create<mlir::bufferization::ToMemrefOp>(loc, outMemrefType, binaryContractionOp.getOut());
    auto allocBuffer = rewriter.create<mlir::memref::AllocOp>(loc, outMemrefType);
    rewriter.create<mlir::memref::CopyOp>(loc, outBuffer, allocBuffer);
    
    SmallVector<Value> ubs;
    auto dim_sizes =  ::llvm::dyn_cast<StringAttr>(config.get("dim_sizes"));
    for(StringRef x : llvm::split(dim_sizes, ',')){
      ubs.push_back(rewriter.create<mlir::arith::ConstantIndexOp>(loc, atoi(x.data())));
    }  
 
    SmallVector<Value> lbs(ubs.size(), zero);
    SmallVector<Value> steps(ubs.size(), one);
 
    (void)mlir::scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &b, Location loc, ValueRange localIvs) {
             
          auto tree = ::llvm::dyn_cast<StringAttr>(config.get("tree")).str();

          std::size_t endLeft = tree.find("]");
          
          SmallVector<Value> leftRange;

          std::string left = tree.substr(1, endLeft - 1);

          for(StringRef x : llvm::split(left, ',')){
            leftRange.push_back(localIvs[atoi(x.data())]);
          }

          SmallVector<Value> rightRange;

          tree = tree.substr(endLeft + 3);   

          std::size_t endRight = tree.find("]");     

          std::string right = tree.substr(0, endRight);

          for(StringRef x : llvm::split(right, ',')){
            rightRange.push_back(localIvs[atoi(x.data())]);
          }

          SmallVector<Value> allocRange;

          tree = tree.substr(endRight + 4);   

          std::string root = tree.substr(0, tree.size() - 1);

          for(StringRef x : llvm::split(root, ',')){
            allocRange.push_back(localIvs[atoi(x.data())]);
          }

          Value leftScalar = b.create<memref::LoadOp>(loc, leftBuffer, leftRange);
          Value rightScalar = b.create<memref::LoadOp>(loc, rightBuffer, rightRange);
          Value allocScalar = b.create<memref::LoadOp>(loc, allocBuffer, allocRange);
         
         Value prod = b.create<arith::MulFOp>(loc, leftScalar, rightScalar);
         Value sum = b.create<arith::AddFOp>(loc, allocScalar, prod);                                                                
         b.create<memref::StoreOp>(loc, sum, allocBuffer, allocRange);
        });
 
    Value result = rewriter.create<mlir::bufferization::ToTensorOp>(loc, outTensorType, allocBuffer, true);
    SmallVector<Value> results({result});
    rewriter.replaceOp(binaryContractionOp, results);
    
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
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};


} // namespace
