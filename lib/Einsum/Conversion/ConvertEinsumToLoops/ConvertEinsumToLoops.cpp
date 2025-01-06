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


struct ConvertBinaryContractionOp
    : public OpRewritePattern<einsum::BinaryContractionOp> {
  using OpRewritePattern<einsum::BinaryContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(einsum::BinaryContractionOp binaryContractionOp,
                                PatternRewriter &rewriter) const override {
    Location loc = binaryContractionOp.getLoc();
   
    DictionaryAttr config = binaryContractionOp.getConfig();
    
    Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
  
    auto lhsTensorType = llvm::cast<ShapedType>(binaryContractionOp.getLhs().getType());
    auto lhsMemrefType =
      MemRefType::get(lhsTensorType.getShape(), lhsTensorType.getElementType());

    auto rhsTensorType = llvm::cast<ShapedType>(binaryContractionOp.getRhs().getType());
    auto rhsMemrefType =
     MemRefType::get(rhsTensorType.getShape(), rhsTensorType.getElementType());  

    auto accTensorType = llvm::cast<ShapedType>(binaryContractionOp.getAcc().getType());
    auto accMemrefType =
      MemRefType::get(accTensorType.getShape(), accTensorType.getElementType());  

    auto lhsBuffer = rewriter.create<mlir::bufferization::ToMemrefOp>(loc, lhsMemrefType, binaryContractionOp.getLhs());
    auto rhsBuffer = rewriter.create<mlir::bufferization::ToMemrefOp>(loc, rhsMemrefType, binaryContractionOp.getRhs());
    auto accBuffer = rewriter.create<mlir::bufferization::ToMemrefOp>(loc, accMemrefType, binaryContractionOp.getAcc());
    auto allocBuffer = rewriter.create<mlir::memref::AllocOp>(loc, accMemrefType);
    rewriter.create<mlir::memref::CopyOp>(loc, accBuffer, allocBuffer);
    
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

          std::size_t endLHS = tree.find("]");
          
          SmallVector<Value> lhsRange;

          std::string lhs = tree.substr(1, endLHS - 1);

          for(StringRef x : llvm::split(lhs, ',')){
            lhsRange.push_back(localIvs[atoi(x.data())]);
          }

          SmallVector<Value> rhsRange;

          tree = tree.substr(endLHS + 3);   

          std::size_t endRHS = tree.find("]");     

          std::string rhs = tree.substr(0, endRHS);

          for(StringRef x : llvm::split(rhs, ',')){
            rhsRange.push_back(localIvs[atoi(x.data())]);
          }

          SmallVector<Value> allocRange;

          tree = tree.substr(endRHS + 4);   

          std::string root = tree.substr(0, tree.size() - 1);

          for(StringRef x : llvm::split(root, ',')){
            allocRange.push_back(localIvs[atoi(x.data())]);
          }

          Value lhsScalar = b.create<memref::LoadOp>(loc, lhsBuffer, lhsRange);
          Value rhsScalar = b.create<memref::LoadOp>(loc, rhsBuffer, rhsRange);
          Value allocScalar = b.create<memref::LoadOp>(loc, allocBuffer, allocRange);
         
         Value prod = b.create<arith::MulFOp>(loc, lhsScalar, rhsScalar);
         Value sum = b.create<arith::AddFOp>(loc, allocScalar, prod);                                                                
         b.create<memref::StoreOp>(loc, sum, allocBuffer, allocRange);
        });
 
    Value result = rewriter.create<mlir::bufferization::ToTensorOp>(loc, accTensorType, allocBuffer, true);
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
