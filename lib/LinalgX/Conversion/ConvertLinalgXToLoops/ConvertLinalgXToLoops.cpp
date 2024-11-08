//
// Created by felix on 30.09.24.
//
#include "LinalgX/Dialect/LinalgX/LinalgXOps.h"
#include "LinalgX/Passes.h"
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
namespace linalgx {
#define GEN_PASS_DEF_CONVERTLINALGXTOLOOPS
#include "LinalgX/Passes.h.inc"
} // namespace linalgX
} // namespace mlir

namespace {


struct ConvertBinaryContractionOp
    : public OpRewritePattern<linalgx::BinaryContractionOp> {
  using OpRewritePattern<linalgx::BinaryContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalgx::BinaryContractionOp binaryContractionOp,
                                PatternRewriter &rewriter) const override {
    Location loc = binaryContractionOp.getLoc();
   
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
    int dimCount = 0;
    for(StringRef x : llvm::split(binaryContractionOp.getDims(), ',')){
      ubs.push_back(rewriter.create<mlir::arith::ConstantIndexOp>(loc, atoi(x.data())));
      dimCount++;
    } 

    SmallVector<Value> lbs(dimCount, zero);
    SmallVector<Value> steps(dimCount, one);
 
    (void)mlir::scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &b, Location loc, ValueRange localIvs) {
             
          std::string tree = binaryContractionOp.getTree().str();

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

void populateLinalgXToLoopsPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertBinaryContractionOp>(
      patterns.getContext());
}

struct ConvertLinalgXToLoops
    : public linalgx::impl::ConvertLinalgXToLoopsBase<ConvertLinalgXToLoops> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateLinalgXToLoopsPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};


} // namespace
