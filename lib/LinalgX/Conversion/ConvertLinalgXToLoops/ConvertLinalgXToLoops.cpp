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
    Value twenty = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 20);
    Value twentyfour = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 24);

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
    
    SmallVector<Value> lbs(7, zero);
    SmallVector<Value> steps(7, one);
    SmallVector<Value> ubs;
    ubs.push_back(twentyfour);
    ubs.push_back(twenty);
    ubs.push_back(twenty);
    ubs.push_back(twentyfour);
    ubs.push_back(twenty);
    ubs.push_back(twenty);
    ubs.push_back(twentyfour);

    (void)mlir::scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &b, Location loc, ValueRange localIvs) {
          SmallVector<Value, 4> lhsRange({localIvs[1],localIvs[2],localIvs[6],localIvs[5]});
          SmallVector<Value, 4> rhsRange({localIvs[0],localIvs[3],localIvs[4],localIvs[6]});
          SmallVector<Value, 8> allocRange({localIvs[0],localIvs[1],localIvs[2],localIvs[3],localIvs[4],localIvs[5]});
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
