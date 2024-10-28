//===- LocalDialectsLowering.cpp ---------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TeCo/PassBundles.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "TeCo/Dialect/TeCo/TeCoDialect.h"

using namespace mlir;
using namespace mlir::teco;

namespace mlir {
namespace teco {
#define GEN_PASS_DEF_TECOLOCALDIALECTSLOWERING
#include "TeCo/PassBundles.h.inc"
} // namespace tpp
} // namespace mlir

// Lower all local dialects (XSMM, check etc.) to standard dialects
// and function calls.
struct LocalDialectsLowering
    : public tpp::impl::LocalDialectsLoweringBase<LocalDialectsLowering>,
      PassBundle<ModuleOp> {
  using LocalDialectsLoweringBase::LocalDialectsLoweringBase;

  void runOnOperation() override {
    auto module = getOperation();

    // Initialize the pipeline if needed.
    // Otherwise, just run the cached one.
    if (pm.empty())
      constructPipeline();

    if (failed(runPipeline(pm, module)))
      return signalPassFailure();
  }

private:
  void constructPipeline() override {
    pm.addNestedPass<func::FuncOp>(createConvertTeCoToLoops());
  }
};
