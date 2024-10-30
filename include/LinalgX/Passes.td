//===- TppPasses.td ----------------------------------------*- Tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINALGX_DIALECT_LINALGX_PASSES
#define LINALGX_DIALECT_LINALGX_PASSES

include "mlir/Pass/PassBase.td"

def ConvertLinalgXToLoops : Pass<"convert-linalgx-to-loops", "func::FuncOp"> {
  let summary = "Convert LinalgX to loops";
  let description = [{
    Convert LinalgX operations to SCF loops.
  }];
  let dependentDialects = ["scf::SCFDialect",
  "bufferization::BufferizationDialect"];
}

#endif // LINALGX_DIALECT_LINALGX_PASSES