//===- TppPasses.td ----------------------------------------*- Tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TECO_DIALECT_TECO_PASSES
#define TECO_DIALECT_TECO_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTeCoToLoops : Pass<"convert-teco-to-loops", "func::FuncOp"> {
  let summary = "Convert teco to loops";
  let description = [{
    Convert teco operations to SCF loops.
  }];
  let dependentDialects = ["scf::SCFDialect",
  "bufferization::BufferizationDialect"];
}

#endif // TECO_DIALECT_TECO_PASSES