//===- TppPasses.td ----------------------------------------*- Tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EINSUM_DIALECT_EINSUM_PASSES
#define EINSUM_DIALECT_EINSUM_PASSES

include "mlir/Pass/PassBase.td"

def ConvertEinsumToLoops : Pass<"convert-einsum-to-loops", "func::FuncOp"> {
  let summary = "Convert Einsum to loops";
  let description = [{
    Convert Einsum operations to SCF loops.
  }];
  let dependentDialects = ["scf::SCFDialect",
  "bufferization::BufferizationDialect",
  "xsmm::XsmmDialect"];
}

#endif // EINSUM_DIALECT_EINSUM_PASSES