//===- PassBundles.td --------------------------------------*- Tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TECO_DIALECT_TECO_PASSBUNDLES
#define TECO_DIALECT_TECO_PASSBUNDLES

include "mlir/Pass/PassBase.td"

def TecoDefaultPipeline : Pass<"teco-default-pipeline", "ModuleOp"> {
  let summary = "The default compiler lowering pipeline";
  let description = [{
    A collection of passes that lower everything to MLIR LLVM IR.
  }];
  let options = [
    Option<"gpuBackend", "gpu", "std::string",
            /*default=*/"\"\"",
           "Optional target GPU backend.">,
  ];
}

def DefaultTecoPasses : Pass<"default-teco-passes", "ModuleOp"> {
  let summary = "Collection of default TPP passes";
  let description = [{
    A collection of passes that lower everything TeCo-related
    to standard low-level dialects.
  }];
}


def LocalDialectsLowering : Pass<"teco-lower-local-dialects", "ModuleOp"> {
  let summary = "Lower all local dialects (XSMM, check etc.).";
  let dependentDialects = ["arith::ArithDialect",
                           "func::FuncDialect",
                           "memref::MemRefDialect",
                           "check::CheckDialect",
                           "perf::PerfDialect",
                           "teco::TeCoDialect",
                           "scf::SCFDialect",
                           "tensor::TensorDialect",
                           "xsmm::XsmmDialect",
                           "LLVM::LLVMDialect"];
}


#endif // TECO_DIALECT_TECO_PASSBUNDLES
