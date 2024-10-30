//===- TeCoDialect.td - TeCo dialect ----------------------*- tablegen -*--===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINALGX_LINALGX_DIALECT
#define LINALGX_LINALGX_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// LinalgX dialect definition.
//===----------------------------------------------------------------------===//

def LinalgX_Dialect : Dialect {
    let name = "linalgx";
    let summary = "Linalg Extension for Tensor Contraction.";
    let description = [{
        This dialect provides a function for tensorn contraction.
    }];
    let cppNamespace = "::mlir::linalgx";
    let useDefaultTypePrinterParser = 1;
    let dependentDialects = [
    "affine::AffineDialect", "memref::MemRefDialect", "tensor::TensorDialect",
    "arith::ArithDialect", "bufferization::BufferizationDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Base operation definition.
//===----------------------------------------------------------------------===//

class LinalgX_Op<string mnemonic, list<Trait> traits = []> :
        Op<LinalgX_Dialect, mnemonic, traits>;

#endif // LINALGX_LINALGX_DIALECT
