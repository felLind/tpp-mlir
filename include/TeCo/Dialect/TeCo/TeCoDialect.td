//===- TeCoDialect.td - TeCo dialect ----------------------*- tablegen -*--===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TECO_TECO_DIALECT
#define TECO_TECO_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TeCo dialect definition.
//===----------------------------------------------------------------------===//

def TeCo_Dialect : Dialect {
    let name = "teco";
    let summary = "Tensor Contraction dialect.";
    let description = [{
        This dialect provides a function for tensorn contraction.
    }];
    let cppNamespace = "::mlir::teco";
    let useDefaultTypePrinterParser = 1;
    let dependentDialects = [
    "affine::AffineDialect", "memref::MemRefDialect", "tensor::TensorDialect",
    "arith::ArithDialect", "bufferization::BufferizationDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Base operation definition.
//===----------------------------------------------------------------------===//

class TeCo_Op<string mnemonic, list<Trait> traits = []> :
        Op<TeCo_Dialect, mnemonic, traits>;

#endif // TECO_TECO_DIALECT
