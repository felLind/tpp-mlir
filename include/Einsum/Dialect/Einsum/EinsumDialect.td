//===- TeCoDialect.td - TeCo dialect ----------------------*- tablegen -*--===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EINSUM_EINSUM_DIALECT
#define EINSUM_EINSUM_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Einsum dialect definition.
//===----------------------------------------------------------------------===//

def Einsum_Dialect : Dialect {
    let name = "einsum";
    let summary = "Einsum dialect for Tensor Contraction.";
    let description = [{
        This dialect provides functions for tensor contraction.
    }];
    let cppNamespace = "::mlir::einsum";
    let useDefaultTypePrinterParser = 1;
    let dependentDialects = [
    "affine::AffineDialect", "memref::MemRefDialect", "tensor::TensorDialect",
    "arith::ArithDialect", "bufferization::BufferizationDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Base operation definition.
//===----------------------------------------------------------------------===//

class Einsum_Op<string mnemonic, list<Trait> traits = []> :
        Op<Einsum_Dialect, mnemonic, traits>;

#endif // EINSUM_EINSUM_DIALECT
