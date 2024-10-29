// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TECO_DIALECT_TECO_TECOOPS
#define TECO_DIALECT_TECO_TECOOPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TeCo_Dialect : Dialect {
  let name = "teco";
  let cppNamespace = "::mlir::teco";
  let summary = [{
    A dialect implementing tensor contraction.
  }];
}

//===----------------------------------------------------------------------===//
// TensorContractionOp
//===----------------------------------------------------------------------===//

def TeCo_ContractionOp : Op<TeCo_Dialect, "contract",
    [MemoryEffects<[MemWrite, MemRead]>, Terminator]> {
  let summary = [{Contracts two given Tensors}];
  let description = [{
    Contract two given tensors of four dimensions.

    Results in a tensor with 6 dimensions

    ```mlir
    teco.contract(%arg0, %arg1, %arg2) : tensor<20x20x24x20xf32>, tensor<24x24x20x24xf32>, tensor<24x20x20x24x20x20xf32>
    ```
  }];

  let arguments = (ins
   F32Tensor:$lhs,
   F32Tensor:$rhs,
   F32Tensor:$acc   
);


   let assemblyFormat = [{
    `(` $lhs `,` $rhs `,` $acc `)`
    attr-dict `:` type($lhs) `,` type($rhs) `,` type($acc)
  }];

}

#endif
