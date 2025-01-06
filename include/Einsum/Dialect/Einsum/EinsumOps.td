// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EINSUM_DIALECT_EINSUM_EINSUMOPS
#define EINSUM_DIALECT_EINSUM_EINSUMOPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Einsum_Dialect : Dialect {
  let name = "einsum";
  let cppNamespace = "::mlir::einsum";
  let summary = [{
    Einsum dialect for Tensor Contraction.
  }];
}

//===----------------------------------------------------------------------===//
// BinaryTensorContractionOp
//===----------------------------------------------------------------------===//

def Einsum_BinaryContractionOp : Op<Einsum_Dialect, "contract_binary",
    [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = [{Contracts two given Tensors}];
  let description = [{
    Contract two given tensors based on the given einsum-tree.


    ```mlir
    %result = einsum.contract_binary(%config, %lhs, %rhs, %acc) : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    ```
  }];

  let arguments = (ins
   DictionaryAttr:$config,
   F32Tensor:$lhs,
   F32Tensor:$rhs,
   F32Tensor:$acc   
  );

  let results = (outs F32Tensor:$result);


   let assemblyFormat = [{
    `(` $config`,` $lhs `,` $rhs `,` $acc `)`
    attr-dict `:` `(` type($lhs) `,` type($rhs) `,` type($acc) `)` `->` type($result)
  }] ;

}

#endif
