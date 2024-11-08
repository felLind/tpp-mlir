// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LINALGX_DIALECT_LINALGX_LINALGXOPS
#define LINALGX_DIALECT_LINALGX_LINALGXOPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def LinalgX_Dialect : Dialect {
  let name = "linalgx";
  let cppNamespace = "::mlir::linalgx";
  let summary = [{
    Linalg Extension for Tensor Contraction.
  }];
}

//===----------------------------------------------------------------------===//
// BinaryTensorContractionOp
//===----------------------------------------------------------------------===//

def LinalgX_BinaryContractionOp : Op<LinalgX_Dialect, "contract_binary",
    [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = [{Contracts two given Tensors}];
  let description = [{
    Contract two given tensors based on the given einsum-tree.


    ```mlir
    %result = linalgx.contract_binary(%tree,%dims, %lhs, %rhs, %acc) : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    ```
  }];

  let arguments = (ins
   StrAttr:$tree,
   StrAttr:$dims,
   F32Tensor:$lhs,
   F32Tensor:$rhs,
   F32Tensor:$acc   
  );

  let results = (outs F32Tensor:$result);


   let assemblyFormat = [{
    `(` $tree `,` $dims `,` $lhs `,` $rhs `,` $acc `)`
    attr-dict `:` `(` type($lhs) `,` type($rhs) `,` type($acc) `)` `->` type($result)
  }] ;

}

#endif
