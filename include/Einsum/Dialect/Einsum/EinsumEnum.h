//===- XsmmEnum.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EINSUM_DIALECT_EINSUM_EINSUMENUM_H
#define EINSUM_DIALECT_EINSUM_EINSUMENUM_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "Einsum/Dialect/Einsum/EinsumEnum.h.inc"

#endif // EINSUM_DIALECT_EINSUM_EINSUMENUM_H
