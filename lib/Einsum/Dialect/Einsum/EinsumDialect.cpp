// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Einsum/Dialect/Einsum/EinsumDialect.h"

#include "Einsum/Dialect/Einsum/EinsumOps.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace mlir::einsum;


void EinsumDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Einsum/Dialect/Einsum/EinsumOps.cpp.inc"
      >();
}

#include "Einsum/Dialect/Einsum/EinsumOpsDialect.cpp.inc"