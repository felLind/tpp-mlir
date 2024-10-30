// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "LinalgX/Dialect/LinalgX/LinalgXDialect.h"

#include "LinalgX/Dialect/LinalgX/LinalgXOps.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace mlir::linalgx;


void LinalgXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "LinalgX/Dialect/LinalgX/LinalgXOps.cpp.inc"
      >();
}

#include "LinalgX/Dialect/LinalgX/LinalgXOpsDialect.cpp.inc"