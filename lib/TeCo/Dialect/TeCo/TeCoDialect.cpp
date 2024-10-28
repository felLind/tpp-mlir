// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TeCo/Dialect/TeCo/TeCoDialect.h"

#include "TeCo/Dialect/TeCo/TeCoOps.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace mlir::teco;


void TECODialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TeCo/Dialect/TeCo/TeCoOps.cpp.inc"
      >();
}

#include "TeCo/Dialect/TeCo/TeCoOpsDialect.cpp.inc"