//
// Created by felix on 30.09.24.
//

#ifndef LINALGX_DIALECT_LINALGX_LINALGXOPS_H
#define LINALGX_DIALECT_LINALGX_LINALGXOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "LinalgX/Dialect/LinalgX/LinalgXOps.h.inc"

#endif //LINALGX_DIALECT_LINALGX_LINALGXOPS_H
