//
// Created by felix on 30.09.24.
//

#ifndef TECO_DIALECT_TECO_TECOOPS_H
#define TECO_DIALECT_TECO_TECOOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "TeCo/Dialect/TeCo/TeCoOps.h.inc"

#endif //TECO_DIALECT_TECO_TECOOPS_H
