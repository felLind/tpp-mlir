//
// Created by felix on 30.09.24.
//

#ifndef EINSUM_DIALECT_EINSUM_EINSUMOPS_H
#define EINSUM_DIALECT_EINSUM_EINSUMOPS_H

#include "Einsum/Dialect/Einsum/EinsumTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Einsum/Dialect/Einsum/EinsumOps.h.inc"

#endif //EINSUM_DIALECT_EINSUM_EINSUMOPS_H
