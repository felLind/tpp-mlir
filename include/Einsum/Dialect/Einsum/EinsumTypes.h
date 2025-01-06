//
// Created by felix on 30.09.24.
//

#ifndef EINSUM_DIALECT_EINSUM_EINSUMTYPES_H
#define EINSUM_DIALECT_EINSUM_EINSUMTYPES_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "Einsum/Dialect/Einsum/EinsumOpsTypes.h.inc"

namespace mlir {
namespace einsum {
struct ContractConfigTypeStorage;
} // namespace einsum
} // namespace mlir

namespace mlir {
namespace einsum {

class ContractConfigType : public mlir::Type::TypeBase<ContractConfigType, mlir::Type,
                                               ContractConfigTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  static ContractConfigType get(::mlir::MLIRContext *ctx, ::llvm::StringRef tree);

  ::llvm::StringRef getTree();

  ::llvm::StringRef getDims();

  ::llvm::LogicalResult setDims(::llvm::StringRef newDims);

  static constexpr StringLiteral name = "einsum.contractconfig";
};
} // namespace einsum
} // namespace mlir


#endif //EINSUM_DIALECT_EINSUM_EINSUMTYPES_H
