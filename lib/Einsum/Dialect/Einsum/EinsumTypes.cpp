#include "Einsum/Dialect/Einsum/EinsumTypes.h"
#include "Einsum/Dialect/Einsum/EinsumDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>

using namespace mlir;
using namespace mlir::einsum;

#include "Einsum/Dialect/Einsum/EinsumOpsDialect.cpp.inc"

namespace mlir {
namespace einsum {

struct ContractConfigTypeStorage : public mlir::TypeStorage {
 
  using KeyTy = ::llvm::StringRef;

  /// A constructor for the type storage instance.
  ContractConfigTypeStorage(::llvm::StringRef tree)
      : tree(tree) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const { return key == tree; }

  /// Define a hash function for the key type. This is used when uniquing
  /// instances of the storage, see the `StructType::get` method.
  /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
  /// have hash functions available, so we could just omit this entirely.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  /// Define a construction function for the key type from a set of parameters.
  /// These parameters will be provided when constructing the storage instance
  /// itself.
  /// Note: This method isn't necessary because KeyTy can be directly
  /// constructed with the given parameters.
  static KeyTy getKey(::llvm::StringRef tree) {
    return KeyTy(tree);
  }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
  /// allocations used to create the type storage and its internal.
  static ContractConfigTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    // Copy the elements from the provided `KeyTy` into the allocator.
    ::llvm::StringRef tree = allocator.copyInto(key);

    // Allocate the storage instance and construct it.
    return new (allocator.allocate<ContractConfigTypeStorage>())
        ContractConfigTypeStorage(tree);
  }

  ::llvm::LogicalResult setDims(::llvm::StringRef newDims) {
     dims = newDims;
    return ::mlir::success();
  }

  /// The following field contains the element types of the struct.
  ::llvm::StringRef tree;
  ::llvm::StringRef dims;
};

/// Create an instance of a `StructType` with the given element types. There
/// *must* be at least one element type.
ContractConfigType ContractConfigType::get(::mlir::MLIRContext *ctx, ::llvm::StringRef tree){
   return Base::get(ctx, tree);
}

::llvm::StringRef ContractConfigType::getTree(){
   return getImpl()->tree;
}

::llvm::StringRef ContractConfigType::getDims(){
   return getImpl()->dims;
}

::llvm::LogicalResult ContractConfigType::setDims(::llvm::StringRef newDims){
    return getImpl()->setDims(newDims);
} 
} // namespace einsum
} // namespace mlir

