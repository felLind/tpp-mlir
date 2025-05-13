add_mlir_dialect(EinsumOps einsum)
add_mlir_doc(EinsumDialect EinsumDialect Einsum/ -gen-dialect-doc)
add_mlir_doc(EinsumOps EinsumOps Einsum/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS EinsumEnum.td)
mlir_tablegen(EinsumEnum.h.inc -gen-enum-decls)
mlir_tablegen(EinsumEnum.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIREinsumAttrDefIncGen)