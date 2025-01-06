add_subdirectory(Dialect)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name EinsumCompiler)
add_public_tablegen_target(EinsumCompilerPassIncGen)
add_mlir_doc(Passes EinsumCompilerPasses ./ -gen-pass-doc)
