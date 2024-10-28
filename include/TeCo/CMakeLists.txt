add_subdirectory(Dialect)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TeCoCompiler)
add_public_tablegen_target(TeCoCompilerPassIncGen)
add_mlir_doc(Passes TeCoCompilerPasses ./ -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS PassBundles.td)
mlir_tablegen(PassBundles.h.inc -gen-pass-decls -name TeCoPassBundle)
add_public_tablegen_target(TeCoCompilerPassBundleIncGen)
add_mlir_doc(PassBundles TeCoCompilerPassBundles ./ -gen-pass-doc)