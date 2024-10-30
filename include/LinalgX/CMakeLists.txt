add_subdirectory(Dialect)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name LinalgXCompiler)
add_public_tablegen_target(LinalgXCompilerPassIncGen)
add_mlir_doc(Passes LinalgXCompilerPasses ./ -gen-pass-doc)
