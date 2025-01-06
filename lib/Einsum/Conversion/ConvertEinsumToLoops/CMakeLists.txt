add_mlir_conversion_library(EinsumEinsumToLoops
  ConvertEinsumToLoops.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Einsum

  DEPENDS
  EinsumCompilerPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  EinsumEinsumDialect
  MLIRSCFDialect
  MLIRMemRefDialect
  )
