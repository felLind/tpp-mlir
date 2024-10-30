add_mlir_conversion_library(LinalgXLinalgXToLoops
  ConvertLinalgXToLoops.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/LinalgX

  DEPENDS
  LinalgXCompilerPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  LinalgXLinalgXDialect
  MLIRSCFDialect
  MLIRMemRefDialect
  )
