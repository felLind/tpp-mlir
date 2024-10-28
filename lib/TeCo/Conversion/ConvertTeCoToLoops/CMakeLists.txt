add_mlir_conversion_library(TeCoTeCoToLoops
  ConvertTeCoToLoops.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TeCo

  DEPENDS
  TeCoCompilerPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  TeCoTeCoDialect
  MLIRSCFDialect
  MLIRMemRefDialect
  )
