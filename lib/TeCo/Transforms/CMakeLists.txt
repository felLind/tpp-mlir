
add_mlir_library(TeCoTransforms
 
  DEPENDS
   TeCoCompilerPassIncGen

  LINK_LIBS PUBLIC
    ${mlir_dialect_libs}
    MLIRIR
    MLIRSupport
    MLIRBufferizationToMemRef
    MLIRBufferizationPipelines
    TeCoTeCoToLoops
  )
