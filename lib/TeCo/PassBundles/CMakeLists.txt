get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_mlir_library(TeCoPassBundles
  LocalDialectsLowering.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/TECO

  DEPENDS
    TeCoCompilerPassIncGen
    TeCoCompilerPassBundleIncGen

  LINK_LIBS PUBLIC
    MLIRIR
   # TeCoTransforms
    ${mlir_dialect_libs}
    ${conversion_libs}
  )
