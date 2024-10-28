add_mlir_dialect_library(TeCoTeCoDialect
    # Ops and dialects
  #  BufferizableOpInterfaceImpl.cpp
    TeCoDialect.cpp
    TeCoOps.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Standalone

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
)

target_include_directories(TeCoTeCoDialect
  PUBLIC
#  $<BUILD_INTERFACE:${TPP_GEN_INCLUDE_DIR}>
#  $<BUILD_INTERFACE:${TPP_MAIN_INCLUDE_DIR}>
)
