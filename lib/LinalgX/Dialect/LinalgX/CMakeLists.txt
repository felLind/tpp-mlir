add_mlir_dialect_library(LinalgXLinalgXDialect
    # Ops and dialects
  #  BufferizableOpInterfaceImpl.cpp
    LinalgXDialect.cpp
    LinalgXOps.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Standalone

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
)

target_include_directories(LinalgXLinalgXDialect
  PUBLIC
#  $<BUILD_INTERFACE:${TPP_GEN_INCLUDE_DIR}>
#  $<BUILD_INTERFACE:${TPP_MAIN_INCLUDE_DIR}>
)
