add_mlir_dialect_library(EinsumEinsumDialect
    # Ops and dialects
  #  BufferizableOpInterfaceImpl.cpp
    EinsumEnum.cpp
    EinsumDialect.cpp
    EinsumOps.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Standalone

  DEPENDS
    # add_mlir_dialect macro force-prefixes with MLIR
    MLIREinsumAttrDefIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
)

target_include_directories(EinsumEinsumDialect
  PUBLIC
#  $<BUILD_INTERFACE:${TPP_GEN_INCLUDE_DIR}>
#  $<BUILD_INTERFACE:${TPP_MAIN_INCLUDE_DIR}>
)
