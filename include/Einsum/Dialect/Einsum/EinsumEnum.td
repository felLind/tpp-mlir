
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "Einsum/Dialect/Einsum/EinsumDialect.td"


def Einsum_DimType: I64EnumAttr<
    "DimType", "dimension type",
    [
      I64EnumAttrCase<"M",  1, "m">,
      I64EnumAttrCase<"N", 2, "n">,
      I64EnumAttrCase<"K", 3, "k">,
      I64EnumAttrCase<"B", 4, "complex">,
      I64EnumAttrCase<"C",52, "batch">
    ]>{
   let cppNamespace = "mlir::einsum";
}
