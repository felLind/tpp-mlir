// RUN: tpp-run %s -n 10 \
// RUN:  -e binary -entry-point-result=void

func.func @binary( %left: memref<48x36x36x36x48xf32>, %right: memref<24x36xf32>, %out: memref<48x36x24x36x48xf32> ) -> memref<48x36x24x36x48xf32>{

	%test = einsum.contract_binary({"dim_sizes"=[48,36,36,48,24,36],"dim_types"=["m","m","m","m","n","k"],
		"strides_left"=[2239488,62208,1728,1,0,48], "strides_right"=[0,0,0,0,36,1], "strides_out"=[1492992,41472,48,1,1728,0],
		"parallel_types"=["seq","shared","shared","prim", "prim", "prim"], "prim_main"="GEMM"},
		%left, %right, %out) : (memref<48x36x36x36x48xf32>, memref<24x36xf32>, memref<48x36x24x36x48xf32>) -> memref<48x36x24x36x48xf32>
	return %out : memref<48x36x24x36x48xf32>
}

