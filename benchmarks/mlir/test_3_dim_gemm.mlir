// RUN: tpp-run %s -n 5 \
// RUN:  -e binary -entry-point-result=void

func.func @binary( %left: memref<4x4x4xf32>, %right: memref<4x4x4xf32>, %out: memref<4x4xf32> ) -> memref<4x4xf32>{

	%test = einsum.contract_binary({dim_names=["d","a","b","c"],"dim_sizes"=[4,4,4,4],"dim_types"=["k","m","n","k"],
		"dims_left"=["d", "c", "a"], "dims_right"=["d", "b", "c"], "dims_out"=["a","b"],
		"strides_left"=[16,1,0, 4], "strides_right"=[16,0,4,1], "strides_out"=[0,1,4,0],
		"parallel_types"=["shared","prim", "prim", "prim"], "prim_main"="GEMM"},
		%left, %right, %out) : (memref<4x4x4xf32>, memref<4x4x4xf32>, memref<4x4xf32>) -> memref<4x4xf32>
	return %out : memref<4x4xf32>
}

