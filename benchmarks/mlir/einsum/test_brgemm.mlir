// RUN: tpp-run %s -n 5 \
// RUN:  -e binary -entry-point-result=void

func.func @binary(%left: memref<96x96x84x84xf32>, %right: memref<96x84x84xf32>, %out: memref<96x84x84xf32>) -> memref<96x84x84xf32>{
	%test = einsum.contract_binary ({dim_names=["a","e","c","b","d"],"dim_sizes"=[96,96,84,84,84],"dim_types"=["m","k","m","n","k"],
		"dims_left"=["a","e","d","c"], "dims_right"=["e", "b", "d"], "dims_out"=["a","b","c"],
		"strides_left"=[677376,7056,1,0,84], "strides_right"=[0,7056,0,84,1], "strides_out"=[7056,0,1,84,0],
		"parallel_types"=["shared", "prim", "prim", "prim", "prim"],"prim_main"="BRGEMM"},
		%left, %right, %out) : (memref<96x96x84x84xf32>, memref<96x84x84xf32>, memref<96x84x84xf32>) -> memref<96x84x84xf32>
	return %out : memref<96x84x84xf32>
}

