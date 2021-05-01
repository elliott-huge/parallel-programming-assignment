// my kernels


// float kernels 
kernel void reduce_add_float(global const float* A, global float* B, local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	// group id used to output index for array B
	int gid = get_group_id(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	// sequential reduce using bit shift
	// inspired from nvidia cuda lecture slides
	// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
	for (int i = N/2; i > 0; i >>= 1) {
		if (lid < i) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	B[gid] = scratch[0];
}

kernel void reduce_minimum_float(global const float* A, global float* B, local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	// group id used to output index for array B
	int gid = get_group_id(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	// sequential reduce using bit shift
	// inspired from nvidia CUDA lecture slides
	// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
	for (int i = N/2; i > 0; i >>= 1) {
		if (lid < i) 
			if (scratch[lid] > scratch[lid + i] && scratch[lid] != 0 && scratch[lid + i] != 0)
				scratch[lid] = scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	B[gid] = scratch[0];
}

kernel void reduce_maximum_float(global const float* A, global float* B, local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	// group id used to output index for array B
	int gid = get_group_id(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	// sequential reduce using bit shift
	// inspired from nvidia CUDA lecture slides
	// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
	for (int i = N/2; i > 0; i >>= 1) {
		if (lid < i) 
			if (scratch[lid] < scratch[lid + i] && scratch[lid] != 0 && scratch[lid + i] != 0)
				scratch[lid] = scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	B[gid] = scratch[0];
}


kernel void map_square_float(global const float* A, global float* B, const float meanVal, const float padValue) {
	int id = get_global_id(0);
	// local memory & barriers not needed since entire operation is a map
	if (A[id] != padValue)
	{
		float inter = A[id]-meanVal;
		B[id] = inter*inter;
	}
	else{
		B[id] = 0.0;
	}
}


// Integer based kernels
/*
kernel void reduce_add(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	// group id used to output index for array B
	int gid = get_group_id(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	// sequential reduce using bit shift
	// inspired from nvidia cuda lecture slides
	// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
	for (int i = N/2; i > 0; i >>= 1) {
		if (lid < i) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	B[gid] = scratch[0];
}

kernel void reduce_minimum(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	// group id used to output index for array B
	int gid = get_group_id(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	// sequential reduce using bit shift
	// inspired from nvidia CUDA lecture slides
	// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
	for (int i = N/2; i > 0; i >>= 1) {
		if (lid < i) 
			if (scratch[lid] > scratch[lid + i] && scratch[lid] != 0 && scratch[lid + i] != 0)
				scratch[lid] = scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	B[gid] = scratch[0];
}

kernel void reduce_maximum(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	// group id used to output index for array B
	int gid = get_group_id(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	// sequential reduce using bit shift
	// inspired from nvidia CUDA lecture slides
	// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
	for (int i = N/2; i > 0; i >>= 1) {
		if (lid < i) 
			if (scratch[lid] < scratch[lid + i] && scratch[lid] != 0 && scratch[lid + i] != 0)
				scratch[lid] = scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	B[gid] = scratch[0];
}

kernel void map_square(global const int* A, global int* B, const int meanVal) {
	int id = get_global_id(0);
	// local memory & barriers not needed since entire operation is a map
	if (A[id] != 0)
	{
		int inter = A[id]-meanVal;
		B[id] = inter*inter;
	}
}
*/



// Sorting Kernel Attempts

/*

void swap(global float* A, global float* B){
	if (*A > *B)
	{
		float temp = *A;
		*A = *B;
		*B = temp;
	}
}


kernel void bubblesort_assignment_float(global float* A, global float* B) {
	int id = get_global_id(0);
	int s = get_global_size(0);


	//int lid = get_local_id(0);
	//int N = get_local_size(0);
	// group id used to output index for array B
	//int gid = get_group_id(0);

	float temp = 0.0;

	if (id % 2)
	{
	
	}
	
	for (int i = 0; i < s; i++)
	{
		if (i % 2 && id < s-1)
		{
			if (A[id] < A[id+1])
			{
				temp = A[id];
				A[id] = A[id+1];
				A[id+1] = temp;
			}
		}
		else if (id < s-2)
		{
			if (A[id+1] < A[id+2])
			{
				temp = A[id+1];
				A[id+1] = A[id+2];
				A[id+2] = temp;
			}
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	B[id] = A[id];
}



kernel void oddeven_assignment_float(global float* A) {
	int id = get_global_id(0);
	int s = get_global_size(0);

	for (int i = 0; i < s; i++)
	{
		if (id % 2 == 0 && (id + 1) < s)
		{
			if (A[id] > A[id+1])
			{
				float temp = A[id];
				A[id] = A[id+1];
				A[id+1] = temp;
			}
		}
		
		barrier(CLK_GLOBAL_MEM_FENCE);
		if (id % 2 == 1 && (id + 1) < s)
		{
			if (A[id] > A[id+1])
			{
				float temp = A[id];
				A[id] = A[id+1];
				A[id+1] = temp;
			}
		}
		
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

kernel void oddeven_assignment_float(global float* A) {
	int id = get_global_id(0);
	int s = get_global_size(0);

	float temp;

	if (id % 2 == 0)
	{
		if (A[id] > A[id+1])
		{
			temp = A[id];
			A[id] = A[id+1];
			A[id+1] = temp;
		}
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
	if (id % 2 == 1)
	{
		if (A[id] > A[id+1])
		{
			temp = A[id];
			A[id] = A[id+1];
			A[id+1] = temp;
		}
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
}

kernel void sort_by_minimum_reduce(global const float* A, global float* B, local float* scratch, global int iter) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	// group id used to output index for array B
	int gid = get_group_id(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id+iter];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	// sequential reduce using bit shift
	// inspired from nvidia CUDA lecture slides
	// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
	for (int i = N/2; i > 0; i >>= 1) {
		if (lid < i && i > iter) 
			if (scratch[lid] > scratch[lid + i])
				scratch[lid] = scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	B[gid] = scratch[0];
}
*/