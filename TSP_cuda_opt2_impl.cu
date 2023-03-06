#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <mma.h>

#ifndef GENERATION_SEED
#define GENERATION_SEED 1
#endif

#ifndef NUM_CITIES
#define NUM_CITIES 100
#endif

#define MAX_DISTANCE 3267
#define MEM_ALIGNMENT 32
#define BLOCK_SIZE 512
#define WARP_SIZE 32
#define STRIDE 16

#define BUFFER_LEN ((NUM_CITIES * (NUM_CITIES - 1)) / 2)
#define NUM_OPTS (((NUM_CITIES * (NUM_CITIES - 3)) / 2) + 1)

#define ALIGN(__X) ((((__X) / MEM_ALIGNMENT) + 1) * MEM_ALIGNMENT)

// Distance matrix represented in compressed form
static half cities[BUFFER_LEN];

// indexes the compressed sparse matrix holding the distances
__inline__ __host__ __device__ int triu_index(const int i, const int j)
{
	const int side_i = NUM_CITIES - (i+1);
	const int side_j = NUM_CITIES - (j+1);
	const int sub_area_i = side_i*(side_i+1)/2;
	const int sub_area_j = side_j*(side_j+1)/2;

	return (((BUFFER_LEN - sub_area_i) + j - i - 1) * (i < j)) +
		(((BUFFER_LEN - sub_area_j) + i - j - 1) * (j < i));
}

// find greatest swap_a such that f(swap_a) <= index, with f(n) = n*(n+1)/2
// find swap_b as f(swap_a) - index - 1
__inline__ __device__ void calculate_swap_indices(int* swap_b, int* swap_a, const int index)
{
	*swap_a = __float2int_rd((1.0f + sqrtf(static_cast<float>(1+8*index))) / 2.0f);
	*swap_b = ((swap_a[0] * (swap_a[0] + 1)) / 2) - index;
	++(*swap_a);
}

// build the data structure on the host
void build_cities(unsigned int seed)
{
	srand(seed);
	 
	int i;
	for (i=0; i<BUFFER_LEN; ++i)
	{	

		cities[i] = __float2half_rd(static_cast<float>((rand() % MAX_DISTANCE)));
	}
}

float greedy_path_dist(int* path, int initial_idx)
{
	float distance = 0.0f;
	int i;
	
	bool visited_cities[NUM_CITIES] = {0};
	
	path[0] = initial_idx;
	visited_cities[initial_idx] = true;
	
	
	// For every node in the path
	for (i=1; i<NUM_CITIES+1; ++i)
	{		
		if (i != NUM_CITIES)
		{
			float best_dist = MAX_DISTANCE+1;
			int best_idx = 0;
			int j;
			
			// For every possible link
			for (j=0; j<NUM_CITIES; ++j)
			{
				float local_distance = __half2float(cities[triu_index(path[i-1], j)]);

				if (path[i-1] != j &&
					local_distance <= best_dist &&
					!visited_cities[j])
				{
					best_dist = local_distance;
					best_idx = j;
				}
			}
			
			visited_cities[best_idx] = true;
			path[i] = best_idx;
		}
		else
		{
			path[i] = initial_idx;
		}

		distance += __half2float(cities[triu_index(path[i-1], path[i])]);
	}
	
	return distance;
}

__inline__ __device__ bool trylock(int* mutex)
{
	// Aquire the lock with an atomic compare exchange
	int old = atomicCAS(mutex, 0, 1);
	
	if (old == 1)
	{
		return false;
	}
	else
	{
		return true;
	}
}

__inline__ __device__ void unlock(int* mutex)
{
	// Release the lock with an atomic exchange
	(void) atomicExch(mutex, 0);
}

struct __align__(32) PackedMemory
{
	float arr[BLOCK_SIZE * STRIDE];
	int lock;
};

// Loads matrix A with -(b-1 -> b), -(b -> b+1), -(a-1 -> a), -(a -> a+1)
// (b-1 -> a), (a -> b+1), (a-1 -> b), (b -> a+1)
__device__ __inline__ void load_matrix_a(half* A, half* device_cities, int* current_path, int swap_a, int swap_b)
{
	A[(threadIdx.x % (WARP_SIZE/2)) * STRIDE] = device_cities[triu_index(current_path[swap_b-1], current_path[swap_b])];
	A[(threadIdx.x % (WARP_SIZE/2)) * STRIDE + 1] = device_cities[triu_index(current_path[swap_b], current_path[swap_b+1])];
	A[(threadIdx.x % (WARP_SIZE/2)) * STRIDE + 2] = device_cities[triu_index(current_path[swap_a-1], current_path[swap_a])];
	A[(threadIdx.x % (WARP_SIZE/2)) * STRIDE + 3] = device_cities[triu_index(current_path[swap_a], current_path[swap_a+1])];
	A[(threadIdx.x % (WARP_SIZE/2)) * STRIDE + 4] = device_cities[triu_index(current_path[swap_b-1], current_path[swap_a])];
	A[(threadIdx.x % (WARP_SIZE/2)) * STRIDE + 5] = device_cities[triu_index(current_path[swap_a], current_path[swap_b+1])];
	A[(threadIdx.x % (WARP_SIZE/2)) * STRIDE + 6] = device_cities[triu_index(current_path[swap_a-1], current_path[swap_b])];
	A[(threadIdx.x % (WARP_SIZE/2)) * STRIDE + 7] = device_cities[triu_index(current_path[swap_b], current_path[swap_a+1])];

    // convert to negatives the first 4 values
    long* tmp = reinterpret_cast<long*>(A + (threadIdx.x % (WARP_SIZE/2)) * STRIDE);
    *tmp |= 0x8000800080008000;

	#pragma unroll
	for (int i=STRIDE/2; i<STRIDE; ++i)
	{
		A[(threadIdx.x % (WARP_SIZE/2)) * STRIDE + i] = 0x0000;
	}
}

// Worker threads
__global__ void cuda_calculate_opts(
	half* device_cities,
	int* memory,
	int initial_idx)
{

	// Constants we need to calculate pointers further on
	const int aligned_unit = ALIGN(sizeof(int) * 2 + sizeof(float)) / sizeof(int);
	const int start_unit = ALIGN(sizeof(int) * (NUM_CITIES+1) + sizeof(float)) / sizeof(int);

	// Lane identification
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	__shared__ PackedMemory block_mem;

	float* C = block_mem.arr + ((threadIdx.x / WARP_SIZE) * (WARP_SIZE/2)) * STRIDE;
	half* A = reinterpret_cast<half*>(C);
	int& lock = block_mem.lock;
	
	int swap_a, swap_b;
	float distance;

	// Fragments  
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;

	// Get pointers to current path and current distance
	int* current_path = memory;
	float* f_current_distance = reinterpret_cast<float*>(current_path) + NUM_CITIES+1;
	
	// Get pointers to output indices and output distance
	int* output = memory + start_unit + aligned_unit * blockIdx.x;
	float* f_output_distance = reinterpret_cast<float*>(output) + 2;
	
	distance = *f_current_distance;

	// Calculate the swap indices for this lane:
	calculate_swap_indices(&swap_b, &swap_a, tid);

	if (threadIdx.x % (WARP_SIZE/2) < 16 && tid < NUM_OPTS)
	{
		// Load A
		load_matrix_a(A, device_cities, current_path, swap_a, swap_b);
	}
	
	// Load tensor core registers: first half of warp
	nvcuda::wmma::load_matrix_sync(a_frag, A, STRIDE);
	nvcuda::wmma::fill_fragment(c_frag, 0.0f);

	// Perform fused multiply add
	nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
	
	// Store result into C
	nvcuda::wmma::store_matrix_sync(C, c_frag, STRIDE, nvcuda::wmma::mem_row_major);

	// Each thread reads his own cell of the diagonal of C
	distance += ((threadIdx.x % WARP_SIZE) < WARP_SIZE/2) * C[(threadIdx.x % (WARP_SIZE/2)) * STRIDE];
	
	if (threadIdx.x % (WARP_SIZE/2) >= 16 && tid < NUM_OPTS)
	{
		// Reload A
		load_matrix_a(A, device_cities, current_path, swap_a, swap_b);
	}

	// Load tensor core registers: second half of warp.
	nvcuda::wmma::load_matrix_sync(a_frag, A + (WARP_SIZE/2) * STRIDE, STRIDE);

	// Perform matrix multiplication
	nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

	// Store result into B
	nvcuda::wmma::store_matrix_sync(C, c_frag, STRIDE, nvcuda::wmma::mem_row_major);

	// Each thread reads his own cell of the diagonal of C
	distance += ((threadIdx.x % WARP_SIZE) >= WARP_SIZE / 2) * C[(threadIdx.x % (WARP_SIZE/2)) * STRIDE];

	if (tid >= NUM_OPTS)
	{
		return;
	}

	// Shuffle down to the first lane of the warp all the values of distance and swap indices.
	// Each lane is receiving and passing down messages to another lane and evaluating whether to update its own
	#pragma unroll
	for (int offset=WARP_SIZE/2; offset > 0; offset /= 2)
	{
		int alt_swap_b, alt_swap_a;
		float alt_distance;
		alt_distance = __shfl_down_sync(__activemask(), distance, offset, WARP_SIZE);
		alt_swap_b = __shfl_down_sync(__activemask(), swap_b, offset, WARP_SIZE);
		alt_swap_a = __shfl_down_sync(__activemask(), swap_a, offset, WARP_SIZE);

		// update distance if it's lower
		swap_b = (alt_distance < distance && alt_distance != 0.0f) * alt_swap_b + (alt_distance >= distance || alt_distance == 0.0f) * swap_b;
		swap_a = (alt_distance < distance && alt_distance != 0.0f) * alt_swap_a + (alt_distance >= distance || alt_distance == 0.0f) * swap_a;
		distance = (alt_distance < distance && alt_distance != 0.0f) * alt_distance + (alt_distance >= distance || alt_distance == 0.0f) * distance;
	}
	
	// initialize calculated distance with 0 ~~ default value/not improved
	*f_output_distance = 0.0f;

	// Block-wide sync
	__syncthreads();

	// Only the leading lane updates the value ~~ decrease dramatically number of mutually exclusive accesses
	if (threadIdx.x % WARP_SIZE == 0)
	{
		while (trylock(&lock) == false);

		if (distance < (*f_current_distance) &&
			((*f_output_distance) == 0.0f || distance < (*f_output_distance)))
		{
			*f_output_distance = distance;

			output[0] = swap_b;
			output[1] = swap_a;
		}	

		// Release the lock
		unlock(&lock);
	}
	
	return;
}

// Control thread
__global__ void cuda_opt2(__half* device_cities, int* memory_block, int initial_idx)
{

	const int num_blocks = (NUM_OPTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
	const int aligned_unit = ALIGN(sizeof(int) * 2 + sizeof(float)) / sizeof(int);
	const int start_unit = ALIGN(sizeof(int) * (NUM_CITIES+1) + sizeof(float)) / sizeof(int);

	float* f_memory_ptr = reinterpret_cast<float*>(memory_block);

	float new_best_dist = f_memory_ptr[NUM_CITIES + 1];
	float old_best_dist = new_best_dist + 10.0f;
	
	while (new_best_dist < old_best_dist) 
	{

		// save previous calculated distance
		old_best_dist = new_best_dist;

		// Launch kernel that computes paths and distances
		cuda_calculate_opts<<<num_blocks, BLOCK_SIZE>>>(
			device_cities,
			memory_block,
			initial_idx
		);

		// Wait for child grid to terminate
		cudaDeviceSynchronize();

		cudaError_t err_code = cudaPeekAtLastError();
		assert(err_code == cudaSuccess);

		// retrieve best calculated distance amongst various blocks
		int best_index = -1;
		for (int i=0; i<num_blocks; ++i)
		{
			float calc_distance = f_memory_ptr[start_unit + aligned_unit * i + 2];
			if (calc_distance > 0.0f && calc_distance < new_best_dist)
			{
				new_best_dist = calc_distance;
				best_index = i; 
			}
		}

		if (best_index != -1)
		{
			// apply the swap
			int swap_b = memory_block[start_unit + aligned_unit * best_index];
			int swap_a = memory_block[start_unit + aligned_unit * best_index + 1];
			
			int temp = memory_block[swap_a];
			memory_block[swap_a] = memory_block[swap_b];
			memory_block[swap_b] = temp;

			f_memory_ptr[NUM_CITIES+1] = new_best_dist;
		}
	}

	
	return;
}

void print_cities()
{
	fprintf(stdout, "Cities:\n");
	for (int i=0; i<BUFFER_LEN; ++i)
	{
		fprintf(stdout, "%f\t", __half2float(cities[i]));
	}
}

int main(void)
{
	__half* device_cities;
	struct timespec begin, end;
	int* memory_block;

	// Errors
	cudaError_t err_code;

	// Build the data structure
	build_cities(GENERATION_SEED);

	// Print cities
	print_cities();

	// Allocate device cities
	err_code = cudaMalloc(&device_cities, BUFFER_LEN * sizeof(__half));

	if (err_code)
	{
		printf("[!] Cuda Error at line %d: %s\n", __LINE__, cudaGetErrorName(err_code));
		return -1;
	}

	// Copy cities from host to device, cities is costant
	err_code = cudaMemcpy(device_cities,
		cities,
		sizeof(__half) * (BUFFER_LEN),
		cudaMemcpyHostToDevice
	);

	if (err_code)
	{
		printf("[!] Cuda Error at line %d: %s\n", __LINE__, cudaGetErrorName(err_code));
		return -1;
	}

	// Calculate number of blocks necessary
	const int num_blocks = (NUM_OPTS + BLOCK_SIZE - 1) / BLOCK_SIZE;

	
	// Calculate memory block size:
	// current_path + float_distance + (2 * swap_indices + float_distance) * number_of_blocks
	const size_t memory_block_size = ALIGN(sizeof(int) * (NUM_CITIES+1) + sizeof(float)) + 
		ALIGN(sizeof(int) * 2 + sizeof(float)) * num_blocks;

	// Allocate memory
	cudaMalloc((void**) &memory_block,
	    memory_block_size
	);
	
	if (err_code)
	{
		printf("[!] Cuda Error at line %d: %s\n", __LINE__, cudaGetErrorName(err_code));
		return -1;
	}


	// initial path chosen with a greedy heuristic, stored in current_path
	int current_path[NUM_CITIES+1];
	float best_dist = greedy_path_dist(current_path, 0);
	
	printf("Greedy best Distance: %f\n", best_dist);
  	puts("Greedy path: ");
  	
  	for (int i=0; i<NUM_CITIES+1; ++i)
  	{
  		printf("%d\t", current_path[i]);
  	}
  	printf("\n");
    	
	err_code = cudaMemcpy(memory_block,
		current_path,
		sizeof(int) * (NUM_CITIES+1),
		cudaMemcpyHostToDevice
	);
    	
    if (err_code)
	{
		printf("[!] Cuda Error at line %d: %s\n", __LINE__, cudaGetErrorName(err_code));
		return -1;
	}
		
  	err_code = cudaMemcpy(memory_block + (NUM_CITIES+1),
  		&best_dist,
  		sizeof(float),
  		cudaMemcpyHostToDevice
  	);

	if (err_code)
	{
		printf("[!] Cuda Error at line %d: %s\n", __LINE__, cudaGetErrorName(err_code));
		return -1;
	}
        

	clock_gettime(CLOCK_MONOTONIC_RAW, &begin);

  	// Call the control thread
  	cuda_opt2<<<1, 1>>>(device_cities, memory_block, 0);        
	
	// Wait for the GPU to finish
  	cudaDeviceSynchronize();

	clock_gettime(CLOCK_MONOTONIC_RAW, &end);


  	err_code = cudaGetLastError();
  	
  	if (err_code)
	{
		printf("[!] Cuda Error at line %d: %s\n", __LINE__, cudaGetErrorName(err_code));
		return -1;
	}
        

	fprintf (stderr, "Total GPU time = %.9f seconds\n",
            (end.tv_nsec - begin.tv_nsec) / 1000000000.0 +
            (end.tv_sec  - begin.tv_sec));

    // Copy best distance from GPU into best_dist
  	err_code = cudaMemcpy(&best_dist,
  		memory_block + (NUM_CITIES+1),
  		sizeof(float),
  		cudaMemcpyDeviceToHost
  	);

	if (err_code)
	{
		printf("[!] Cuda Error at line %d: %s\n", __LINE__, cudaGetErrorName(err_code));
		return -1;
	}
        
	// Copy the best path from GPU into current_path
	cudaMemcpy(current_path,
		memory_block,
		sizeof(int) * (NUM_CITIES+1),
		cudaMemcpyDeviceToHost
  	);

  	
  	printf("Best Distance: %f\n", best_dist);
  	puts("Path: ");
  	
  	for (int i=0; i<NUM_CITIES+1; ++i)
  	{
  		printf("%d\t", current_path[i]);
  	}
  	printf("\n");
 	cudaFree(memory_block);
  	return 0;
}
