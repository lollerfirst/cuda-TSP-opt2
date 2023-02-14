#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <cuda_profiler_api.h>

#ifndef GENERATION_SEED
#define GENERATION_SEED 1
#endif

#ifndef NUM_CITIES
#define NUM_CITIES 100
#endif

#define MAX_DISTANCE 32767
#define MEM_ALIGNMENT 32
#define BLOCK_SIZE 1024

#define BUFFER_LEN ((NUM_CITIES * (NUM_CITIES - 1)) / 2)
#define ALIGNED_UNIT_SIZE ((((sizeof(int) * NUM_CITIES + sizeof(int)) / MEM_ALIGNMENT) + 1) * MEM_ALIGNMENT)
#define NUM_OPTS (((NUM_CITIES * (NUM_CITIES - 3)) / 2) + 1)

// Distance matrix represented in compressed form
static int cities[BUFFER_LEN];

// indexes the compressed sparse matrix holding the distances
__inline__ __host__ __device__ int triu_index(const int i, const int j)
{
	const int side_i = NUM_CITIES - (i+1);
	const int side_j = NUM_CITIES - (j+1);
	const int sub_area_i = side_i*(side_i+1)/2;
	const int sub_area_j = side_j*(side_j+1)/2;

	return (((BUFFER_LEN - sub_area_i) + j - i - 1)* (i < j)) +
		(((BUFFER_LEN - sub_area_j) + i - j - 1) * (j < i));
}

// find greatest swap_a such that f(swap_a) <= index, with f(n) = n*(n+1)/2
// find swap_b as f(swap_a) - index - 1
__inline__ __device__ void calculate_swap_indices(int* swap_b, int* swap_a, const int index)
{
	*swap_a = static_cast<int>(floorf((1.0f + sqrtf(static_cast<float>(1+8*index))) / 2.0f));
	*swap_b = ((swap_a[0] * (swap_a[0] + 1)) / 2) - index - 1;
}

// build the data structure on the host
void build_cities(unsigned int seed)
{
	srand(seed);
	 
	int i;
	for (i=0; i<BUFFER_LEN; ++i)
	{	
		cities[i] = rand() % MAX_DISTANCE;
	}
}

int greedy_path_dist(int* path, int initial_idx)
{
	int distance = 0;
	size_t i;
	int idx = initial_idx;
	
	bool visited_cities[NUM_CITIES] = {0};
	
	// Can't choose the initial before having ended the tour
	visited_cities[initial_idx] = true;
	
	
	// For every node in the path
	for (i=0; i<NUM_CITIES; ++i)
	{		
		if (i != NUM_CITIES - 1)
		{
			int best_dist = INT_MAX;
			int best_idx = -1;
			size_t j;
			
			// For every possible link
			for (j=0; j<NUM_CITIES; ++j)
			{
				if (idx != j &&
					cities[triu_index(idx, j)] <= best_dist &&
					!visited_cities[j])
				{
					best_dist = cities[triu_index(idx, j)];
					best_idx = j;
				}
			}
			
			if (best_idx == -1)
			{
				return -1;
			}
			
			visited_cities[best_idx] = true;
			path[i] = best_idx;
			idx = best_idx;
			distance += best_dist;
		}
		else
		{
			// LAST MUST BE the initial idx
			path[i] = initial_idx;
			distance += cities[triu_index(idx, initial_idx)];
		}
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

template <typename TYPE>
__global__ void copy(TYPE* dest, const TYPE* src, size_t count)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid < count)
	{
		dest[tid] = src[tid];
	}
	
	return;
}

// Worker threads
__global__ void cuda_calculate_opts(int* device_cities,
	int* memory_block,
	int initial_idx)
{

	// Thread identification
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	
	// Boundaries control
	if (tid >= NUM_OPTS)
	{
		return;
	}
	
	__shared__ int lock;
	lock = 0;
	
	int* current_path = memory_block;
		
	int* output_path = memory_block + (ALIGNED_UNIT_SIZE / sizeof(int)) * (blockIdx.x + 1);
	
	// ENUMERATION OF OPTS:
	int swap_a, swap_b;
	calculate_swap_indices(&swap_b, &swap_a, tid);
	

	int swap_bin[2];
	int distance = current_path[NUM_CITIES];
	
	// Load only the nodes to swap
	swap_bin[1] = current_path[swap_b];
	swap_bin[0] = current_path[swap_a];
	
	
	// RECALCULATE DISTANCE:
	// subtract distance from swap_b - 1 to swap_b and from swap_b to swap_b + 1
	// subtract distance from swap_a - 1 to swap_a and from swap_a to swap_a + 1
	// If swap_b + 1 is swap_a and swap_a - 1 is swap_b, subtract 0.
	distance -= (swap_b > 0) ? device_cities[triu_index(current_path[swap_b-1], current_path[swap_b])] : 0;
	
	distance -= device_cities[triu_index(initial_idx, current_path[swap_b])]
		* (swap_b == 0);
	distance -= device_cities[triu_index(current_path[swap_b], current_path[swap_b+1])]
		* (swap_b + 1 != swap_a);
	distance -= device_cities[triu_index(current_path[swap_a-1], current_path[swap_a])]
		* (swap_a - 1 != swap_b);
	distance -= device_cities[triu_index(current_path[swap_a], current_path[swap_a+1])];
	
	// add distance from swap_b - 1 to swap_a and from swap_a to swap_b + 1
	// add distance from swap_a - 1 to swap_b and from swap_b to swap_a + 1
	// If swap_b + 1 is swap_a and swap_a - 1 is swap_b, add 0.
	distance += (swap_b > 0) ? device_cities[triu_index(current_path[swap_b-1], current_path[swap_a])] : 0;
	
	distance += device_cities[triu_index(initial_idx, current_path[swap_a])]
		* (swap_b == 0);
	distance += device_cities[triu_index(current_path[swap_a], current_path[swap_b+1])]
		* (swap_b + 1 != swap_a);
	distance += device_cities[triu_index(current_path[swap_a-1], current_path[swap_b])]
		* (swap_a - 1 != swap_b);
	distance += device_cities[triu_index(current_path[swap_b], current_path[swap_a+1])];
	
	// initialize calculated distance with 0 ~~ default value/not improved
	output_path[NUM_CITIES] = 0;

	// Block-wide sync
	__syncthreads();

	// Acquire the lock
	while (trylock(&lock) == false);
	
	if (distance < current_path[NUM_CITIES] &&
		(output_path[NUM_CITIES] == 0 || distance < output_path[NUM_CITIES]))
	{
		output_path[NUM_CITIES] = distance;
	
		const int num_blocks_copy = (NUM_CITIES + BLOCK_SIZE - 1) / BLOCK_SIZE;
		copy<<<num_blocks_copy, BLOCK_SIZE>>>(output_path, current_path, NUM_CITIES);
		cudaDeviceSynchronize();
		
		output_path[swap_a] = swap_bin[1];
		output_path[swap_b] = swap_bin[0];
	}	
	
	// Release the lock
	unlock(&lock);
	__syncwarp();
	
	return;
}

// Control thread
__global__ void cuda_opt2(int* device_cities, int* memory_block, int initial_idx)
{

	const int num_blocks = (NUM_OPTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
	const int num_blocks_copy = ((ALIGNED_UNIT_SIZE/sizeof(int)) + BLOCK_SIZE - 1) / BLOCK_SIZE;
	
	int new_best_dist = memory_block[NUM_CITIES];
	int old_best_dist = new_best_dist + 1;
	int best_index = -1;
	
	switch (best_index)
	{
		while (new_best_dist < old_best_dist) 
		{
			// THIS SWITCH CUTS OFF THE UNNECESSARY COPY KERNEL CALL AT THE BEGINNING
			
			default:
				// Copy best path and distance to current path
				copy<<<num_blocks_copy, BLOCK_SIZE>>>(
					memory_block,
					&memory_block[(ALIGNED_UNIT_SIZE/sizeof(int)) * best_index],
					ALIGNED_UNIT_SIZE / sizeof(int)
				);
				
				// Wait for child grid to terminate
				cudaDeviceSynchronize();
			
			case -1:
				
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

				cudaError_t err_code = cudaGetLastError();
				if (err_code)
				{
					return;
				}

				// retrieve best calculated distance amongst various blocks
				for (int i=1; i<num_blocks+1; ++i)
				{
					int calc_distance = memory_block[(ALIGNED_UNIT_SIZE/sizeof(int)) * i + NUM_CITIES];
					if (calc_distance > 0 && calc_distance < new_best_dist)
					{
						new_best_dist = calc_distance;
						best_index = i; 
					}
				}
		}
	}

	
	return;
}


int main(void)
{
	int* device_cities;
	struct timespec begin, end;

	// Build the data structure
	build_cities(GENERATION_SEED);

	// Errors
	cudaError_t err_code;

	// Allocate device cities
	err_code = cudaMalloc(&device_cities, BUFFER_LEN * sizeof(int));

	if (err_code)
	{
		printf("[!] Cuda Error at line %d: %s\n", __LINE__, cudaGetErrorName(err_code));
		return -1;
	}

	// Copy cities from host to device, cities is costant
	err_code = cudaMemcpy(device_cities,
		cities,
		sizeof(int) * (BUFFER_LEN),
		cudaMemcpyHostToDevice
	);

	if (err_code)
	{
		printf("[!] Cuda Error at line %d: %s\n", __LINE__, cudaGetErrorName(err_code));
		return -1;
	}

	// Calculate number of blocks necessary
	const int num_blocks = (NUM_OPTS + BLOCK_SIZE - 1) / BLOCK_SIZE;

	int* memory_block;

	
	// Calculate memory block size
	size_t memory_block_size = ALIGNED_UNIT_SIZE * (num_blocks+1);

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
	int current_path[NUM_CITIES];
	int best_dist = greedy_path_dist(current_path, 0);
	
	printf("Greedy best Distance: %d\n", best_dist);
  	puts("Greedy path: ");
  	
  	for (int i=0; i<NUM_CITIES; ++i)
  	{
  		printf("%d\t", current_path[i]);
  	}
  	printf("\n");
    	
	err_code = cudaMemcpy(memory_block,
		current_path,
		sizeof(int) * NUM_CITIES,
		cudaMemcpyHostToDevice
	);
    	
    	if (err_code)
	{
        	printf("[!] Cuda Error at line %d: %s\n", __LINE__, cudaGetErrorName(err_code));
        	return -1;
	}
		
  	err_code = cudaMemcpy(memory_block + NUM_CITIES,
  		&best_dist,
  		sizeof(int),
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
  		memory_block + NUM_CITIES,
  		sizeof(int),
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
  		sizeof(int) * (NUM_CITIES),
  		cudaMemcpyDeviceToHost
  	);

  	
  	printf("Best Distance: %d\n", best_dist);
  	puts("Path: ");
  	
  	for (int i=0; i<NUM_CITIES; ++i)
  	{
  		printf("%d\t", current_path[i]);
  	}
  	printf("\n");
  	
  	cudaFree(memory_block);
  	return 0;
}
