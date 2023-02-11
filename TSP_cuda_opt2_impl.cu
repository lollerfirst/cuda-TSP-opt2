#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <cuda_profiler_api.h>

#define NUM_CITIES 25
#define MAX_DISTANCE 32767
#define MEM_ALIGNMENT 32
#define BLOCK_SIZE 512

#define BUFFER_LEN ((NUM_CITIES * (NUM_CITIES - 1)) / 2)
#define ALIGNED_UNIT_SIZE ((((sizeof(int) * NUM_CITIES + sizeof(int)) / MEM_ALIGNMENT) + 1) * MEM_ALIGNMENT)
#define NUM_OPTS (((NUM_CITIES * (NUM_CITIES - 3)) / 2) + 1)

// Distance matrix represented in compressed form
static int cities[BUFFER_LEN];

// device data
__constant__ int device_cities[BUFFER_LEN];

// indexes the compressed sparse matrix holding the distances
__inline__ __host__ __device__ int triu_index(const int i, const int j)
{
	const int side_i = NUM_CITIES - (i+1);
	const int side_j = NUM_CITIES - (j+1);
	const int sub_value_i = side_i*(side_i+1)/2;
	const int sub_value_j = side_j*(side_j+1)/2;

	return ((BUFFER_LEN - sub_value_i) + j - i - 1 * (i < j)) +
		((BUFFER_LEN - sub_value_j) + i - j - 1 * (j < i));
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
__global__ void cuda_calculate_opts(int* memory_block,
	bool switched_pointers,
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
	int swap_b = 0;
	int accumulator = NUM_CITIES - 2;
	int range = accumulator;
	
	while (tid / accumulator > 0)
	{
		
		++swap_b;
		
		if (swap_b >= NUM_CITIES - 2)
		{
			return;
		}
		
		range = accumulator;
		accumulator += (NUM_CITIES - 2) - swap_b;
	}
	
	// warp-scoped branching causes a mess without it
	__syncwarp();
	

	int swap_bin[2];
	
	
	int swap_a = swap_b + (tid % range) + 1;
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
	
	// Block-wide sync
	__syncthreads();

	// Acquire the lock
	while (trylock(&lock) == false);
	
	if (distance < current_path[NUM_CITIES])
	{
		output_path[NUM_CITIES] = distance;
	
		constexpr int num_blocks_copy = (NUM_CITIES + BLOCK_SIZE + 1) / BLOCK_SIZE;
		copy<<<num_blocks_copy, BLOCK_SIZE>>>(output_path, current_path, NUM_CITIES);
		
		output_path[swap_a] = swap_bin[1];
		output_path[swap_b] = swap_bin[0];
	}	
	
	// Release the lock
	unlock(&lock);
	__syncwarp();
	
	return;
}

// Control thread
__global__ void cuda_opt2(int* memory_block, int initial_idx)
{

	constexpr int num_blocks = (NUM_OPTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
	constexpr int num_blocks_copy = ((ALIGNED_UNIT_SIZE/sizeof(int)) + BLOCK_SIZE + 1) / BLOCK_SIZE;
	
	int new_best_dist = memory_block[NUM_CITIES];
	int old_best_dist = new_best_dist + 1;
	bool switched_pointers = false;
	
	int best_index = -1;
	
	while (new_best_dist < old_best_dist) 
	{
		// THIS SWITCH CUTS OFF THE UNNECESSARY COPY KERNEL CALL AT THE BEGINNING
		switch (best_index)
		{
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
				memory_block,
				switched_pointers,
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
				new_best_dist = (calc_distance < new_best_dist) ? calc_distance : new_best_dist;
				best_index = (calc_distance < new_best_dist) ? i : best_index; 
			}
		}
	}

	
	return;
}


int main(void)
{
	// Build the data structure
	build_cities(1);

	// Errors
	cudaError_t err_code;

	// Copy cities from host to device, cities is costant
	err_code = cudaMemcpyToSymbol(device_cities,
		cities,
		sizeof(int) * (BUFFER_LEN)
	);

	if (err_code)
	{
		printf("[!] Cuda Error at line %d: %s\n", __LINE__, cudaGetErrorName(err_code));
		return -1;
	}

	// Calculate number of blocks necessary
	constexpr int num_blocks = (NUM_OPTS + BLOCK_SIZE - 1) / BLOCK_SIZE;

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
        

  	// Start the profiler
    	cudaProfilerStart();

  	// Call the control thread
  	cuda_opt2<<<1, 1>>>(memory_block, 0);        
	
	// Wait for the GPU to finish
  	cudaDeviceSynchronize();

    	// Stop the profiler
    	cudaProfilerStop();
    
  	err_code = cudaGetLastError();
  	
  	if (err_code)
        {
                printf("[!] Cuda Error at line %d: %s\n", __LINE__, cudaGetErrorName(err_code));
                return -1;
        }
        
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
