#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

#define NUM_CITIES 25
#define MAX_DISTANCE 32767
#define BLOCK_SIZE 256
#define NUM_OPTS (((NUM_CITIES * (NUM_CITIES - 3)) / 2) + 1)
#define MEM_ALIGNMENT 32

// distances matrix, maybe embedded in the future
static int cities[NUM_CITIES * NUM_CITIES];

// device data
__constant__ int device_cities[NUM_CITIES * NUM_CITIES];

__device__ __align__(32) int lock;	// Justification: have to be visible
__device__ __align__(32) int sync_var;	// to all threads in a grid = only way

// build the data structure on the host
void build_cities(void)
{
	srand(time(NULL));
	 
	int i;
	for (i=0; i<NUM_CITIES; ++i)
	{	
		int j;
		for (j=0; j<NUM_CITIES; ++j)
		{
			if (j == i)
			{
				continue;
			}
			
			cities[(i * NUM_CITIES) + j] = rand() % MAX_DISTANCE;
		}
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
					cities[(idx*NUM_CITIES) + j] <= best_dist &&
					!visited_cities[j])
				{
					best_dist = cities[(idx*NUM_CITIES) + j];
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
			distance += cities[(idx*NUM_CITIES) + initial_idx];
		}
	}
	
	return distance;
}

__inline__ __device__ void inter_block_sync(int* sync_var)
{
	// Inter block synchronization
	atomicAdd(sync_var, 1);
	
	
	while (*sync_var != NUM_OPTS);
	
	return;
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

// Worker threads
__global__ void cuda_calculate_opts(int* current_path,
	int* output_path,
	int initial_idx)
{

	// Thread identification
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	
	
	// Boundaries control
	if (tid >= NUM_OPTS)
	{
		return;
	}
	
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
	int distance = 0;
	int previous_idx = initial_idx;
	
	// Load only the nodes to swap
	swap_bin[1] = current_path[swap_b];
	swap_bin[0] = current_path[swap_a];
	
	
	// Recalculate distance
	// Maybe substitution with another kernel launch?
	for (int i=0; i<NUM_CITIES; ++i)
	{
		// Avoid branching with these instructions
		distance += device_cities[(previous_idx*NUM_CITIES) + current_path[i]]
			* (i != swap_a && i != swap_b)
			+ device_cities[(previous_idx*NUM_CITIES) + swap_bin[0]]
			* (i == swap_b)
			+ device_cities[(previous_idx*NUM_CITIES) + swap_bin[1]]
			* (i == swap_a);
			
		previous_idx = current_path[i] * (i != swap_a && i != swap_b)
			+ swap_bin[0] * (i == swap_b)
			+ swap_bin[1] * (i == swap_a);
	}

	// Inter block thread synchronization i came up with
	inter_block_sync(&sync_var);

	// Acquire the lock
	while (trylock(&lock) == false);
	
	if (distance < current_path[NUM_CITIES])
	{
		output_path[NUM_CITIES] = distance;
		current_path[NUM_CITIES] = distance;
	
		// Maybe substitution with another kernel launch?
		for (int j=0; j<NUM_CITIES; ++j)
		{
			output_path[j] = current_path[j];
		}

		output_path[swap_a] = swap_bin[1];
		output_path[swap_b] = swap_bin[0];
	}	
	
	// Release the lock
	unlock(&lock);
	__syncwarp();
	
	return;
}

// Control thread
__global__ void cuda_opt2(int* current_path, int* output_path, int num_blocks, int initial_idx)
{
	
	int old_best_dist;
	int new_best_dist;
	int final_mask = 0x80000000;
	
	do
	{
		old_best_dist = *(current_path + NUM_CITIES);
		final_mask ^= 0x80000000;
		
		// Sort out sync vars initialization
		lock = 0;
		sync_var = 0;
			
		// Dynamic Parallelism CUDA 5.0+
		cuda_calculate_opts<<<num_blocks, BLOCK_SIZE>>>(
			current_path,
			output_path,
			initial_idx
		);
		
		// Wait for child grid to terminate
		cudaDeviceSynchronize();
		
		cudaError_t err_code = cudaGetLastError();
		if (err_code)
		{
			return;
		}
		
		new_best_dist = *(output_path + NUM_CITIES);
		
		// switch-up pointers for the next iteration
		// because the output path becomes the current_path in case
		// there is another iteration
		
		int* temp = current_path;
		current_path = output_path;
		output_path = temp;
	}
	while (new_best_dist < old_best_dist);
	
	// Embed an initial 1 bit if the best path is on the second unit
	current_path[NUM_CITIES] |= final_mask;
	return;
}


int main(void)
{
	// Build the data structure
	build_cities();

	// Errors
	cudaError_t err_code;

	// Copy cities from host to device, cities is costant
	err_code = cudaMemcpyToSymbol(device_cities,
		cities,
		sizeof(int) * (NUM_CITIES * NUM_CITIES)
	);

	if (err_code)
	{
		printf("[!] Cuda Error at line %d: %s\n", __LINE__, cudaGetErrorName(err_code));
		return -1;
	}

	// Calculate number of blocks necessary
	const int num_blocks = (NUM_OPTS + BLOCK_SIZE - 1) / BLOCK_SIZE;

	int* memory_block;
	
	// Calculate the length of a unit: 1 path + 1 total_distance
	size_t aligned_unit_size = sizeof(int) * NUM_CITIES + sizeof(int);
	aligned_unit_size = ((aligned_unit_size / MEM_ALIGNMENT) + 1) * MEM_ALIGNMENT;
	
	// Calculate memory block size
	size_t memory_block_size = aligned_unit_size * 2;

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
        
        // Get output path ptr
        int* output_path = (int*) (((uintptr_t) memory_block) + aligned_unit_size);

  	
  	// Call the control thread
  	cuda_opt2<<<1, 1>>>(memory_block, output_path, num_blocks, 0);        
	
	
	// Wait for the GPU to finish
  	cudaDeviceSynchronize();
  	err_code = cudaGetLastError();
  	
  	if (err_code)
        {
                printf("[!] Cuda Error at line %d: %s\n", __LINE__, cudaGetErrorName(err_code));
                return -1;
        }
        
        // Copy best distance from GPU
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
        

	// Extract the first bit of best distance to understand which unit
	// contains the best path
  	if (!(best_dist & 0x80000000))
  	{
	  	
	  	cudaMemcpy(current_path,
	  		memory_block,
	  		sizeof(int) * (NUM_CITIES),
	  		cudaMemcpyDeviceToHost
	  	);

		if (err_code)
		{
		        printf("[!] Cuda Error at line %d: %s\n", __LINE__, cudaGetErrorName(err_code));
		        return -1;
		}
	}
	else
	{
		cudaMemcpy(current_path,
	  		output_path,
	  		sizeof(int) * (NUM_CITIES),
	  		cudaMemcpyDeviceToHost
	  	);

		if (err_code)
		{
		        printf("[!] Cuda Error at line %d: %s\n", __LINE__, cudaGetErrorName(err_code));
		        return -1;
		}
		
		best_dist &= 0x7FFFFFFF;
	}
  	

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
