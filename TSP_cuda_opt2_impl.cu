#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

#define NUM_CITIES 10
#define MAX_DISTANCE 32767
#define BLOCK_SIZE 256
#define NUM_OPTS (((NUM_CITIES * (NUM_CITIES - 3)) / 2) + 1)
#define MEM_ALIGNMENT 32

// distances matrix, maybe embedded in the future
static int cities[NUM_CITIES * NUM_CITIES];

// device data
__constant__ int device_cities[NUM_CITIES * NUM_CITIES];

__device__ int lock;		// Justification: have to be visible
__device__ int sync_var;	// to all threads in a grid = only way

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
__global__ void cuda_calculate_opts(
	int* current_path,
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
	

	int swap_bin[2];
	
	
	size_t i;
	int swap_a = swap_b + (tid % range) + 1;
	int distance = 0;
	int previous_idx = initial_idx;
	
	// Load only the nodes to swap
	swap_bin[1] = current_path[swap_b];
	swap_bin[0] = current_path[swap_a];
	
	
	// Recalculate distance
	for (i=0; i<NUM_CITIES; ++i)
	{
		if (i == swap_b)
		{
			distance += device_cities[(previous_idx*NUM_CITIES) + swap_bin[0]];
			previous_idx = swap_bin[0];
		}
		else if (i == swap_a)
		{
			distance += device_cities[(previous_idx*NUM_CITIES) + swap_bin[1]];
			previous_idx = swap_bin[1];
		}
		else
		{
			distance += device_cities[(previous_idx*NUM_CITIES) + current_path[i]];
			previous_idx = current_path[i];
		}
		
	}

	// Inter block thread synchronization i came up with
	inter_block_sync(&sync_var);
	

	// Acquire the lock
	while (trylock(&lock) == false);
	
	if (distance >= current_path[NUM_CITIES])
	{
		unlock(&lock);
		return;
	}
	
	current_path[NUM_CITIES] = distance;
	
	// Adjust current path
	current_path[swap_a] = swap_bin[1];
	current_path[swap_b] = swap_bin[0];	
	
	// Release the lock
	unlock(&lock);
	
	return;
}

// Control thread
__global__ void cuda_opt2(int* memory_block, int num_blocks, int initial_idx)
{
	
	int old_best_dist;
	int new_best_dist;
	do
	{
		old_best_dist = *(memory_block + NUM_CITIES);
		
		// Sort out sync vars initialization
		lock = 0;
		sync_var = 0;
			
		// Dynamic Parallelism CUDA 5.0+
		cuda_calculate_opts<<<num_blocks, BLOCK_SIZE>>>(
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
		
		new_best_dist = *(memory_block + NUM_CITIES);
	}
	while (new_best_dist < old_best_dist);
	
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
	
	// Calculate the length of a unit
	size_t memory_block_size = sizeof(int) * NUM_CITIES + sizeof(int);

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

  	
  	// Call the control thread
  	cuda_opt2<<<1, 1>>>(memory_block, num_blocks, 0);        

	
	// Wait for the GPU to finish
  	cudaDeviceSynchronize();
  	err_code = cudaGetLastError();
  	
  	if (err_code)
        {
                printf("[!] Cuda Error at line %d: %s\n", __LINE__, cudaGetErrorName(err_code));
                return -1;
        }
        
        
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

  	
  	cudaFree(memory_block);
  	
  	size_t i;
  	printf("Best Distance: %d\n", best_dist);
  	puts("Path: ");
  	
  	for (i=0; i<NUM_CITIES; ++i)
  	{
  		printf("%d\t", current_path[i]);
  	}
  	 	 	
  	return 0;
}

