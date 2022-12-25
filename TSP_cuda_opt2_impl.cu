#include <time.h>
#include <stdlib.h>
#include <stdbool.h>

#define NUM_CITIES 10
#define MAX_LENGTH 32767
#define BLOCK_SIZE 256
#define NUM_OPTS (((NUM_CITIES * (NUM_CITIES - 3)) / 2) + 1)

// distances matrix
static int cities[NUM_CITIES * NUM_CITIES];

// device data
__constant__ int device_cities[NUM_CITIES * NUM_CITIES];
__device__ short lock;

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
	
	// can't choose the initial before having ended the tour
	visited_cities[initial_idx] = true;
	
	for (i=0; i<NUM_CITIES; ++i)
	{		
		if (i != NUM_CITIES - 1)
		{
			int best_dist = INT_MAX;
			int best_idx = -1;
			size_t j;
			
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

__device__ bool trylock(short* mutex)
{
	// Aquire the lock with an atomic compare exchange
	short old = atomicCAS(mutex, 0, 1);
	
	if (old == 1)
	{
		return false;
	}
	else
	{
		return true;
	}
}

__device__ void unlock(short* mutex)
{
	// Release the lock with an atomic exchange
	atomicExch(mutex, 0);
}

// Worker threads
__global__ void cuda_calculate_opts(
	int* opts_path,
	int* current_path,
	int* opts_best_dist,
	int initial_idx)
{
	// Thread identification
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	
	// Boundaries control
	if (tid >= NUM_OPTS)
	{
		return;
	}
	
	// Calculate the swapping range for this thread
	int swap_range = 0;
	int accumulator = NUM_CITIES - 2;
	while (tid / accumulator > 0)
	{
		++swap_range;
		
		if (swap_range >= NUM_CITIES - 2)
		{
			return;
		}
		
		accumulator += (NUM_CITIES - 2) - swap_range;
	}
	
	opts_path = opts_path + (tid * NUM_CITIES);
	
	// Find a way to univokely change 2 nodes in the path
	size_t i;
	int to_swap = swap_range + (tid % accumulator);
	int distance = 0;
	int previous_idx = initial_idx;
	
	for (i=0; i<NUM_CITIES; ++i)
	{
		opts_path[i] = current_path[i];
		
		
		if (to_swap == i)
		{
			int temp = opts_path[swap_range];
			opts_path[swap_range] =  opts_path[to_swap];
			opts_path[to_swap] = temp;
		}
		
		distance += device_cities[(previous_idx*NUM_CITIES) + opts_path[i]];
		previous_idx = opts_path[i];
	}
	
	
	// Acquire the lock
	while (trylock(&lock) == false);
	
	if (distance >= (*opts_best_dist))
	{
		unlock(&lock);
		return;
	}
	
	*opts_best_dist = distance;
	
	// Copy the path calculated in this thread
	for (i=0; i<NUM_CITIES; ++i)
	{
		current_path[i] = opts_path[i];
	}
	
	// Release the lock
	unlock(&lock);
}

// Control thread
__global__ void cuda_opt2(int* opts_paths,
	int* current_path,
	int* opts_best_dist,
	int initial_idx)
{
  	
  	// set the block and grid sizes
	const int num_blocks = (NUM_OPTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
	
	int best_dist;
	
	do
	{
		best_dist = *opts_best_dist;
		
		// Dynamic Parallelism CUDA 5.0+
		cuda_calculate_opts<<<num_blocks, BLOCK_SIZE>>>(
			opts_paths,
			current_path,
			opts_best_dist,
			initial_idx
		);
		
	}
	while ((*opts_best_dist) < best_dist);
	
}


int main(void)
{
	// Build the data structure
	build_cities();

	// Copy cities from host to device, cities is costant
	cudaMemcpyToSymbol(device_cities,
		cities,
		sizeof(int) * (NUM_CITIES * NUM_CITIES)
	);
	
	// contigous memory where different calculated paths will be stored
	int* opts_paths;
	cudaMalloc((void**) &opts_paths,
	
		// opts + current_path + device_best_dist
		sizeof(int) * (NUM_OPTS + 1) * (NUM_CITIES) + sizeof(int)
	);
	
    	// initial path chosen with a greedy heuristic, stored in current_path
    	int current_path[NUM_CITIES];
    	int best_dist = greedy_path_dist(current_path);
    	
    	// copy current_path to the device
    	int* device_current_path = opts_paths + NUM_OPTS * (NUM_CITIES + 1);
    	cudaMemcpy(device_current_path,
    		current_path,
    		sizeof(int) * (NUM_CITIES),
    		cudaMemcpyHostToDevice
    	); 
    	
    	// allocate memory for device_best_dist
	int* device_best_dist = opts_paths + (NUM_OPTS + 1) * (NUM_CITIES);
  	cudaMemcpy(device_best_dist,
  		&best_dist,
  		sizeof(int),
  		cudaMemcpyHostToDevice
  	);
  	
  	// Call the control thread
  	cuda_opt2<<<1, 1>>>(opts_paths, device_current_path, device_best_dist);
  	
  	// device_best_dist now contains the best distance found
  	// device_current_path contains the best path
  	cudaMemcpy(&best_dist,
  		device_best_dist,
  		sizeof(int),
  		cudaMemcpyDeviceToHost
  	);
  	
  	cudaMemcpy(current_path,
  		device_current_path,
  		sizeof(int) * (NUM_CITIES),
  		cudaMemcpyDeviceToHost
  	);
  	
  	cudaFree(opts_paths);
  	
  	size_t i;
  	printf("Best Distance: %d\n", best_distance);
  	puts("Path: ");
  	
  	for (i=0; i<NUM_CITIES; ++i)
  	{
  		printf("%d\t", current_path[i]);
  	}
  	 	 	
  	return 0;
}
