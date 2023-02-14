#include <omp.h>
#include <cstddef>
#include <climits>
#include <iostream>
#include <ctime>
#include <cmath>
#include <random>
#include <mutex>
#include <cerrno>
#include <cstring>

#ifndef GENERATION_SEED
#define GENERATION_SEED 1
#endif

#ifndef MAX_THREADS
#define MAX_THREADS 64
#endif

#ifndef NUM_CITIES
#define NUM_CITIES 100
#endif

#define MAX_DISTANCE 32767

#define BUFFER_LEN ((NUM_CITIES * (NUM_CITIES - 1)) / 2)
#define NUM_OPTS (((NUM_CITIES * (NUM_CITIES - 3)) / 2) + 1)

static int cities[BUFFER_LEN];

inline int triu_index(const int i, const int j)
{
	const auto side_i = NUM_CITIES - (i+1);
	const auto side_j = NUM_CITIES - (j+1);
	const auto sub_area_i = side_i*(side_i+1)/2;
	const auto sub_area_j = side_j*(side_j+1)/2;

	return (((BUFFER_LEN - sub_area_i) + j - i - 1)* (i < j)) +
		(((BUFFER_LEN - sub_area_j) + i - j - 1) * (j < i));
}

inline void calculate_swap_indices(int& swap_b, int& swap_a, const int index)
{
	swap_a = static_cast<int>((1.0f + sqrt(static_cast<double>(1+8*index))) / 2.0f);
	swap_b = ((swap_a * (swap_a + 1)) / 2) - index - 1;
}


void build_cities(unsigned int seed)
{
	srand(seed);
	 
	std::size_t i;
	for (i=0; i<BUFFER_LEN; ++i)
	{	
		cities[i] = rand() % MAX_DISTANCE;
	}
}

int greedy_path_dist(int* path, int initial_idx)
{
	int distance = 0;
	int i;
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
			int best_idx = INT_MAX;
			int j;
			
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
			
			if (best_idx == INT_MAX)
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

void opt2(int* current_path, int* output_path, const int initial_idx)
{

    constexpr std::size_t num_threads = (NUM_OPTS < MAX_THREADS) ? NUM_OPTS : MAX_THREADS;
    omp_set_num_threads(num_threads);

    int* still_memory_pointer = current_path;
    int new_best_distance, old_best_distance;
    new_best_distance = current_path[NUM_CITIES];
    old_best_distance = new_best_distance+1;
    int final_mask = 0x80000000;


    while (new_best_distance < old_best_distance)
    {
        final_mask ^= 0x80000000;
        old_best_distance = new_best_distance;
        output_path[NUM_CITIES] = 0;

        #pragma omp parallel for
        for (int i=0; i<NUM_OPTS; ++i)
        {
            int swap_a, swap_b;
	        calculate_swap_indices(swap_b, swap_a, i);
            //fprintf(stderr, "i = %d, a = %d, b = %d\n", i, swap_a, swap_b);

            int swap_bin[2];
	        int distance = current_path[NUM_CITIES];
            swap_bin[1] = current_path[swap_b];
            swap_bin[0] = current_path[swap_a];

            // RECALCULATE DISTANCE:
            // subtract distance from swap_b - 1 to swap_b and from swap_b to swap_b + 1
            // subtract distance from swap_a - 1 to swap_a and from swap_a to swap_a + 1
            // If swap_b + 1 is swap_a and swap_a - 1 is swap_b, subtract 0.
            distance -= (swap_b > 0) ? cities[triu_index(current_path[swap_b-1], current_path[swap_b])] : 0;
            
            distance -= cities[triu_index(initial_idx, current_path[swap_b])]
                * (swap_b == 0);
            distance -= cities[triu_index(current_path[swap_b], current_path[swap_b+1])]
                * (swap_b + 1 != swap_a);
            distance -= cities[triu_index(current_path[swap_a-1], current_path[swap_a])]
                * (swap_a - 1 != swap_b);
            distance -= cities[triu_index(current_path[swap_a], current_path[swap_a+1])];
            
            // add distance from swap_b - 1 to swap_a and from swap_a to swap_b + 1
            // add distance from swap_a - 1 to swap_b and from swap_b to swap_a + 1
            // If swap_b + 1 is swap_a and swap_a - 1 is swap_b, add 0.
            distance += (swap_b > 0) ? cities[triu_index(current_path[swap_b-1], current_path[swap_a])] : 0;
            
            distance += cities[triu_index(initial_idx, current_path[swap_a])]
                * (swap_b == 0);
            distance += cities[triu_index(current_path[swap_a], current_path[swap_b+1])]
                * (swap_b + 1 != swap_a);
            distance += cities[triu_index(current_path[swap_a-1], current_path[swap_b])]
                * (swap_a - 1 != swap_b);
            distance += cities[triu_index(current_path[swap_b], current_path[swap_a+1])];

            #pragma omp critical
            {
                // check if the calculated distance is better than the previously calculated one
                // and better of any other threads' of this iteration
                if (distance < current_path[NUM_CITIES] &&
                    (output_path[NUM_CITIES] == 0 || distance < output_path[NUM_CITIES]))
                {
                    output_path[NUM_CITIES] = distance;

                    memcpy(output_path, current_path, NUM_CITIES*sizeof(int));
                    output_path[swap_b] = swap_bin[0];
                    output_path[swap_a] = swap_bin[1];
                }
            }
        }

        #pragma omp barrier

        new_best_distance = (output_path[NUM_CITIES] > 0) ? output_path[NUM_CITIES] : new_best_distance;
        int* temp = current_path;
        current_path = output_path;
        output_path = temp;
    }

    // embed initial 1 bit in the distance if the result is on the output_path
    still_memory_pointer[NUM_CITIES] |= final_mask;
}

int main(void)
{
    int *current_path, *output_path;
    struct timespec begin, end;

    current_path = (int*) calloc(NUM_CITIES+1, sizeof(int));
    if (current_path == nullptr)
    {
        perror(strerror(errno));
        return -1;
    }

    output_path = (int*) calloc(NUM_CITIES+1, sizeof(int));
    if (output_path == nullptr)
    {
        free(current_path);
        perror(strerror(errno));
        return -1;
    }

    build_cities(GENERATION_SEED);
    int distance = greedy_path_dist(current_path, 0);
    current_path[NUM_CITIES] = distance;

    std::cout << "Greedy best distance: " << distance << "\nGreedy path:\n";
  	
  	for (std::size_t i=0; i<NUM_CITIES; ++i)
  	{
  		std::cout << current_path[i] << "\n";
  	}
  	std::cout << "\n";

    clock_gettime(CLOCK_MONOTONIC_RAW, &begin);
    opt2(current_path, output_path, 0);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    
    std::cerr << "Total opt2 time = " << (end.tv_nsec - begin.tv_nsec) / 1000000000.0 +
            (end.tv_sec  - begin.tv_sec) << " seconds\n";

    int* selected = (current_path[NUM_CITIES] & 0x80000000) ? output_path : current_path;

    distance = selected[NUM_CITIES];

    std::cout << "Opt-2 best distance: " << distance << "\nOpt-2 path:\n";
    for (std::size_t i=0; i<NUM_CITIES; ++i)
    {
        std::cout << selected[i] << "\n";
    }
    std::cout << "\n";

    free(output_path);
    free(current_path);
}