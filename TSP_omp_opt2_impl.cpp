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

#define MAX_DISTANCE 100

#define BUFFER_LEN ((NUM_CITIES * (NUM_CITIES - 1)) / 2)
#define NUM_OPTS (((NUM_CITIES * (NUM_CITIES - 3)) / 2) + 1)

static float cities[BUFFER_LEN];

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
	swap_b = ((swap_a * (swap_a + 1)) / 2) - index;
    ++swap_a;
}

void build_cities(unsigned int seed)
{
	srand(seed);
    using T = std::remove_reference_t<decltype(cities[0])>;
	 
	int i;
	for (i=0; i<BUFFER_LEN; ++i)
	{	
		cities[i] = static_cast<T>((rand() % MAX_DISTANCE) + 1);
	}
}

template <typename T>
T greedy_path_dist(int* path, int initial_idx)
{
	T distance = 0;
	int i;
	
	bool visited_cities[NUM_CITIES] = {0};
	
	path[0] = initial_idx;
	visited_cities[initial_idx] = true;
	
	
	// For every node in the path
	for (i=1; i<NUM_CITIES+1; ++i)
	{		
		if (i != NUM_CITIES)
		{
			T best_dist = MAX_DISTANCE+1;
			int best_idx = 0;
			int j;
			
			// For every possible link
			for (j=0; j<NUM_CITIES; ++j)
			{
				T local_distance = cities[triu_index(path[i-1], j)];

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

		distance += cities[triu_index(path[i-1], path[i])];
	}
	
	return distance;
}

template <typename T>
void opt2(int* current_path)
{

    constexpr std::size_t num_threads = (NUM_OPTS < MAX_THREADS) ? NUM_OPTS : MAX_THREADS;
    omp_set_num_threads(num_threads);

    int new_best_distance, old_best_distance;
    T* f_current_distance = reinterpret_cast<T*>(current_path + NUM_CITIES + 1);
    new_best_distance = *f_current_distance;
    old_best_distance = new_best_distance + 10;
    
    int output[3];
    T* f_output_distance = reinterpret_cast<T*>(output + 2);


    while (new_best_distance < old_best_distance)
    {
        old_best_distance = new_best_distance;
        *f_output_distance = static_cast<T>(0);

        #pragma omp parallel for
        for (int i=0; i<NUM_OPTS; ++i)
        {
            int swap_a, swap_b;
	        calculate_swap_indices(swap_b, swap_a, i);
            //fprintf(stderr, "i = %d, a = %d, b = %d\n", i, swap_a, swap_b);

	        T distance = *f_current_distance;
            float cached_values[8];
            cached_values[0] = cities[triu_index(current_path[swap_b-1], current_path[swap_b])];
            cached_values[1] = cities[triu_index(current_path[swap_b], current_path[swap_b+1])];
            cached_values[2] = cities[triu_index(current_path[swap_a-1], current_path[swap_a])];
            cached_values[3] = cities[triu_index(current_path[swap_a], current_path[swap_a+1])];
            cached_values[4] = cities[triu_index(current_path[swap_b-1], current_path[swap_a])];
            cached_values[5] = cities[triu_index(current_path[swap_a], current_path[swap_b+1])];
            cached_values[6] = cities[triu_index(current_path[swap_a-1], current_path[swap_b])];
            cached_values[7] = cities[triu_index(current_path[swap_b], current_path[swap_a+1])];


            // RECALCULATE DISTANCE:
            // subtract distance from swap_b - 1 to swap_b and from swap_b to swap_b + 1
            // subtract distance from swap_a - 1 to swap_a and from swap_a to swap_a + 1
            // If swap_b + 1 is swap_a and swap_a - 1 is swap_b, subtract 0.
            
            distance -= cached_values[0]
                + cached_values[1]
                    * (swap_b + 1 != swap_a)
                + cached_values[2]
                    * (swap_b + 1 != swap_a)
                + cached_values[3];
            
            // add distance from swap_b - 1 to swap_a and from swap_a to swap_b + 1
            // add distance from swap_a - 1 to swap_b and from swap_b to swap_a + 1
            // If swap_b + 1 is swap_a and swap_a - 1 is swap_b, add 0.
            
            distance += cached_values[4]
                + cached_values[5]
                    * (swap_b + 1 != swap_a)
                + cached_values[6]
                    * (swap_b + 1 != swap_a)
                + cached_values[7];

            #pragma omp critical
            {
                // check if the calculated distance is better than the previously calculated one
                // and better of any other threads' of this iteration
                if (distance < *f_current_distance &&
                    (*f_output_distance == static_cast<T>(0) || distance < *f_output_distance))
                {
                    *f_output_distance = distance;

                    output[0] = swap_b;
                    output[1] = swap_a;

                    /*
                    fprintf(stderr, "[Thread %d] Found better distance: %.1f\nSubtracting values:\
                        swap indices: b=%d, a=%d\n", i, distance, swap_b, swap_a);
                    
                    for (int j=0; j<8; ++j)
                    {
                        fprintf(stderr, "%.1f\t", cached_values[j]);
                    }

                    fprintf(stderr, "\n\n");
                    */
                }
            }
        }

        #pragma omp barrier

        if (*f_output_distance > static_cast<T>(0) && *f_output_distance < new_best_distance)
        {
            new_best_distance = *f_output_distance;
            *f_current_distance = *f_output_distance;

            int& b = output[0];
            int& a = output[1];
            int temp = current_path[b];
            current_path[b] = current_path[a];
            current_path[a] = temp;
        }
    }
}

void print_cities()
{
	fprintf(stdout, "Cities:\n");
	for (int i=0; i<BUFFER_LEN; ++i)
	{
		fprintf(stdout, "%.1f\t", cities[i]);
	}
    fprintf(stdout, "\n");
}

template <typename T>
float verify_result(int* path)
{
	T distance = static_cast<T>(0);

	for (int i=1; i<NUM_CITIES+1; ++i)
	{
		distance += (cities[triu_index(path[i-1], path[i])]);
	}

	return distance;
}

int main(void)
{
    int *current_path;
    struct timespec begin, end;

    current_path = (int*) calloc(NUM_CITIES+2, sizeof(int));
    if (current_path == nullptr)
    {
        perror(strerror(errno));
        return -1;
    }

    build_cities(GENERATION_SEED);

    // print cities
    //print_cities();
    
    float distance = greedy_path_dist<float>(current_path, 0);
    *reinterpret_cast<float*>(current_path+NUM_CITIES+1) = distance;

    fprintf(stdout, "Greedy best distance: %.1f \nGreedy path:\n", distance);
  	
  	for (int i=0; i<NUM_CITIES+1; ++i)
  	{
  		fprintf(stdout, "%d\t", current_path[i]);
  	}
  	fprintf(stdout, "\n");

    clock_gettime(CLOCK_MONOTONIC_RAW, &begin);
    opt2<float>(current_path);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    
    fprintf(stderr, "Total opt2 time = %.9f seconds\n", (end.tv_nsec - begin.tv_nsec) / 1000000000.0 +
            (end.tv_sec  - begin.tv_sec));

    distance = *reinterpret_cast<float*>(current_path + NUM_CITIES+1);

    fprintf(stdout, "Opt2 best distance: %.1f \nOpt2 path:\n", distance);
    for (int i=0; i<NUM_CITIES+1; ++i)
  	{
  		fprintf(stdout, "%d\t", current_path[i]);
  	}
  	fprintf(stdout, "\n");

    float ref_distance = verify_result<float>(current_path);
	printf("Result verification: distance: %.1f --> recalc_distance: %.1f\n", distance, ref_distance);

    free(current_path);
}