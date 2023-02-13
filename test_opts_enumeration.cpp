#include <cstdio>
#include <cmath>

#define NUM_CITIES 10
#define NUM_OPTS (((NUM_CITIES*(NUM_CITIES-3))/2)+1)

inline void calculate_swap_indices(int& swap_b, int& swap_a, const int index)
{
	swap_b = static_cast<int>((1.0f + sqrt(static_cast<double>(1+8*index))) / 2.0f);
	swap_a = ((swap_b * (swap_b + 1)) / 2) - index - 1;
}

int main(void)
{
    for (int i=0; i<NUM_OPTS; ++i)
    {
        int a, b;
        calculate_swap_indices(b, a, i);
        fprintf(stdout, "i = %d, b = %d, a = %d\n", i, a, b);
    }
}