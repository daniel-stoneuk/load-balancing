#include <stdlib.h>
#include <time.h>
#include <algorithm>

extern "C"
{
    int *K_Greedy(int balls, int bins, int k)
    {
        srand(time(NULL));
        int *load = new int[bins];
        for (int bin = 0; bin < bins; bin++)
        {
            load[bin] = 0;
        }

        for (int ball = 0; ball < balls; ball++)
        {
            int lowestBin = rand() % bins;
            for (int i = 1; i < k; i++)
            {
                int choice = rand() % bins;
                if (load[choice] < load[lowestBin])
                {
                    lowestBin = choice;
                }
            }
            load[lowestBin] += 1;
        }
        return std::max_element(load, load + bins);
    }
}

int main()
{
}
