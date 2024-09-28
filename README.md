# Leveraging-CUDA-for-Efficient-Genetic-Algorithms-in-High-Dimensional-Spaces

Introduction
In the field of optimization, Genetic Algorithms (GAs) are a powerful class of evolutionary algorithms inspired by the principles of natural selection. They are particularly effective in solving complex problems where traditional optimization techniques may struggle. This project aims to implement a CUDA-accelerated Genetic Algorithm to optimize high-dimensional functions efficiently, leveraging the parallel processing capabilities of modern GPUs.

The workflow of this project can be summarized in the following steps:

Initialization: A population of candidate solutions (individuals) is generated randomly within a defined search space. Each individual represents a potential solution to the optimization problem. The size of the population and the number of dimensions for each individual can be adjusted based on the complexity of the problem.

Fitness Evaluation: The fitness of each individual in the population is evaluated using a fitness function. In this project, a high-dimensional sphere function is utilized as the optimization objective, where the goal is to minimize the function value. This evaluation process is performed in parallel using CUDA kernels, allowing for significant speedup by leveraging the GPU’s computational power.

Selection: Individuals are selected from the population based on their fitness values. The selection process favors better-performing individuals, ensuring that their genetic information is passed to the next generation. This mimics natural selection, where the fittest individuals are more likely to reproduce.

Crossover: The crossover operator combines pairs of selected individuals to create new offspring. This process introduces genetic diversity and allows for the exploration of new regions in the search space. The crossover rate can be adjusted to balance exploration and exploitation in the optimization process.

Mutation: To further enhance genetic diversity and prevent premature convergence, mutation is applied to the offspring. Random changes are introduced to some individuals, ensuring a broader search of the solution space. The mutation rate can be fine-tuned based on the problem characteristics.

Iteration: The process of evaluating fitness, selecting individuals, performing crossover, and mutating continues over multiple generations. Each generation aims to produce a population with better fitness values than the previous one. The best fitness values across generations are recorded for analysis.

Results Analysis: The performance of the Genetic Algorithm is evaluated by plotting the best fitness values over generations. Insights are drawn from the visualizations to assess convergence behavior and optimization efficiency. Parameter adjustments can be made to improve the algorithm’s performance based on these insights.

By utilizing CUDA for parallel processing, this project aims to enhance the speed and efficiency of the Genetic Algorithm, making it suitable for tackling high-dimensional optimization problems. This approach not only demonstrates the capabilities of GPU acceleration but also provides a foundation for further exploration into more complex optimization challenges.

**Step 1: Set Up Google Colab for CUDA
Enable GPU Support:**
Go to Runtime in the menu.
Select Change runtime type.
In the "Hardware accelerator" dropdown, choose GPU.
Click Save.
**Step 2: Install CUDA Toolkit**
You will need to install the necessary CUDA toolkit and dependencies in your Colab environment. Add and run the following code in a new cell:
**Step 3: Write GA Code **
%%writefile ga_cuda.cu
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>

// Function to optimize (High-Dimensional Sphere Function)
__device__ float fitness_function(float* individual, int dimensions) {
    float fitness = 0.0f;
    for (int i = 0; i < dimensions; ++i) {
        fitness += individual[i] * individual[i];  // Example: Sphere function
    }
    return fitness;
}

// CUDA kernel to evaluate the fitness of each individual in the population
__global__ void evaluate_population(float* population, float* fitness_values, int population_size, int dimensions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < population_size) {
        fitness_values[idx] = fitness_function(&population[idx * dimensions], dimensions);
    }
}

// CUDA kernel for crossover
__global__ void crossover(float* population, float* new_population, int population_size, int dimensions, float crossover_rate, curandState* rand_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < population_size) {
        int parent1_idx = idx * 2;
        int parent2_idx = parent1_idx + 1;

        for (int i = 0; i < dimensions; ++i) {
            float r = curand_uniform(&rand_states[idx]);
            if (r < crossover_rate) {
                new_population[idx * dimensions + i] = 0.5f * (population[parent1_idx * dimensions + i] + population[parent2_idx * dimensions + i]);
            } else {
                new_population[idx * dimensions + i] = population[parent1_idx * dimensions + i];
            }
        }
    }
}

// CUDA kernel for mutation
__global__ void mutate(float* population, int population_size, int dimensions, float mutation_rate, curandState* rand_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < population_size) {
        for (int i = 0; i < dimensions; ++i) {
            float r = curand_uniform(&rand_states[idx]);
            if (r < mutation_rate) {
                population[idx * dimensions + i] += curand_normal(&rand_states[idx]) * 0.1f;
            }
        }
    }
}

// CUDA kernel to initialize random states
__global__ void init_rand_states(curandState* rand_states, int population_size, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < population_size) {
        curand_init(seed, idx, 0, &rand_states[idx]);
    }
}

int main() {
    const int population_size = 1024;
    const int dimensions = 100;  // High-dimensional space
    const int generations = 1000;
    const float crossover_rate = 0.7f;
    const float mutation_rate = 0.01f;

    // Allocate memory for the population and fitness values
    float* population;
    float* new_population;
    float* fitness_values;
    curandState* rand_states;

    cudaMalloc(&population, population_size * dimensions * sizeof(float));
    cudaMalloc(&new_population, population_size * dimensions * sizeof(float));
    cudaMalloc(&fitness_values, population_size * sizeof(float));
    cudaMalloc(&rand_states, population_size * sizeof(curandState));

    // Initialize random states for CUDA
    init_rand_states<<<(population_size + 255) / 256, 256>>>(rand_states, population_size, time(0));

    // Open file to save fitness values in CSV format
    std::ofstream fitnessFile("/content/fitness_values.csv");
    fitnessFile << "Generation,Best Fitness\n"; // CSV header

    // Main Genetic Algorithm loop
    for (int gen = 0; gen < generations; ++gen) {
        // Evaluate the fitness of the population
        evaluate_population<<<(population_size + 255) / 256, 256>>>(population, fitness_values, population_size, dimensions);
        cudaDeviceSynchronize();

        // Perform crossover
        crossover<<<(population_size + 255) / 256, 256>>>(population, new_population, population_size / 2, dimensions, crossover_rate, rand_states);
        cudaDeviceSynchronize();

        // Mutate the new population
        mutate<<<(population_size + 255) / 256, 256>>>(new_population, population_size, dimensions, mutation_rate, rand_states);
        cudaDeviceSynchronize();

        // Write the best fitness to the file
        float best_fitness = INFINITY;
        float* h_fitness_values = new float[population_size];
        cudaMemcpy(h_fitness_values, fitness_values, population_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < population_size; ++i) {
            if (h_fitness_values[i] < best_fitness) {
                best_fitness = h_fitness_values[i];
            }
        }

        fitnessFile << gen << "," << best_fitness << "\n"; // Write generation and best fitness
        delete[] h_fitness_values;

        // Swap populations
        std::swap(population, new_population);
    }

    // Cleanup
    fitnessFile.close();
    cudaFree(population);
    cudaFree(new_population);
    cudaFree(fitness_values);
    cudaFree(rand_states);

    return 0;
}
**Step 4: Compile the CUDA Code**
Run the following command in a new cell to compile the CUDA code:

!nvcc ga_cuda.cu -o ga_cuda
**Step 5: Run the CUDA Executable**
After compiling, run the following command to execute the program:

!./ga_cuda
**Step 6: Check Output**
Once the execution is complete, check the generated CSV file for fitness values. You can use the following command to list the files:

!ls /content/

By visualizing these aspects of genetic algorithm's performance,developers can gain insights into its behavior, efficiency, and effectiveness. The chosen visualizations can also help communicate our results effectively to others, making it easier to understand the optimization process and outcomes.


**To adjust the parameters in CUDA Genetic Algorithm (GA) implementation and to reevaluate the performance after making these changes, follow the steps below.**

Step 1: Adjust Parameters
Population Size: You can increase or decrease the population size to see how it affects convergence. Larger populations may provide better exploration but will take more computation time.
Crossover Rate: Modify the crossover rate to find a balance between exploration (high crossover rates) and exploitation (low crossover rates).
Mutation Rate: Adjust the mutation rate to control how often individuals change. A higher mutation rate can introduce more diversity but might disrupt convergence.
Generations: Change the number of generations based on how long you want to run the algorithm.

**Example Workflow**
Initial Run: Use the original parameters and save the results.
Adjust Parameters: Change a specific parameter (e.g., increase population size).
Re-run: Execute the modified code.
Analyze and Compare: Evaluate the fitness values and plot the results. Look for improvements in the convergence rate and best fitness.
Iterate: Continue adjusting parameters and reevaluating until you find the optimal configuration.

