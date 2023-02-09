/**
 * @file test_optim.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-02-09
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <functional>

#include "fmt/format.h"
#include "fmt/ostream.h"
#include "fmt/ranges.h"

#include "SimulatedAnneal.hpp"


// Optimize simple square function

double cost_function(std::vector<double> variables) {
    double x0 = variables[0];
    double alpha = 4.0 * abs(variables[0]);
    double beta = 2.0;
    double x2 = variables[0] * variables[0];
    return x2 + alpha * pow(sin(M_PI * x0), 3.0) + beta * pow(cos(M_PI * x0), 3.0);
}

// Export map of the cost function with fmt
void export_cost_function() {
    std::ofstream file("cost_function.csv");
    for (double x = -10; x < 10; x += 0.1) {
        file << fmt::format("{},{}\n", x, cost_function({x}));
    }
}
    

// Generate random neighbour
std::vector<double> neighbour_function(std::vector<double> variables) {
    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::vector<double> new_variables;
    for (auto variable : variables) {
        new_variables.push_back(variable + distribution(generator));
    }
    return new_variables;
}

int main() {
    export_cost_function();

    // Create simulated annealing object
    std::size_t max_iter = 2000;
    double initial_temp = 500;
    double final_temp = 0.001;
    std::size_t nb_parameters = 1;
    CoolingSchedule cooling_schedule = CoolingSchedule::Geometrical;
    SimulatedAnnealing sa(nb_parameters, cooling_schedule, max_iter, initial_temp, final_temp, cost_function, neighbour_function);

    // Set bounds
    sa.set_bounds({{-10, 10}});

    // Create random initial solution
    // sa.create_random_initial_solution();
    sa.set_initial_solution({11.0});
    // Run simulated annealing
    sa.run();

    // Print results with fmt
    fmt::print("Best solution: {}\n", sa.get_best_solution());
    fmt::print("Best cost: {}\n", sa.get_best_cost());

    return 0;
}