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

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "SimulatedAnneal.hpp"
#include "fmt/format.h"
#include "fmt/ostream.h"
#include "fmt/ranges.h"

// Optimize simple square function

double cost_function(std::vector<double> variables) {
    double sum     = 0;
    double sum_sq  = 0;
    double sum_cub = 0;
    for (auto variable : variables) {
        sum += variable;
        sum_sq += variable * variable;
        sum_cub += variable * variable * variable;
    }
    double y_sin = sin(sum);
    double y_cos = cos(sum_sq);
    return fabs(sum_sq);
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
    std::random_device                     random_device;
    std::mt19937                           generator(random_device());
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::vector<double>                    new_variables;
    for (auto variable : variables) {
        new_variables.push_back(variable + distribution(generator));
    }
    return new_variables;
}

int main() {
    // export_cost_function();

    // Create simulated annealing object
    std::size_t        max_iter         = 500;
    double             initial_temp     = 100;
    double             final_temp       = 1e-6;
    std::size_t        nb_parameters    = 3;
    CoolingSchedule    cooling_schedule = CoolingSchedule::Geometrical;
    SimulatedAnnealing sa(nb_parameters, cooling_schedule, max_iter, initial_temp, final_temp, cost_function);
    sa.set_alpha_cooling(0.99);

    // Set bounds
    sa.set_bounds({{-10, 10}, {-10, 10}, {-10, 10}});

    // Create random initial solution
    // sa.create_random_initial_solution();
    sa.set_initial_solution({9.0, -7.0, 3.0});
    // Run simulated annealing
    sa.run();

    // Print results with fmt
    fmt::print("Best solution: {}\n", sa.get_best_solution());
    fmt::print("Best cost: {}\n", sa.get_best_cost());

    return 0;
}