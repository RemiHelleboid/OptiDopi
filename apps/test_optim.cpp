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
#include "ParticleSwarm.hpp"

#include "fmt/format.h"
#include "fmt/ostream.h"
#include "fmt/ranges.h"

// Optimize simple square function

double rastrigin_function(std::vector<double> variables) {
    double sum = 0;
    for (auto variable : variables) {
        sum += variable * variable - 10 * cos(2 * M_PI * variable);
    }
    return 10 * variables.size() + sum;
}

int main() {
    // export_cost_function();

    // Create simulated annealing object
    std::size_t        max_iter         = 10000;
    double             initial_temp     = 100;
    double             final_temp       = 1e-6;
    std::size_t        nb_parameters    = 2;
    double             alpha_cooling    = 0.999;
    CoolingSchedule    cooling_schedule = CoolingSchedule::Geometrical;
    SimulatedAnnealing sa(nb_parameters, cooling_schedule, max_iter, initial_temp, final_temp, rastrigin_function);
    sa.set_alpha_cooling(alpha_cooling);
    sa.set_bounds({{-10, 10}, {-10, 10}});
    sa.set_initial_solution({8, -9});
    sa.run();

    // Print results with fmt
    fmt::print("Best solution: {}\n", sa.get_best_solution());
    fmt::print("Best cost: {}\n", sa.get_best_cost());


    // Particle swarm optimization
    double NbIterations = 1000;
    std::size_t     nb_particles = 20;
    double          c1           = 2;
    double          c2           = 2;
    double          w            = 0.9;
    Optimization::ParticleSwarm pso(NbIterations, nb_particles, nb_parameters, rastrigin_function);
    pso.set_bounds({{-10, 10}, {-10, 10}});
    pso.set_cognitive_weight(c1);
    pso.set_social_weight(c2);
    pso.set_inertia_weight(w);


    fmt::print("Best solution: {}\n", pso.get_best_position());
    fmt::print("Best cost: {}\n", pso.get_best_fitness());

    return 0;
}
