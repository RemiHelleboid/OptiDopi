#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/xchar.h>

#include <chrono>
#include <filesystem>
#include <numeric>


#include "MainFunctions.hpp"

#include "omp.h"


int main(int argc, const char** argv) {
    // Simulated annealing
    // Parse command line arguments (NbDOE, NbIterMax)
    std::size_t nb_doe = 8;
    std::size_t nb_iter_max  = 100;
    if (argc == 3) {
        nb_doe = std::stoul(argv[1]);
        nb_iter_max  = std::stoul(argv[2]);
    } else {
        fmt::print("Usage: {} NbDOE NbIterMax \n", argv[0]);
        fmt::print("Using default values: NbDOE = {}, NbIterMax = {} \n", nb_doe, nb_iter_max);
    }
    auto start = std::chrono::high_resolution_clock::now();
    Optimization::MainSimulatedAnnealingSPAD(nb_doe, nb_iter_max);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    fmt::print("Total time : {} s \n\n", elapsed_seconds.count());

    double poisson_time              = NewtonPoissonSolver::get_poisson_solver_time();
    double mcintyre_time             = mcintyre::McIntyre::get_mcintyre_time();
    double ration_converged_mcintyre = mcintyre::McIntyre::get_ratio_converged_sim() * 100.0;
    fmt::print("Total time spent in Poisson solver: {} s \n", poisson_time);
    fmt::print("Total time spent in McIntyre solver: {} s \n", mcintyre_time);
    fmt::print("Ratio of converged simulations: {:.2f}% \n", ration_converged_mcintyre);
    return 0;
}
