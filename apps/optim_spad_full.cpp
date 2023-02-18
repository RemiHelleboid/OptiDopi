#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/xchar.h>

#include <filesystem>
#include <numeric>

#include "ImpactIonization.hpp"
#include "OptimStruct.hpp"
#include "ParticleSwarm.hpp"
#include "PoissonSolver.hpp"
#include "SimulatedAnneal.hpp"
#include "device.hpp"
#include "doping_profile.hpp"
#include "fill_vector.hpp"
#include "omp.h"

// Set number of threads

#define NAN_DOUBLE std::numeric_limits<double>::quiet_NaN()
#define BIG_DOUBLE 1.0e10

static int IDX_ITER = 0;

#define N_X 11

void export_best_path(std::vector<std::vector<double>> best_path, std::string dirname) {
    std::filesystem::create_directories(dirname);

    double              x_length        = 10.0;
    std::size_t         nb_points       = 500;
    double              donor_length    = 1.0;
    double              intrisic_length = 0.0;
    double              donor_level     = 5.0e19;
    double              intrisic_level  = 1.0e13;
    std::vector<double> acceptor_x      = {1.0, 1.10, 1.20, 1.30, 1.40, 1.50, 1.75, 2.0, 3.0, 6.0, 10.0};

    for (std::size_t i = 0; i < best_path.size(); ++i) {
        std::vector<double> acceptor_levels(best_path[i].size());
        std::transform(best_path[i].begin(), best_path[i].end(), acceptor_levels.begin(), [](double x) { return pow(10, x); });
        device my_device;
        my_device.set_up_complex_diode(x_length,
                                       nb_points,
                                       donor_length,
                                       intrisic_length,
                                       donor_level,
                                       intrisic_level,
                                       acceptor_x,
                                       acceptor_levels);
        my_device.smooth_doping_profile(5);
        my_device.export_doping_profile(fmt::format("{}/doping_profile_{:03d}.csv", dirname, i));
    }
}

double intermediate_cost_function(std::vector<double> log_acceptor_levels) {
    // Create a complexe pin diode.
    double      x_length        = 10.0;
    std::size_t nb_points       = 500;
    double      donor_length    = 1.0;
    double      intrisic_length = 0.0;

    double donor_level    = 5.0e19;
    double intrisic_level = 1.0e13;

    // std::vector<double> acceptor_x = utils::linspace(donor_length + intrisic_length, x_length, N_X);
    std::vector<double> acceptor_x = {1.0, 1.10, 1.20, 1.30, 1.40, 1.50, 1.75, 2.0, 3.0, 6.0, 10.0};

    std::vector<double> acceptor_levels(log_acceptor_levels.size());
    // Take the power 10 of the acceptor levels
    std::transform(log_acceptor_levels.begin(), log_acceptor_levels.end(), acceptor_levels.begin(), [](double x) { return pow(10, x); });
    // Check if the size of the vector is the same
    if (acceptor_levels.size() != acceptor_x.size()) {
        fmt::print("Error: the size of the acceptor_levels vector is not the same as the acceptor_x vector.\n");
        exit(1);
    }
    device my_device;
    my_device
        .set_up_complex_diode(x_length, nb_points, donor_length, intrisic_length, donor_level, intrisic_level, acceptor_x, acceptor_levels);
    my_device.smooth_doping_profile(5);

    double       target_anode_voltage  = 30.0;
    double       tol                   = 1.0e-6;
    const int    max_iter              = 100;
    double       voltage_step          = 0.01;
    double       mcintyre_voltage_step = 0.25;
    const double stop_above_bv         = 5.0;
    double       BiasAboveBV           = 3.0;

    my_device.solve_poisson_and_mcintyre(target_anode_voltage, tol, max_iter, mcintyre_voltage_step, stop_above_bv);
    bool poisson_success = my_device.get_poisson_success();
    if (!poisson_success) {
        // fmt::print("Poisson failed\n");
        return BIG_DOUBLE;
    }

    cost_function_result cost_resultr = my_device.compute_cost_function(BiasAboveBV);
    double               cost         = cost_resultr.total_cost;
    return cost;
}

double cost_function(std::vector<double> variables) {
    // Call the intermediate cost function
    double cost = intermediate_cost_function(variables);
    // fmt::print("Doping: {:.5e}, Length: {:.5e}, Cost: {:.5e}\n", pow(10, doping_acceptor), length_intrinsic, cost);
    return cost;
}

int main(int argc, const char** argv) {
    // PSO parameters
    std::string DIR_RES = "results_pso";
    if (!std::filesystem::exists(DIR_RES)) {
        std::filesystem::create_directory(DIR_RES);
    } else {
        std::filesystem::remove_all(DIR_RES);
        std::filesystem::create_directory(DIR_RES);
    }

    double min_doping = 1.0e13;
    double max_doping = 1.0e19;

    std::vector<double> min_values(N_X, log10(min_doping));
    std::vector<double> max_values(N_X, log10(max_doping));

    std::vector<double> x_acceptors = utils::linspace(1.0, 10.0, N_X);
    // fmt print
    fmt::print("x_acceptors: {}\n", x_acceptors);

    std::size_t nb_particles = 1;
#pragma omp parallel
    { nb_particles = omp_get_num_threads(); }
    std::cout << "Number particles: " << nb_particles << std::endl;
    std::size_t max_iter         = 250;
    std::size_t nb_parameters    = N_X;
    double      c1               = 3.0;
    double      c2               = 1.0;
    double      w                = 0.9;
    double      velocity_scaling = 0.5;

    Optimization::ParticleSwarm pso(max_iter, nb_particles, nb_parameters, cost_function);
    pso.set_bounds(min_values, max_values);
    pso.set_cognitive_weight(c1);
    pso.set_social_weight(c2);
    pso.set_inertia_weight(w);
    pso.set_velocity_scaling(velocity_scaling);
    pso.set_cognitive_learning_scheme(Optimization::LearningScheme::Linear);
    pso.optimize();

    auto best_path = pso.get_history_best_position();
    export_best_path(best_path, fmt::format("{}/BEST/", DIR_RES));
}
