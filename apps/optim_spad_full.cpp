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


double intermediate_cost_function(std::vector<double> log_acceptor_levels) {
    // Create a complexe pin diode.
    double      x_length        = 10.0;
    std::size_t nb_points       = 350;
    double      donor_length    = 1.0;
    double      intrisic_length = 0.0;

    double donor_level    = 5.0e19;
    double intrisic_level = 1.0e13;

    std::vector<double> acceptor_x = {1.0, 2.0, 3.0, 5.0, 10.0};
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
    my_device.smooth_doping_profile(10);
    // int         thread_num = omp_get_thread_num();
    // auto files_list = std::filesystem::directory_iterator("doping_profile");
    // my_device.export_doping_profile(filename);

    double    target_anode_voltage = 40.0;
    double    tol                  = 1.0e-6;
    const int max_iter             = 100;
    double    voltage_step         = 0.01;
    my_device.solve_poisson(target_anode_voltage, tol, max_iter);

    bool poisson_success = my_device.get_poisson_success();
    if (!poisson_success) {
        fmt::print("Poisson failed\n");
        return BIG_DOUBLE;
    }

    const double stop_above_bv         = 5.0;
    double       mcintyre_voltage_step = 0.25;
    my_device.solve_mcintyre(mcintyre_voltage_step, stop_above_bv);
    const double brp_threshold = 1e-3;
    double       BV            = my_device.extract_breakdown_voltage(brp_threshold);
    double       BiasAboveBV   = 3.0;
    if (std::isnan(BV) || (BV + 1.5 * BiasAboveBV) > target_anode_voltage) {
        // fmt::print("BV is nan or BV + 1.5 * BiasAboveBV > target_anode_voltage\n");
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

    double min_doping = 1.0e15;
    double max_doping = 1.0e19;

    std::vector<double> min_values = {log10(min_doping), log10(min_doping), log10(min_doping), log10(min_doping), log10(min_doping)};
    std::vector<double> max_values = {log10(max_doping), log10(max_doping), log10(max_doping), log10(max_doping), log10(max_doping)};

    std::size_t max_iter      = 200;
    std::size_t nb_parameters = 5;
    std::size_t nb_particles  = 32;
    double      c1            = 3.0;
    double      c2            = 0.5;
    double      w             = 0.95;

    Optimization::ParticleSwarm pso(nb_particles, nb_parameters, cost_function);
    pso.set_bounds(min_values, max_values);
    pso.set_cognitive_weight(c1);
    pso.set_social_weight(c2);
    pso.set_inertia_weight(w);

    pso.optimize(max_iter);
}
