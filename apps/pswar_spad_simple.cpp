#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/xchar.h>

#include <filesystem>
#include <numeric>

#include "ImpactIonization.hpp"
#include "ParticleSwarm.hpp"
#include "PoissonSolver.hpp"
#include "SimulatedAnneal.hpp"
#include "device.hpp"
#include "doping_profile.hpp"
#include "fill_vector.hpp"
#include "omp.h"
#include "OptimStruct.hpp"

// Set number of threads

#define NAN_DOUBLE std::numeric_limits<double>::quiet_NaN()
static int idx = 0;

// Static file for log
static std::string filename = "ps_optim_path_full.csv";
std::ofstream      file_path(filename);

result_sim intermediate_cost_function(double length_intrinsic, double log_doping_acceptor, int thread_id = 0) {
    std::size_t number_points    = 400;
    double      total_length     = 10.0;
    double      length_donor     = 0.5;
    double      doping_donor     = 5.0e19;
    double      doping_intrinsic = 1.0e13;

    double doping_acceptor = pow(10, log_doping_acceptor);

    device my_device;
    my_device.setup_pin_diode(total_length, number_points, length_donor, length_intrinsic, doping_donor, doping_acceptor, doping_intrinsic);
    my_device.smooth_doping_profile(5);
    idx++;
    // std::string filename = fmt::format("results/thread_{}/doping_profile_{}.csv", thread_id, idx);
    // my_device.export_doping_profile(filename);

    double    target_anode_voltage = 40.0;
    double    tol                  = 1.0e-6;
    const int max_iter             = 100;
    double    voltage_step         = 0.01;
    my_device.solve_poisson(target_anode_voltage, tol, max_iter);
    // my_device.export_poisson_solution("poisson_solution", "poisson_solution_");
    bool poisson_success = my_device.get_poisson_success();
    if (!poisson_success) {
        fmt::print("Poisson failed\n");
        cost_function_result cost_resultr_NaN;
        cost_resultr_NaN.BV_cost    = -1.0e10;
        cost_resultr_NaN.BP_cost    = +1.0e10;
        cost_resultr_NaN.DW_cost    = +1.0e10;
        cost_resultr_NaN.total_cost = +1.0e10;

        return {length_intrinsic, doping_acceptor, NAN_DOUBLE, NAN_DOUBLE, NAN_DOUBLE, cost_resultr_NaN};
    }

    const double stop_above_bv         = 5.0;
    double       mcintyre_voltage_step = 0.25;
    my_device.solve_mcintyre(mcintyre_voltage_step, stop_above_bv);
    const double brp_threshold = 1e-3;
    double       BV            = my_device.extract_breakdown_voltage(brp_threshold);
    double       BiasAboveBV   = 3.0;
    if (std::isnan(BV) || (BV + 1.5 * BiasAboveBV) > target_anode_voltage) {
        // fmt::print("NaN BV\n");
        cost_function_result cost_resultr_NaN;
        cost_resultr_NaN.BV_cost    = -1.0e10;
        cost_resultr_NaN.BP_cost    = +1.0e10;
        cost_resultr_NaN.DW_cost    = +1.0e10;
        cost_resultr_NaN.total_cost = +1.0e10;

        return {length_intrinsic, doping_acceptor, BV, NAN_DOUBLE, NAN_DOUBLE, cost_resultr_NaN};
    }

    double meter_to_micron           = 1.0e6;
    double DepletionWidth_at_Biasing = my_device.get_depletion_at_voltage(BV + BiasAboveBV) * meter_to_micron;
    double BrP_at_Biasing            = my_device.get_brp_at_voltage(BV + BiasAboveBV);
    // fmt::print("BV: {:.5e}, BrP: {:.5e}, DW: {:.5e}\n", BV, BrP_at_Biasing, DepletionWidth_at_Biasing);
    // Put the result in the log file
    // fmt::print(file_path, "{:.5e},{:.5e},{:.5e}\n", BV, BrP_at_Biasing, DepletionWidth_at_Biasing);

    cost_function_result cost_resultr = my_device.compute_cost_function(BiasAboveBV);
    result_sim           full_result(length_intrinsic, doping_acceptor, BV, BrP_at_Biasing, DepletionWidth_at_Biasing, cost_resultr);

    return full_result;
}

double cost_function(std::vector<double> variables) {
    double length_intrinsic = variables[0];
    double doping_acceptor  = variables[1];
    double cost             = intermediate_cost_function(length_intrinsic, doping_acceptor).cost_result.total_cost;
    // fmt::print("Doping: {:.5e}, Length: {:.5e}, Cost: {:.5e}\n", pow(10, doping_acceptor), length_intrinsic, cost);
    return cost;
}

int main() {
    std::cout << "Particle Swarm Optimization" << std::endl;
    // Set number of threads

    std::string DIR_RES = "results_pso";
    if (!std::filesystem::exists(DIR_RES)) {
        std::filesystem::create_directory(DIR_RES);
    } else {
        std::filesystem::remove_all(DIR_RES);
        std::filesystem::create_directory(DIR_RES);
    }

    double min_doping = 1.0e16;
    double max_doping = 1.0e19;
    double min_length = 0.0;
    double max_length = 1.0;

    std::vector<double> min_values = {min_length, log10(min_doping)};
    std::vector<double> max_values = {max_length, log10(max_doping)};

    std::size_t max_iter      = 200;
    std::size_t nb_parameters = 2;
    double      c1            = 2.0;
    double      c2            = 0.5;
    double      w             = 0.95;
    std::size_t nb_particles  = 1;
#pragma omp parallel
    {
        nb_particles  = omp_get_num_threads();
    }
    std::cout << "Number particles: " << nb_particles << std::endl;

    Optimization::ParticleSwarm pso(nb_particles*4, nb_parameters, cost_function);
    pso.set_bounds(min_values, max_values);
    pso.set_cognitive_weight(c1);
    pso.set_social_weight(c2);
    pso.set_inertia_weight(w);

    pso.optimize(max_iter);
}