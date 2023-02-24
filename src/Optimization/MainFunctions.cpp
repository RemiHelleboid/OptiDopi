#include "MainFunctions.hpp"

#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/xchar.h>

#include <filesystem>
#include <numeric>

#include "Functions.hpp"
#include "ImpactIonization.hpp"
#include "OptimStruct.hpp"
#include "ParticleSwarm.hpp"
#include "PoissonSolver.hpp"
#include "SimulatedAnneal.hpp"
#include "device.hpp"
#include "doping_profile.hpp"
#include "fill_vector.hpp"
#include "omp.h"

namespace Optimization {

#define NAN_DOUBLE std::numeric_limits<double>::quiet_NaN()
#define BIG_DOUBLE 1.0e10

static int IDX_ITER = 0;

#define N_X 8
#define DopSmooth 11
#define NBPOINTS 500

#define DonorMIN 20
#define DonorMAX 21

std::vector<double> x_acceptors(double length_donor, double total_length, std::size_t nb_points_acceptor) {
    // The x positions are first on a fine grid then on a coarse grid
    double dx_fine        = 0.25;
    double size_fine_area = 1.5;

    std::vector<double> x_acceptor(nb_points_acceptor);
    x_acceptor[0] = length_donor;

    // First we fill the fine area
    std::size_t i = 1;
    while (x_acceptor[i - 1] < length_donor + size_fine_area) {
        x_acceptor[i] = x_acceptor[i - 1] + dx_fine;
        ++i;
    }
    double dx_coarse = (total_length - length_donor - size_fine_area) / (nb_points_acceptor - i);
    while (i < nb_points_acceptor) {
        x_acceptor[i] = x_acceptor[i - 1] + dx_coarse;
        ++i;
    }
    return x_acceptor;
}

void set_up_bounds(double               length_min,
                   double               length_max,
                   double               log_donor_min,
                   double               log_donor_max,
                   double               log_accpetor_min,
                   double               log_acceptor_max,
                   std::vector<double>& min_bounds,
                   std::vector<double>& max_bounds) {
    min_bounds.resize(N_X + 2);
    max_bounds.resize(N_X + 2);
    // Donnor levels
    min_bounds[0] = length_min;
    max_bounds[0] = length_max;
    min_bounds[1] = log_donor_min;
    max_bounds[1] = log_donor_max;
    std::fill(min_bounds.begin() + 2, min_bounds.end(), log_accpetor_min);
    std::fill(max_bounds.begin() + 2, max_bounds.end(), log_acceptor_max);
}

void export_best_path(std::vector<std::vector<double>> best_path, std::string dirname) {
    std::filesystem::create_directories(dirname);
    std::cout << "Exporting best path to " << dirname << std::endl;
    double      intrinsic_level  = 1.0e13;
    double      x_length         = 10.0;
    std::size_t nb_points        = NBPOINTS;
    double      intrinsic_length = 0.0;

    // File to export the best path figures (BV, BrP, DW, ...)
    std::ofstream best_path_file(dirname + "/SPAD_figures_best_path.csv");
    best_path_file << "Iteration,BV,BrP,DW,Cost" << std::endl;

    const std::string poisson_dir = fmt::format("{}/poisson_res/", dirname);

#pragma omp parallel for
    for (std::size_t i = 0; i < best_path.size(); ++i) {
        std::cout << "\r" << i << "/" << best_path.size() << std::flush;
        double              donor_length = best_path[i][0];
        double              donor_level  = pow(10, best_path[i][1]);
        std::vector<double> acceptor_x   = x_acceptors(donor_length, x_length, N_X);
        std::vector<double> acceptor_levels(best_path[i].size() - 2);
        std::vector<double> acceptor_levels_log(best_path[i].begin() + 2, best_path[i].end());
        std::transform(best_path[i].begin() + 2, best_path[i].end(), acceptor_levels.begin(), [](double x) { return pow(10, x); });
        device my_device;
        my_device.set_up_complex_diode(x_length,
                                       nb_points,
                                       donor_length,
                                       intrinsic_length,
                                       donor_level,
                                       intrinsic_level,
                                       acceptor_x,
                                       acceptor_levels);
        my_device.smooth_doping_profile(DopSmooth);

        // Solve the Poisson and McIntyre equations
        double       target_anode_voltage  = 40.0;
        double       tol                   = 1.0e-8;
        const int    max_iter              = 100;
        double       voltage_step          = 0.01;
        double       mcintyre_voltage_step = 0.25;
        const double stop_above_bv         = 5.0;
        double       BiasAboveBV           = 3.0;

        my_device.solve_poisson_and_mcintyre(target_anode_voltage, tol, max_iter, mcintyre_voltage_step, stop_above_bv);
        bool poisson_success = my_device.get_poisson_success();
        if (!poisson_success) {
            fmt::print("Poisson failed\n");
        }
        double time = i / static_cast<double>(best_path.size());
        cost_function_result cost_result = my_device.compute_cost_function(BiasAboveBV, time);
        double               BV          = cost_result.result.BV;
        double               BRP         = cost_result.result.BrP;
        double               DW          = cost_result.result.DW;
        double               cost        = cost_result.total_cost;
        fmt::print(best_path_file, "{},{:.2f},{:.2f},{:.2e},{:.2f}\n", i, BV, BRP, DW, cost);
#pragma omp critical
        my_device.export_doping_profile(fmt::format("{}/doping_profile_{:03d}.csv", dirname, i));
        my_device.export_poisson_solution_at_voltage(BV + BiasAboveBV, poisson_dir, fmt::format("poisson_{}_", i));
    }
}

double intermediate_cost_function(double donor_length, double log_donor_level, std::vector<double> log_acceptor_levels, std::vector<double> parameters) {
    double              x_length         = 10.0;
    std::size_t         nb_points        = NBPOINTS;
    double              intrinsic_length = 0.0;
    double              donor_level      = pow(10, log_donor_level);
    double              intrinsic_level  = 1.0e13;
    std::vector<double> acceptor_x       = x_acceptors(donor_length, x_length, N_X);
    std::vector<double> acceptor_levels(log_acceptor_levels.size());
    std::transform(log_acceptor_levels.begin(), log_acceptor_levels.end(), acceptor_levels.begin(), [](double x) { return pow(10, x); });
    if (acceptor_levels.size() != acceptor_x.size()) {
        fmt::print("Error: the size of the acceptor_levels vector is not the same as the acceptor_x vector.\n");
        exit(1);
    }
    device my_device;
    my_device.set_up_complex_diode(x_length,
                                   nb_points,
                                   donor_length,
                                   intrinsic_length,
                                   donor_level,
                                   intrinsic_level,
                                   acceptor_x,
                                   acceptor_levels);
    my_device.smooth_doping_profile(DopSmooth);

    double       target_anode_voltage  = 40.0;
    double       tol                   = 1.0e-8;
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
    std::size_t iter_nb = parameters[0];
    std::size_t max_iter_nb = parameters[1];
    double time = parameters[0] / static_cast<double>(parameters[1]);
    cost_function_result cost_result = my_device.compute_cost_function(BiasAboveBV, time);
    double               BV          = cost_result.result.BV;
    double               BRP         = cost_result.result.BrP;
    double               DW          = cost_result.result.DW;
    double               cost        = cost_result.total_cost;

    // fmt::print("BV: {:.2f}, BRP: {:.2f}, DW: {:.2e}, Cost: {:.2f}\n", BV, BRP, DW, cost);
    return cost;
}

/**
 * @brief Cost function that will be called by the optimizer.
 * The first two variables are the donor length and the donor level.
 * The other variables are the acceptor levels.
 *
 * @param variables
 * @return double
 */

double costyy_function(std::vector<double> variables, const std::vector<double>& parameters) {
    double              donor_length    = variables[0];
    double              log_donor_level = variables[1];
    std::vector<double> log_acceptor_levels(variables.begin() + 2, variables.end());
    double              cost = intermediate_cost_function(donor_length, log_donor_level, log_acceptor_levels, parameters);
    // fmt::print("Doping: {:.5e}, Length: {:.5e}, Cost: {:.5e}\n", pow(10, doping_acceptor), length_intrinsic, cost);
    return cost;
}

std::function<double(std::vector<double>, const std::vector<double>&)> cost_function_wrapper = costyy_function;

/**
 * @brief Main function for the Particle Swarm Optimization.
 *
 */
void MainParticleSwarmSPAD() {
    const std::string timestamp = fmt::format("{:%Y-%m-%d_%H-%M-%S}", fmt::localtime(std::time(nullptr)));
    const std::string DIR_RES   = fmt::format("results_pso/{}/", timestamp);
    if (!std::filesystem::exists(DIR_RES)) {
        std::filesystem::create_directories(DIR_RES);
    } else {
        std::filesystem::remove_all(DIR_RES);
        std::filesystem::create_directories(DIR_RES);
    }

    std::size_t nb_parameters = N_X + 2;
    // Boundaries setup
    // Boundaries setup
    double              min_length_donor = 0.5;
    double              max_length_donor = 0.50001;
    double              min_doping       = 14.0;
    double              max_doping       = 19.0;
    double              donor_min_doping = DonorMIN;
    double              donor_max_doping = DonorMAX;
    std::vector<double> min_values(nb_parameters);
    std::vector<double> max_values(nb_parameters);
    set_up_bounds(min_length_donor, max_length_donor, donor_min_doping, donor_max_doping, min_doping, max_doping, min_values, max_values);

    std::size_t nb_threads = 1;
#pragma omp parallel
    { nb_threads = omp_get_num_threads(); }
    std::cout << "Number threads: " << nb_threads << std::endl;

    std::size_t max_iter         = 150;
    double      c1               = 3.0;
    double      c2               = 1.0;
    double      w                = 0.9;
    double      velocity_scaling = 0.1;
    std::size_t nb_particles     = 1 * nb_threads;
    std::cout << "Number particles: " << nb_particles << std::endl;
    Optimization::ParticleSwarm pso(max_iter, nb_particles, nb_parameters, cost_function_wrapper);
    pso.set_dir_export(DIR_RES);
    pso.set_bounds(min_values, max_values);
    pso.set_cognitive_weight(c1);
    pso.set_social_weight(c2);
    pso.set_inertia_weight(w);
    pso.set_velocity_scaling(velocity_scaling);
    pso.set_cognitive_learning_scheme(Optimization::LearningScheme::Constant);
    std::vector<double> initial_solution = random_initial_position(min_values, max_values);
    pso.optimize();

    auto best_path = pso.get_history_best_position();
    export_best_path(best_path, fmt::format("{}/BEST/", DIR_RES));
}

/**
 * @brief MAIN FUNCTION FOR SIMULATED ANNEALING.
 *
 */
void MainSimulatedAnnealingSPAD() {
    const std::string timestamp = fmt::format("{:%Y-%m-%d_%H-%M-%S}", fmt::localtime(std::time(nullptr)));
    const std::string DIR_RES   = fmt::format("results_sa/{}/", timestamp);
    if (!std::filesystem::exists(DIR_RES)) {
        std::filesystem::create_directories(DIR_RES);
    } else {
        std::filesystem::remove_all(DIR_RES);
        std::filesystem::create_directories(DIR_RES);
    }

    // Create simulated annealing object
    std::size_t     max_iter         = 1000;
    double          initial_temp     = 10;
    double          final_temp       = 0.005;
    std::size_t     nb_parameters    = N_X + 2;
    CoolingSchedule cooling_schedule = CoolingSchedule::Geometrical;
    double          cooling_factor   = 0.95;
    // Boundaries setup
    double              min_length_donor = 0.5;
    double              max_length_donor = 0.50001;
    double              min_doping       = 14.0;
    double              max_doping       = 19.0;
    double              donor_min_doping = DonorMIN;
    double              donor_max_doping = DonorMAX;
    std::vector<double> min_values(nb_parameters);
    std::vector<double> max_values(nb_parameters);
    set_up_bounds(min_length_donor, max_length_donor, donor_min_doping, donor_max_doping, min_doping, max_doping, min_values, max_values);

    std::size_t nb_threads = 1;
#pragma omp parallel
    { nb_threads = omp_get_num_threads(); }
    std::cout << "Number threads: " << nb_threads << std::endl;
    std::size_t nb_doe = 8;
    std::cout << "Number DOE: " << nb_doe << std::endl;
    // Run simulated annealing with different initial solutions, one for each thread
    std::vector<std::vector<double>> initial_solutions(nb_doe);
    std::vector<std::vector<double>> final_solutions(nb_doe);
    std::vector<double>              final_costs(nb_doe);

    std::vector<SimulatedAnnealHistory> histories(nb_doe);

#pragma omp parallel for schedule(dynamic) num_threads(nb_threads)
    for (int i = 0; i < nb_doe; i++) {
        std::string directory = fmt::format("{}/thread_{}/", DIR_RES, i);
        std::filesystem::create_directory(directory);
        SimulatedAnnealing  sa(nb_parameters, cooling_schedule, max_iter, initial_temp, final_temp, cost_function_wrapper);
        std::vector<double> initial_solution = random_initial_position(min_values, max_values);
        sa.set_initial_solution(initial_solution);
        sa.set_prefix_name(directory);
        sa.set_bounds(min_values, max_values);
        sa.set_alpha_cooling(cooling_factor);
        sa.create_random_initial_solution();
        sa.set_frequency_print(50);
        sa.run();

#pragma omp critical
        {
            // Save best solution
            sa.export_history();
            std::vector<double> best_solution = sa.get_best_solution();
            double              best_cost     = sa.get_best_cost();
            final_solutions[i]                = best_solution;
            final_costs[i]                    = best_cost;
            // Save all solutions
            histories[i] = sa.get_history();
        }
    }

    // Find best solution
    std::vector<double>::iterator it    = std::min_element(final_costs.begin(), final_costs.end());
    int                           index = std::distance(final_costs.begin(), it);
    std::cout << "*****************************************************" << std::endl;
    std::cout << "Best cost: " << final_costs[index] << std::endl;
    std::cout << "Best solution: " << fmt::format("{}", final_solutions[index]) << std::endl;

    // Copy the folder with the best solution into best_path_index
    std::string best_path_file = fmt::format("{}/thread_{}/history_optimization.csv", DIR_RES, index);
    std::string new_path       = fmt::format("{}/BEST/", DIR_RES, index);
    std::filesystem::create_directory(new_path);
    std::filesystem::copy(best_path_file, DIR_RES, std::filesystem::copy_options::recursive);

    // Get history of best path and save it
    std::cout << "Saving best path" << std::endl;
    std::vector<std::vector<double>> best_path = histories[index].solutions;
    export_best_path(best_path, new_path);
}

}  // namespace Optimization