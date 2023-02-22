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

#define N_X 7

std::vector<double> x_acceptors(double length_donor, double total_length, std::size_t nb_points_acceptor) {
    std::vector<double> x_acc = utils::geomspace(length_donor, total_length, nb_points_acceptor);
    return x_acc;
}

void export_best_path(std::vector<std::vector<double>> best_path, std::string dirname) {
    std::filesystem::create_directories(dirname);
    std::cout << "Exporting best path to " << dirname << std::endl;

    double      x_length  = 10.0;
    std::size_t nb_points = 500;
    // double              donor_length    = 1.0;
    double intrisic_length = 0.0;
    // double              donor_level     = 5.0e19;
    double intrisic_level = 1.0e13;

    for (std::size_t i = 0; i < best_path.size(); ++i) {
        double              length_don = best_path[i][0];
        double              level_don  = pow(10, best_path[i][1]);
        std::vector<double> acceptor_x = x_acceptors(length_don, x_length, N_X);
        std::vector<double> acceptor_levels(best_path[i].size() - 2);
        std::transform(best_path[i].begin() + 2, best_path[i].end(), acceptor_levels.begin(), [](double x) { return pow(10, x); });
        device my_device;
        my_device
            .set_up_complex_diode(x_length, nb_points, length_don, intrisic_length, level_don, intrisic_level, acceptor_x, acceptor_levels);
        my_device.smooth_doping_profile(5);
        my_device.export_doping_profile(fmt::format("{}/doping_profile_{:03d}.csv", dirname, i));
    }
}

double intermediate_cost_function(double donor_length, double log_donor_level, std::vector<double> log_acceptor_levels) {
    // Create a complexe pin diode.
    double      x_length        = 10.0;
    std::size_t nb_points       = 500;
    double      ddonor_length   = 1.0;
    double      intrisic_length = 0.0;

    double donor_level    = pow(10, log_donor_level);
    double intrisic_level = 1.0e13;

    // std::vector<double> acceptor_x = utils::linspace(donor_length + intrisic_length, x_length, N_X);
    double              dx_fine    = 0.20;
    double              dx_neutral = 0.50;
    double              dx_coarse  = 2.0;
    std::vector<double> acceptor_x = x_acceptors(donor_length, x_length, N_X);
    // Print the acceptor x vector
    // fmt::print("acceptor_x = {}\n", acceptor_x);
    // std::cout << "Size of acceptor_x = " << acceptor_x.size() << std::endl;
    std::vector<double> acceptor_levels(log_acceptor_levels.size());
    // Take the power 10 of the acceptor levels
    std::transform(log_acceptor_levels.begin(), log_acceptor_levels.end(), acceptor_levels.begin(), [](double x) { return pow(10, x); });
    // Check if the size of the vector is the same
    if (acceptor_levels.size() != acceptor_x.size()) {
        fmt::print("Error: the size of the acceptor_levels vector is not the same as the acceptor_x vector.\n");
        exit(1);
    }
    device my_device;
    my_device.set_up_complex_diode(x_length,
                                   nb_points,
                                   ddonor_length,
                                   intrisic_length,
                                   donor_level,
                                   intrisic_level,
                                   acceptor_x,
                                   acceptor_levels);
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

/**
 * @brief Cost function that will be called by the optimizer.
 * The first two variables are the donor length and the donor level.
 * The other variables are the acceptor levels.
 *
 * @param variables
 * @return double
 */
double cost_function(std::vector<double> variables) {
    double              donor_length    = variables[0];
    double              log_donor_level = variables[1];
    std::vector<double> log_acceptor_levels(variables.begin() + 2, variables.end());
    double              cost = intermediate_cost_function(donor_length, log_donor_level, log_acceptor_levels);
    // fmt::print("Doping: {:.5e}, Length: {:.5e}, Cost: {:.5e}\n", pow(10, doping_acceptor), length_intrinsic, cost);
    return cost;
}

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
    double              min_length_donor = 0.5;
    double              max_length_donor = 5.0;
    double              min_doping       = 1.0e13;
    double              max_doping       = 1.0e19;
    std::vector<double> min_values(nb_parameters);
    std::vector<double> max_values(nb_parameters);
    min_values[0] = min_length_donor;
    max_values[0] = max_length_donor;
    std::fill(min_values.begin() + 1, min_values.end(), log10(min_doping));
    std::fill(max_values.begin() + 1, max_values.end(), log10(max_doping));
    // Donnor levels
    min_values[1] = 18;
    max_values[1] = 20;

    std::size_t nb_threads = 1;
#pragma omp parallel
    { nb_threads = omp_get_num_threads(); }
    std::cout << "Number threads: " << nb_threads << std::endl;

    std::size_t                 max_iter         = 150;
    double                      c1               = 3.0;
    double                      c2               = 1.0;
    double                      w                = 0.9;
    double                      velocity_scaling = 0.5;
    std::size_t                 nb_particles     = nb_threads * 2;
    std::cout << "Number particles: " << nb_particles << std::endl;
    Optimization::ParticleSwarm pso(max_iter, nb_particles, nb_parameters, cost_function);
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
    std::size_t     max_iter         = 500;
    double          initial_temp     = 500;
    double          final_temp       = 0.001;
    std::size_t     nb_parameters    = N_X + 2;
    CoolingSchedule cooling_schedule = CoolingSchedule::Geometrical;

    // Boundaries setup
    double              min_length_donor = 0.5;
    double              max_length_donor = 5.0;
    double              min_doping       = 1.0e13;
    double              max_doping       = 1.0e19;
    std::vector<double> min_values(nb_parameters);
    std::vector<double> max_values(nb_parameters);
    min_values[0] = min_length_donor;
    max_values[0] = max_length_donor;
    // Donnor levels
    std::fill(min_values.begin() + 1, min_values.end(), log10(min_doping));
    std::fill(max_values.begin() + 1, max_values.end(), log10(max_doping));
    min_values[1] = 16;
    max_values[1] = 20;

    std::size_t nb_threads = 1;
#pragma omp parallel
    { nb_threads = omp_get_num_threads(); }
    std::cout << "Number threads: " << nb_threads << std::endl;
    // Run simulated annealing with different initial solutions, one for each thread
    std::vector<std::vector<double>> initial_solutions(nb_threads);
    std::vector<std::vector<double>> final_solutions(nb_threads);
    std::vector<double>              final_costs(nb_threads);

    std::vector<SimulatedAnnealHistory> histories(nb_threads);

    std::size_t nb_doe = nb_threads;
#pragma omp parallel for schedule(dynamic) num_threads(nb_threads)
    for (int i = 0; i < nb_doe; i++) {
        std::string directory = fmt::format("{}/thread_{}/", DIR_RES, i);
        std::filesystem::create_directory(directory);
        SimulatedAnnealing  sa(nb_parameters, cooling_schedule, max_iter, initial_temp, final_temp, cost_function);
        std::vector<double> initial_solution = random_initial_position(min_values, max_values);
        sa.set_initial_solution(initial_solution);
        sa.set_prefix_name(directory);
        sa.set_bounds(min_values, max_values);
        sa.set_alpha_cooling(0.98);
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
    std::string best_path_folder = fmt::format("{}/thread_{}/history_optimization.csv", DIR_RES, index);
    std::string new_path         = fmt::format("{}/best_path_{}/", DIR_RES, index);
    std::filesystem::create_directory(new_path);
    std::filesystem::copy(best_path_folder, DIR_RES, std::filesystem::copy_options::recursive);

    // Get history of best path and save it
    std::cout << "Saving best path" << std::endl;
    std::vector<std::vector<double>> best_path = histories[index].solutions;
    // export_best_path(best_path, DIR_RES + "best_path/");
}

}  // namespace Optimization