#include "MainFunctions.hpp"

#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/xchar.h>

#include <filesystem>
#include <memory>
#include <numeric>

#include "Device1D.hpp"
#include "DopingProfile1D.hpp"
#include "Functions.hpp"
#include "ImpactIonization.hpp"
#include "OptimStruct.hpp"
#include "ParticleSwarm.hpp"
#include "PoissonSolver1D.hpp"
#include "SimulatedAnneal.hpp"
#include "fill_vector.hpp"
#include "omp.h"

namespace Optimization {

#define NAN_DOUBLE std::numeric_limits<double>::quiet_NaN()
#define BIG_DOUBLE 1.0e10

#define N_X 8
#define DopSmooth 11
#define NBPOINTS 500
#define ITER_MAX 10

#define DonorMIN 16
#define DonorMAX 21

std::vector<double> x_acceptors(double length_donor, double total_length, std::size_t nb_points_acceptor) {
    // The x positions are first on a fine grid then on a coarse grid
    double dx_fine        = 0.3;
    double size_fine_area = 2.0;

    std::vector<double> x_acceptor(nb_points_acceptor);
    x_acceptor[0] = length_donor;

    // First we fill the fine area
    std::size_t i = 1;
    while (x_acceptor[i - 1] < length_donor + size_fine_area) {
        x_acceptor[i] = x_acceptor[i - 1] + dx_fine;
        ++i;
    }
    double dx_coarse = (total_length - length_donor - size_fine_area) / (nb_points_acceptor - i);
    while (i < nb_points_acceptor && x_acceptor[i - 1] + dx_coarse <= total_length) {
        x_acceptor[i] = x_acceptor[i - 1] + dx_coarse;
        ++i;
    }
    x_acceptor[x_acceptor.size() - 1] = total_length;
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
    double      intrinsic_level     = 1.0e13;
    double      x_length            = 8.0;
    std::size_t nb_points           = NBPOINTS;
    double      intrinsic_length    = 0.0;
    double      voltage_step_export = 0.5;
    // ADMC parameters
    double               temperature = 300.0;
    double               time_step   = 5.0e-14;
    double               final_time  = 10.0e-9;
    ADMC::ParametersADMC parameters_admc;
    parameters_admc.m_time_step                  = time_step;
    parameters_admc.m_max_time                   = final_time;
    parameters_admc.m_temperature                = temperature;
    parameters_admc.m_activate_impact_ionization = true;
    parameters_admc.m_activate_particle_creation = true;
    parameters_admc.m_max_particles              = 200;
    parameters_admc.m_avalanche_threshold        = parameters_admc.m_max_particles;
    std::size_t nb_simulation_per_point          = 100;
    std::size_t NbPointsX                        = 100;

    // File to export the best path figures (BV, BrP, DW, ...)
    std::ofstream best_path_file(dirname + "/SPAD_figures_best_path.csv");
    best_path_file << "Iteration,BV,BrP,DW,Cost" << std::endl;

    const std::string poisson_dir = fmt::format("{}/poisson_res/", dirname);

    // std::vector<std::size_t> iter_to_save_for_jitter = {0, 25, 50, 75, 100, 200, 300, 400, 500};
    std::vector<std::size_t> iter_to_save_for_jitter;
    int StepJitter = 10;
    for (int idx_Jitter=0; idx_Jitter < best_path.size(); idx_Jitter+=StepJitter) {
        iter_to_save_for_jitter.push_back(idx_Jitter);
    }

    std::vector<std::unique_ptr<Device1D>> saved_devices(iter_to_save_for_jitter.size());
    std::vector<double>                    saved_BV(iter_to_save_for_jitter.size());

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < best_path.size(); ++i) {
        std::cout << "\r" << i << "/" << best_path.size() << std::flush;
        double              donor_length = best_path[i][0];
        double              donor_level  = pow(10, best_path[i][1]);
        std::vector<double> acceptor_x   = x_acceptors(donor_length, x_length, N_X);
        std::vector<double> acceptor_levels(best_path[i].size() - 2);
        std::vector<double> acceptor_levels_log(best_path[i].begin() + 2, best_path[i].end());
        std::transform(best_path[i].begin() + 2, best_path[i].end(), acceptor_levels.begin(), [](double x) { return pow(10, x); });
        std::unique_ptr<Device1D> my_device = std::make_unique<Device1D>();
        my_device->set_up_complex_diode(x_length,
                                        nb_points,
                                        donor_length,
                                        intrinsic_length,
                                        donor_level,
                                        intrinsic_level,
                                        acceptor_x,
                                        acceptor_levels);
        my_device->smooth_doping_profile(DopSmooth);

        // Solve the Poisson and McIntyre equations
        double       target_anode_voltage  = 30.0;
        double       tol                   = 1.0e-8;
        const int    max_iter              = 1000;
        double       mcintyre_voltage_step = 0.25;
        const double stop_above_bv         = 5.0;
        double       BiasAboveBV           = 3.0;

        my_device->solve_poisson_and_mcintyre(target_anode_voltage, tol, max_iter, mcintyre_voltage_step, stop_above_bv);
        bool poisson_success = my_device->get_poisson_success();
        if (!poisson_success) {
            fmt::print("Poisson failed\n");
        }
        double               time        = i / static_cast<double>(best_path.size());
        cost_function_result cost_result = my_device->compute_cost_function(BiasAboveBV, time);
        double               BV          = cost_result.result.BV;
        double               BRP         = cost_result.result.BrP;
        double               DW          = cost_result.result.DW;
        double               cost        = cost_result.total_cost;
        fmt::print(best_path_file, "{},{:.2f},{:.2f},{:.2e},{:.2f}\n", i, BV, BRP, DW, cost);


#pragma omp critical
        {
            if (i%10 == 0 || i == best_path.size()) {
                my_device->export_doping_profile(fmt::format("{}/doping_profile_{:03d}.csv", dirname, i));
                const std::string poisson_dir_iter = fmt::format("{}/poisson_res_{:03d}", dirname, i);
                std::filesystem::create_directories(poisson_dir_iter);
                my_device->export_poisson_solution(poisson_dir_iter, fmt::format("poisson_{}_", i), voltage_step_export);
            }

            // Save the device for the jitter computation.
            for (std::size_t j = 0; j < iter_to_save_for_jitter.size(); ++j) {
                if (i == iter_to_save_for_jitter[j]) {
                    saved_devices[j] = std::move(my_device);
                    saved_BV[j]      = BV;
                }
            }
        }
    }
    best_path_file.close();
    // Compute Jitter for the saved devices.
    // std::cout << std::endl;
    // std::cout << "Computing jitter" << std::endl;
    // for (std::size_t i = 0; i < saved_devices.size(); ++i) {
    //     std::cout << "Computing jitter for iteration " << i << " / " << saved_devices.size() << std::endl;
    //     double            BV           = saved_BV[i];
    //     double            BiasAboveBV  = 3.0;
    //     double            voltage_AMDC = BV + BiasAboveBV;
    //     const std::string prefix_ADMC  = fmt::format("{}/ADMC_Iter_{:03d}_", dirname, iter_to_save_for_jitter[i]);
    //     saved_devices[i]->DeviceADMCSimulation(parameters_admc, voltage_AMDC, nb_simulation_per_point, NbPointsX, prefix_ADMC);
    // }
}

double intermediate_cost_function(double              donor_length,
                                  double              log_donor_level,
                                  std::vector<double> log_acceptor_levels,
                                  std::vector<double> parameters) {
    double              x_length         = 8.0;
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
    Device1D my_device;
    my_device.set_up_complex_diode(x_length,
                                   nb_points,
                                   donor_length,
                                   intrinsic_length,
                                   donor_level,
                                   intrinsic_level,
                                   acceptor_x,
                                   acceptor_levels);
    my_device.smooth_doping_profile(DopSmooth);

    double       target_anode_voltage  = 30.0;
    double       tol                   = 1.0e-8;
    const int    max_iter              = 1000;
    double       mcintyre_voltage_step = 0.25;
    const double stop_above_bv         = 5.0;
    double       BiasAboveBV           = 3.0;

    my_device.solve_poisson_and_mcintyre(target_anode_voltage, tol, max_iter, mcintyre_voltage_step, stop_above_bv);
    bool poisson_success = my_device.get_poisson_success();
    if (!poisson_success) {
        // fmt::print("Poisson failed\n");
        return BIG_DOUBLE;
    }
    std::size_t          iter_nb     = parameters[0];
    std::size_t          max_iter_nb = parameters[1];
    double               time        = iter_nb / static_cast<double>(max_iter_nb);
    cost_function_result cost_result = my_device.compute_cost_function(BiasAboveBV, time);

    // double               BV          = cost_result.result.BV;
    // double               BRP         = cost_result.result.BrP;
    // double               DW          = cost_result.result.DW;
    double cost = cost_result.total_cost;
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
void MainParticleSwarmSPAD(std::size_t nb_particles, std::size_t max_iter, double w, double c1, double c2) {
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
    double              min_length_donor = 0.1;
    double              max_length_donor = 4.0;
    double              min_doping       = 14.0;
    double              max_doping       = 19.0;
    double              donor_min_doping = DonorMIN;
    double              donor_max_doping = DonorMAX;
    std::vector<double> min_values(nb_parameters);
    std::vector<double> max_values(nb_parameters);
    set_up_bounds(min_length_donor, max_length_donor, donor_min_doping, donor_max_doping, min_doping, max_doping, min_values, max_values);

    double velocity_scaling = 0.1;
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
void MainSimulatedAnnealingSPAD(std::size_t nb_doe, std::size_t max_iter) {
    const std::string timestamp = fmt::format("{:%Y-%m-%d_%H-%M-%S}", fmt::localtime(std::time(nullptr)));
    const std::string DIR_RES   = fmt::format("results_sa/{}/", timestamp);
    if (!std::filesystem::exists(DIR_RES)) {
        std::filesystem::create_directories(DIR_RES);
    } else {
        std::filesystem::remove_all(DIR_RES);
        std::filesystem::create_directories(DIR_RES);
    }

    // Create simulated annealing object
    double          initial_temp     = 100000;
    double          final_temp       = 0.005;
    std::size_t     nb_parameters    = N_X + 2;
    CoolingSchedule cooling_schedule = CoolingSchedule::Geometrical;
    double          cooling_factor   = 0.98;

    SimulatedAnnealOptions sa_options;
    sa_options.m_nb_variables        = nb_parameters;
    sa_options.m_max_iterations      = max_iter;
    sa_options.m_initial_temperature = initial_temp;
    sa_options.m_final_temperature   = final_temp;
    sa_options.m_cooling_schedule    = cooling_schedule;
    sa_options.m_alpha_cooling       = cooling_factor;
    sa_options.m_beta_cooling        = 0.0;
    sa_options.m_log_frequency       = 10;
    // Boundaries setup
    double              min_length_donor = 0.1;
    double              max_length_donor = 4.0;
    double              min_doping       = 14.0;
    double              max_doping       = 19.0;
    double              donor_min_doping = DonorMIN;
    double              donor_max_doping = DonorMAX;
    std::vector<double> min_values(nb_parameters);
    std::vector<double> max_values(nb_parameters);
    set_up_bounds(min_length_donor, max_length_donor, donor_min_doping, donor_max_doping, min_doping, max_doping, min_values, max_values);

    std::cout << "Number DOE: " << nb_doe << std::endl;
    // Run simulated annealing with different initial solutions, one for each thread
    std::vector<std::vector<double>> initial_solutions(nb_doe);
    std::vector<std::vector<double>> final_solutions(nb_doe);
    std::vector<double>              final_costs(nb_doe);

    std::vector<SimulatedAnnealHistory> histories(nb_doe);

#pragma omp parallel for
    for (std::size_t idx_doe = 0; idx_doe < nb_doe; idx_doe++) {
        std::string directory        = fmt::format("{}/thread_{}/", DIR_RES, idx_doe);
        SimulatedAnnealOptions spec_sa_options(sa_options);
        spec_sa_options.m_prefix_name_log = directory;
        std::filesystem::create_directories(directory);
        SimulatedAnnealing  sa(spec_sa_options, cost_function_wrapper);
        std::vector<double> initial_solution = random_initial_position(min_values, max_values);
        sa.set_initial_solution(initial_solution);
        sa.run();

#pragma omp critical
        {
            // Save best solution
            sa.export_history();
            std::vector<double> best_solution = sa.get_best_solution();
            double              best_cost     = sa.get_best_cost();
            final_solutions[idx_doe]                = best_solution;
            final_costs[idx_doe]                    = best_cost;
            // Save all solutions
            histories[idx_doe] = sa.get_history();
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
    exit(0);
    // Get history of best path and save it
    std::cout << "Saving best path" << std::endl;
    std::vector<std::vector<double>> best_path = histories[index].solutions;
    export_best_path(best_path, new_path);
}

}  // namespace Optimization
