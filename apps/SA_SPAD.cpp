#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/xchar.h>

#include <chrono>
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

#define N_X 6

void export_best_path(std::vector<std::vector<double>> best_path, std::string dirname) {
    std::filesystem::create_directories(dirname);
    const std::string poisson_dir = dirname + "/poisson_solution/";
    std::filesystem::create_directories(poisson_dir);
    

    double              x_length        = 10.0;
    std::size_t         nb_points       = 500;
    double              donor_length    = 1.0;
    double              intrisic_length = 0.0;
    double              donor_level     = 5.0e19;
    double              intrisic_level  = 1.0e13;
    std::vector<double> acceptor_x      = {1.0, 1.50, 2.0, 3.0, 5.0, 10.0};

    std::size_t nb_iter = best_path.size();
    std::vector<double> list_bv(nb_iter);
    std::vector<double> list_brp(nb_iter);
    std::vector<double> list_dw(nb_iter);

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
        const std::string filename = fmt::format("{}/doping_profile_{:03d}.csv", dirname, i);
        my_device.export_doping_profile(filename);

        double       target_anode_voltage  = 30.0;
        double       tol                   = 1.0e-6;
        const int    max_iter              = 100;
        double       voltage_step          = 0.01;
        double       mcintyre_voltage_step = 0.25;
        const double stop_above_bv         = 5.0;
        double       BiasAboveBV           = 3.0;

        my_device.solve_poisson_and_mcintyre(target_anode_voltage, tol, max_iter, mcintyre_voltage_step, stop_above_bv);
        double BV = my_device.extract_breakdown_voltage(1.0e-4);
        double BVPLUS = BV + BiasAboveBV;
        my_device.export_poisson_solution_at_voltage(BVPLUS, poisson_dir, fmt::format("poisson_solution_{:03d}_", i));
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
    std::vector<double> acceptor_x = {1.0, 1.50, 2.0, 3.0, 5.0, 10.0};

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
    // fmt::print("Cost: {:.5e}\n", cost);
    return cost;
}

double cost_function(std::vector<double> variables) {
    // Call the intermediate cost function
    double cost = intermediate_cost_function(variables);
    // fmt::print("Doping: {:.5e}, Length: {:.5e}, Cost: {:.5e}\n", pow(10, doping_acceptor), length_intrinsic, cost);
    return cost;
}

int main(int argc, const char** argv) {
    const std::string timestamp = fmt::format("{:%Y-%m-%d_%H-%M-%S}", fmt::localtime(std::time(nullptr)));
    const std::string DIR_RES   = fmt::format("results_sa/{}/", timestamp);
    if (!std::filesystem::exists(DIR_RES)) {
        std::filesystem::create_directories(DIR_RES);
    } else {
        std::filesystem::remove_all(DIR_RES);
        std::filesystem::create_directories(DIR_RES);
    }

    // Create simulated annealing object
    std::size_t     max_iter         = 20;
    double          initial_temp     = 500;
    double          final_temp       = 0.001;
    std::size_t     nb_parameters    = N_X;
    CoolingSchedule cooling_schedule = CoolingSchedule::Geometrical;

    double              min_doping = 1.0e13;
    double              max_doping = 1.0e19;
    std::vector<double> min_values(N_X, log10(min_doping));
    std::vector<double> max_values(N_X, log10(max_doping));

    std::size_t nb_threads = 1;
#pragma omp parallel
    { nb_threads = omp_get_num_threads(); }
    std::cout << "Number threads: " << nb_threads << std::endl;
    // Run simulated annealing with different initial solutions, one for each thread
    std::vector<std::vector<double>> initial_solutions(nb_threads);
    std::vector<std::vector<double>> final_solutions(nb_threads);
    std::vector<double>              final_costs(nb_threads);

    std::vector<SimulatedAnnealHistory> histories(nb_threads);

#pragma omp parallel for schedule(dynamic) num_threads(nb_threads)
    for (int i = 0; i < nb_threads; i++) {
        std::string directory = fmt::format("{}/thread_{}/", DIR_RES, i);
        std::filesystem::create_directory(directory);

        SimulatedAnnealing sa(nb_parameters, cooling_schedule, max_iter, initial_temp, final_temp, cost_function);
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
            final_solutions[i] = best_solution;
            final_costs[i]     = best_cost;
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
    std::string new_path = fmt::format("{}/best_path_{}/", DIR_RES, index);
    std::filesystem::create_directory(new_path);
    std::filesystem::copy(best_path_folder, DIR_RES, std::filesystem::copy_options::recursive);


    // Get history of best path and save it
    std::vector<std::vector<double>> best_path = histories[index].solutions;
    export_best_path(best_path, DIR_RES + "best_path/");

    // Run device simulation with best solution

    double poisson_time              = NewtonPoissonSolver::get_poisson_solver_time();
    double mcintyre_time             = mcintyre::McIntyre::get_mcintyre_time();
    double ration_converged_mcintyre = mcintyre::McIntyre::get_ratio_converged_sim() * 100.0;
    fmt::print("Total time spent in Poisson solver: {} s \n", poisson_time);
    fmt::print("Total time spent in McIntyre solver: {} s \n", mcintyre_time);
    fmt::print("Ratio of converged simulations: {:.2f}% \n", ration_converged_mcintyre);
    return 0;
}
