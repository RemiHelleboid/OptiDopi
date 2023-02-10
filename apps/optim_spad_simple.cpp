#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/xchar.h>

#include <filesystem>

#include "PoissonSolver.hpp"
#include "SimulatedAnneal.hpp"
#include "device.hpp"
#include "doping_profile.hpp"
#include "fill_vector.hpp"
#include "omp.h"

// Set number of threads


double cost_function_formal(double BreakdownVoltage, double BreakdownProbability, double DepletionWidth) {
    double       BV_Target = 20.0;
    const double alpha_BV  = 100.0;
    const double alpha_BP  = 100.0;
    const double alpha_DW  = 10.0;
    double       BV_cost   = alpha_BV * std::pow((BreakdownVoltage - BV_Target) / BV_Target, 2);
    double       BP_cost   = alpha_BP * BreakdownProbability;
    double       DW_cost   = alpha_DW * DepletionWidth;
    if (std::isnan(BV_cost)) {
        BV_cost = -1.0e6;
    } else {
        BV_cost *= -1.0;
    }
    std::cout << "BV: " << BreakdownVoltage << " ----> BV cost: " << BV_cost << std::endl;
    // std::cout << "BP cost: " << BP_cost << std::endl;
    // std::cout << "DW cost: " << DW_cost << std::endl;
    double cost = DW_cost + BV_cost;
    return -cost;
}

double intermediate_cost_function(double length_intrinsic, double log_doping_acceptor, int thread_id=0) {
    std::size_t number_points    = 500;
    double      total_length     = 10.0;
    double      length_donor     = 0.5;
    double      doping_donor     = 5.0e19;
    double      doping_intrinsic = 1.0e13;

    double doping_acceptor = pow(10, log_doping_acceptor);

    device my_device;
    my_device.setup_pin_diode(total_length, number_points, length_donor, length_intrinsic, doping_donor, doping_acceptor, doping_intrinsic);
    my_device.smooth_doping_profile(10);
    std::string filename = fmt::format("results/doping_profile_{:.5e}_{:.5e}.csv", length_intrinsic, doping_acceptor);
    my_device.export_doping_profile(filename);

    double    target_anode_voltage = 50.0;
    double    tol                  = 1.0e-6;
    const int max_iter             = 100;
    double    voltage_step         = 0.01;
    my_device.solve_poisson(target_anode_voltage, tol, max_iter);
    // my_device.export_poisson_solution("poisson_solution", "poisson_solution_");

    const double stop_above_bv         = 5.0;
    double       mcintyre_voltage_step = 0.25;
    my_device.solve_mcintyre(mcintyre_voltage_step, stop_above_bv);
    const double brp_threshold = 1e-3;
    double       BV            = my_device.extract_breakdown_voltage(brp_threshold);
    double       BiasAboveBV   = 3.0;
    if (std::isnan(BV) || (BV + 1.5 * BiasAboveBV) > target_anode_voltage) {
        return 1.1e10;
    }

    // double BrP_at_Biasing            = my_device.get_brp_at_voltage(BV + BiasAboveBV);

    // double BV                        = 20.0;
    double BrP_at_Biasing            = 1.0e-3;
    double meter_to_micron           = 1.0e6;
    double DepletionWidth_at_Biasing = my_device.get_depletion_at_voltage(BV + BiasAboveBV) * meter_to_micron;
    fmt::print("BV: {:.5e}, BrP: {:.5e}, DW: {:.5e}\n", BV, BrP_at_Biasing, DepletionWidth_at_Biasing);
    double cost = cost_function_formal(BV, BrP_at_Biasing, DepletionWidth_at_Biasing);

    return cost;
}

double cost_function(std::vector<double> variables) {
    double length_intrinsic = variables[0];
    double doping_acceptor  = variables[1];
    double cost             = intermediate_cost_function(length_intrinsic, doping_acceptor);
    fmt::print("Doping: {:.5e}, Length: {:.5e}, Cost: {:.5e}\n", pow(10, doping_acceptor), length_intrinsic, cost);
    return cost;
}



void create_map_cost_function(std::string filename) {
    std::vector<double> length_intrinsic = utils::linspace(0.0, 4.0, 50);
    std::vector<double> doping_acceptor  = utils::linspace(16.0, 19.0, 50);
    std::vector<std::vector<double>> cost_function(length_intrinsic.size(), std::vector<double>(doping_acceptor.size(), 0.0));
#pragma omp parallel for num_threads(16)
    for (std::size_t i = 0; i < length_intrinsic.size(); i++) {
        for (std::size_t j = 0; j < doping_acceptor.size(); j++) {
            cost_function[i][j] = intermediate_cost_function(length_intrinsic[i], doping_acceptor[j]);
        }
    }
    // Write to file with fmt
    std::ofstream file(filename);
    fmt::print(file, "length_intrinsic,doping_acceptor,cost\n");
    for (std::size_t i = 0; i < length_intrinsic.size(); i++) {
        for (std::size_t j = 0; j < doping_acceptor.size(); j++) {
            fmt::print(file, "{:.5e},{:.5e},{:.5e}\n", length_intrinsic[i], doping_acceptor[j], cost_function[i][j]);
        }
    }
    file.close();
}


int main() {
    omp_set_num_threads(16);
    std::cout << "Simulated Annealing for SPAD optimization" << std::endl;

    std::string DIR_RES = "results";
    if (!std::filesystem::exists(DIR_RES)) {
        std::filesystem::create_directory(DIR_RES);
    } else {
        std::filesystem::remove_all(DIR_RES);
        std::filesystem::create_directory(DIR_RES);
    }
    create_map_cost_function("main_cost_function.csv");

    exit(0);

    // Create simulated annealing object
    std::size_t     max_iter         = 250;
    double          initial_temp     = 10;
    double          final_temp       = 0.001;
    std::size_t     nb_parameters    = 2;
    CoolingSchedule cooling_schedule = CoolingSchedule::Geometrical;

    double min_doping = 5.0e16;
    double max_doping = 1.0e19;
    double min_length = 0.0;
    double max_length = 1.0;

    int nb_threads = 4;
    std::cout << "Number of threads: " << nb_threads << std::endl;
    // Run simulated annealing with different initial solutions, one for each thread
    std::vector<std::vector<double>> initial_solutions(nb_threads);
    std::vector<std::vector<double>> final_solutions(nb_threads);
    std::vector<double>              final_costs(nb_threads);

#pragma omp parallel for num_threads(nb_threads)
    for (int i = 0; i < nb_threads; i++) {
        std::cout << "Thread " << i << std::endl;
        std::string directory = fmt::format("{}/thread_{}", DIR_RES, i);
        std::filesystem::create_directory(directory);

        SimulatedAnnealing sa(nb_parameters, cooling_schedule, max_iter, initial_temp, final_temp, cost_function);
        sa.set_prefix_name(fmt::format("{}/thread_{}_", DIR_RES, i));
        sa.set_bounds({{min_length, max_length}, {log10(min_doping), log10(max_doping)}});
        sa.set_alpha_cooling(0.99);
        double length_intrinsic = 0.01;
        double doping_acceptor  = 5.0e16;
        // sa.set_initial_solution({length_intrinsic, log10(doping_acceptor)});
        sa.create_random_initial_solution();
        sa.run();
        // Get best solution
        std::vector<double> best_solution = sa.get_best_solution();
        double              best_cost     = sa.get_best_cost();

        // Print best solution
        fmt::print("Best solution: {}\n", best_solution);
        fmt::print("Best cost: {}\n", best_cost);

        final_solutions[i] = best_solution;
        final_costs[i]     = best_cost;
    }

    // Find best solution
    std::vector<double>::iterator it    = std::min_element(final_costs.begin(), final_costs.end());
    int                           index = std::distance(final_costs.begin(), it);
    std::cout << "*****************************************************" << std::endl;
    std::cout << "Best cost: " << final_costs[index] << std::endl;
    std::cout << "Best solution: " << fmt::format("{:.5e}, {:.5e}", final_solutions[index][0], pow(10, final_solutions[index][1]))
              << std::endl;

    // Run device simulation with best solution
    double length_intrinsic    = final_solutions[index][0];
    double log_doping_acceptor = final_solutions[index][1];
    std::cout << "Length: " << length_intrinsic << std::endl;
    std::cout << "Doping: " << log_doping_acceptor << std::endl;
    intermediate_cost_function(length_intrinsic, log_doping_acceptor);
}