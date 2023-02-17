#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/xchar.h>

#include <filesystem>
#include <numeric>

#include "PoissonSolver.hpp"
#include "SimulatedAnneal.hpp"
#include "ImpactIonization.hpp"
#include "device.hpp"
#include "doping_profile.hpp"
#include "fill_vector.hpp"
#include "omp.h"

// Set number of threads

#define NAN_DOUBLE std::numeric_limits<double>::quiet_NaN()
static int idx = 0;

// Static file for log
static std::string filename = "optim_path_full.csv";
std::ofstream file_path(filename);

struct result_sim {
    double               length_intrinsic;
    double               doping_acceptor;
    double               BV;
    double               BrP;
    double               DW;
    cost_function_result cost_result;

    result_sim(double length_intrinsic, double doping_acceptor, double BV, double BrP, double DW, cost_function_result cost) {
        this->length_intrinsic = length_intrinsic;
        this->doping_acceptor  = doping_acceptor;
        this->BV               = BV;
        this->BrP              = BrP;
        this->DW               = DW;
        this->cost_result      = cost;
    }
};

struct mapping_cost_function {
    std::vector<double>              length_intrinsic;
    std::vector<double>              doping_acceptor;
    std::vector<std::vector<double>> breakdown_voltage;
    std::vector<std::vector<double>> breakdown_probability;
    std::vector<std::vector<double>> depletion_width;
    std::vector<std::vector<double>> cost_function;
};

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
    std::string filename = fmt::format("results/thread_{}/doping_profile_{}.csv", thread_id, idx);
    my_device.export_doping_profile(filename);

    double    target_anode_voltage = 40.0;
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
    fmt::print(file_path, "{:.5e},{:.5e},{:.5e}\n", BV, BrP_at_Biasing, DepletionWidth_at_Biasing);

    cost_function_result cost_resultr = my_device.compute_cost_function(BiasAboveBV);
    result_sim           full_result(length_intrinsic, doping_acceptor, BV, BrP_at_Biasing, DepletionWidth_at_Biasing, cost_resultr);

    return full_result;
}

void create_map_cost_function(std::string filename) {
    int                 Ndop             = 25;
    int                 Nlen             = 25;
    double              min_doping       = 16;
    double              max_doping       = 19;
    double              min_length       = 0.0;
    double              max_length       = 1.0;
    std::vector<double> length_intrinsic = utils::linspace(min_length, max_length, Nlen);
    std::vector<double> doping_acceptor  = utils::linspace(min_doping, max_doping, Ndop);

    std::vector<std::vector<double>> BV(Ndop, std::vector<double>(Nlen, 0.0));
    std::vector<std::vector<double>> BP(Ndop, std::vector<double>(Nlen, 0.0));
    std::vector<std::vector<double>> DW(Ndop, std::vector<double>(Nlen, 0.0));
    std::vector<std::vector<double>> BV_COST(Ndop, std::vector<double>(Nlen, 0.0));
    std::vector<std::vector<double>> BP_COST(Ndop, std::vector<double>(Nlen, 0.0));
    std::vector<std::vector<double>> DW_COST(Ndop, std::vector<double>(Nlen, 0.0));
    std::vector<std::vector<double>> COST(Ndop, std::vector<double>(Nlen, 0.0));

    std::cout << "Start computation over " << length_intrinsic.size() * doping_acceptor.size() << " points." << std::endl;
    int total_done = 0;
#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < length_intrinsic.size(); i++) {
        for (std::size_t j = 0; j < doping_acceptor.size(); j++) {
            // std::cout << "Intrinsic length: " << length_intrinsic[i] << ", doping: " << doping_acceptor[j] << std::endl;
            result_sim res = intermediate_cost_function(length_intrinsic[i], doping_acceptor[j]);
            BV[i][j]       = res.BV;
            BP[i][j]       = res.BrP;
            DW[i][j]       = res.DW;
            BV_COST[i][j]  = res.cost_result.BV_cost;
            BP_COST[i][j]  = res.cost_result.BP_cost;
            DW_COST[i][j]  = res.cost_result.DW_cost;
            COST[i][j]     = res.cost_result.total_cost;
            total_done++;
            if (total_done % 100 == 0) {
                std::cout << "Done: " << total_done << std::endl;
            }
        }
    }
    // Write to file with fmt
    std::ofstream file(filename);
    fmt::print(file, "length_intrinsic,doping_acceptor,BV,BP,DW,BV_COST,BP_COST,DW_COST,COST\n");
    for (std::size_t i = 0; i < length_intrinsic.size(); i++) {
        for (std::size_t j = 0; j < doping_acceptor.size(); j++) {
            fmt::print(file,
                       "{:.5e},{:.5e},{:.5e},{:.5e},{:.5e},{:.5e},{:.5e},{:.5e},{:.5e}\n",
                       length_intrinsic[i],
                       doping_acceptor[j],
                       BV[i][j],
                       BP[i][j],
                       DW[i][j],
                       BV_COST[i][j],
                       BP_COST[i][j],
                       DW_COST[i][j],
                       COST[i][j]);
        }
    }
}

double cost_function(std::vector<double> variables) {
    double length_intrinsic = variables[0];
    double doping_acceptor  = variables[1];
    double cost             = intermediate_cost_function(length_intrinsic, doping_acceptor).cost_result.total_cost;
    // fmt::print("Doping: {:.5e}, Length: {:.5e}, Cost: {:.5e}\n", pow(10, doping_acceptor), length_intrinsic, cost);
    return cost;
}

int main() {
    std::cout << "Simulated Annealing for SPAD optimization" << std::endl;

    create_map_cost_function("main_cost_function2.csv");

    file_path << "BV,BrP,DW\n";

    // mcintyre::export_to_file_impact_rates("impact_rates.csv");

    std::string DIR_RES = "results";
    if (!std::filesystem::exists(DIR_RES)) {
        std::filesystem::create_directory(DIR_RES);
    } else {
        std::filesystem::remove_all(DIR_RES);
        std::filesystem::create_directory(DIR_RES);
    }

    // Create simulated annealing object
    std::size_t     max_iter         = 500;
    double          initial_temp     = 500;
    double          final_temp       = 0.001;
    std::size_t     nb_parameters    = 2;
    CoolingSchedule cooling_schedule = CoolingSchedule::Geometrical;

    double min_doping = 1.0e16;
    double max_doping = 1.0e19;
    double min_length = 0.0;
    double max_length = 2.0;

    int nb_threads = 1;
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
        sa.set_prefix_name(fmt::format("{}/thread_{}/", DIR_RES, i));
        sa.set_bounds({{min_length, max_length}, {log10(min_doping), log10(max_doping)}});
        sa.set_alpha_cooling(0.98);
        double length_intrinsic = 0.01;
        double doping_acceptor  = 5.0e16;
        sa.set_initial_solution({2.0, log10(1.0e16)});
        // sa.create_random_initial_solution();
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




    double poisson_time  = NewtonPoissonSolver::get_poisson_solver_time();
    double mcintyre_time = mcintyre::McIntyre::get_mcintyre_time();
    double ration_converged_mcintyre = mcintyre::McIntyre::get_ratio_converged_sim() * 100.0;
    fmt::print("Total time spent in Poisson solver: {} s \n", poisson_time);
    fmt::print("Total time spent in McIntyre solver: {} s \n", mcintyre_time);
    fmt::print("Ratio of converged simulations: {:.2f}% \n", ration_converged_mcintyre);
    file_path.close();
    exit(0);


}