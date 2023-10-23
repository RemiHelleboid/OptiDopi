/**
 *
 */

#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/xchar.h>
#include <omp.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <random>

#include "AdvectionDiffusionMC.hpp"
#include "Device1D.hpp"
#include "McIntyre.hpp"
#include "PoissonSolver1D.hpp"
#include "fill_vector.hpp"

int main(int argc, char** argv) {
    fmt::print("Start of the program {}.\n", argv[0]);
    double      total_length     = 5.0;
    std::size_t number_points    = 350;
    double      length_donor     = 0.5;
    double      doping_donor     = atof(argv[1]);
    double      doping_intrinsic = 1.0e13;

    double min_acceptor           = 1.0e16;
    double max_acceptor           = 1.0e19;
    int    number_acceptor_points = 250;
    int    nb_length_intrinsic    = 250;
    auto   list_doping_acceptor   = utils::geomspace(min_acceptor, max_acceptor, number_acceptor_points);
    auto   list_length_intrisic   = utils::linspace(0.0, 1.0, nb_length_intrinsic);

    std::vector<std::vector<double>> BV_list(number_acceptor_points);
    std::vector<std::vector<double>> Breakdown_Probability_list(number_acceptor_points);
    std::vector<std::vector<double>> Depletion_Width_list(number_acceptor_points);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < number_acceptor_points; ++i) {
        BV_list[i].resize(nb_length_intrinsic);
        Breakdown_Probability_list[i].resize(nb_length_intrinsic);
        Depletion_Width_list[i].resize(nb_length_intrinsic);
        const double doping_acceptor = list_doping_acceptor[i];
        for (int idx_length = 0; idx_length < nb_length_intrinsic; ++idx_length) {
            const double length_intrinsic = list_length_intrisic[idx_length];
            Device1D     my_device;
            my_device.setup_pin_diode(total_length,
                                      number_points,
                                      length_donor,
                                      length_intrinsic,
                                      doping_donor,
                                      doping_acceptor,
                                      doping_intrinsic);
            my_device.smooth_doping_profile(10);
            my_device.export_doping_profile("doping_profile.csv");
            double       target_anode_voltage  = 150.0;
            double       tol                   = 1.0e-6;
            const int    max_iter              = 100;
            double       voltage_step          = 0.01;
            double       mcintyre_voltage_step = 0.25;
            const double stop_above_bv         = 5.0;
            double       BiasAboveBV           = 2.0;

            my_device.solve_poisson_and_mcintyre(target_anode_voltage, tol, max_iter, mcintyre_voltage_step, stop_above_bv);

            // my_device.export_poisson_solution("poisson_solution", "poisson_solution_");
            const double brp_threshold = 1e-3;
            double       BV            = my_device.extract_breakdown_voltage(brp_threshold);

            double BrP_at_Biasing            = my_device.get_brp_at_voltage(BV + BiasAboveBV);
            double DepletionWidth_at_Biasing = my_device.get_depletion_at_voltage(BV + BiasAboveBV);
            double max_electric_field        = my_device.get_poisson_solution_at_voltage(BV + BiasAboveBV).get_max_electric_field();
            if (omp_get_thread_num() != -1) {
                fmt::print("({}/{}) \t Acceptor : {:.2e} \t Intrinsic : {:.2e} \t BV : {:.2f} V \t BrP : {:.0f}% \t Depletion : {:.1f} \n",
                           i * number_acceptor_points + idx_length,
                           number_acceptor_points * nb_length_intrinsic,
                           doping_acceptor,
                           length_intrinsic,
                           BV,
                           BrP_at_Biasing * 100,
                           DepletionWidth_at_Biasing * 1.0e6);
            }

            BV_list[i][idx_length]                    = BV;
            Breakdown_Probability_list[i][idx_length] = BrP_at_Biasing;
            Depletion_Width_list[i][idx_length]       = DepletionWidth_at_Biasing;
        }
    }
    // Export BV list
    // Get the current time
    auto time = std::time(nullptr);
    // Convert to local time
    auto        local_time  = *std::localtime(&time);
    std::string time_string = fmt::format("{:%Y-%m-%d_%H-%M-%S}", local_time);
    std::string filename    = fmt::format("BV_list_{}.csv", time_string);

    std::ofstream BV_file(filename);
    BV_file << "Acceptor,L_intrinsic,BV,BrP,Depletion" << std::endl;
    for (int i = 0; i < number_acceptor_points; ++i) {
        for (int idx_length = 0; idx_length < nb_length_intrinsic; ++idx_length) {
            fmt::print(BV_file,
                       "{:3e},{:3e},{:3e},{:3e},{:3e}\n",
                       list_doping_acceptor[i],
                       list_length_intrisic[idx_length],
                       BV_list[i][idx_length],
                       Breakdown_Probability_list[i][idx_length],
                       Depletion_Width_list[i][idx_length]);
        }
    }
    BV_file.close();

    double poisson_time  = NewtonPoissonSolver::get_poisson_solver_time();
    double mcintyre_time = mcintyre::McIntyre::get_mcintyre_time();
    fmt::print("Total time spent in Poisson solver: {} s \n", poisson_time);
    fmt::print("Total time spent in McIntyre solver: {} s \n", mcintyre_time);
}
