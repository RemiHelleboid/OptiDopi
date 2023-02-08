/**
 *
 */

#include <fmt/core.h>

#include "PoissonSolver.hpp"
#include "device.hpp"
#include "doping_profile.hpp"
#include "fill_vector.hpp"

int main(int argc, char** argv) {
    fmt::print("Start of the program {}.\n", argv[0]);
    double      total_length  = 10.0;
    std::size_t number_points = 350;
    double      length_donor  = 0.5;
    double      doping_donor  = atof(argv[1]);
    // double length_intrinsic = atof(argv[2]);
    double doping_intrinsic = 1.0e13;

    double min_acceptor           = 5.0e16;
    double max_acceptor           = 1.0e19;
    int    number_acceptor_points = 16;
    auto   list_doping_acceptor   = utils::geomspace(min_acceptor, max_acceptor, number_acceptor_points);
    int    nb_length_intrinsic    = 16;
    auto   list_length_intrisic   = utils::linspace(0.0, 1.0, nb_length_intrinsic);

    std::vector<std::vector<double>> BV_list(number_acceptor_points);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < number_acceptor_points; ++i) {
        BV_list[i].resize(nb_length_intrinsic);
        const double doping_acceptor = list_doping_acceptor[i];
        for (int idx_length = 0; idx_length < nb_length_intrinsic; ++idx_length) {
            const double length_intrinsic = list_length_intrisic[idx_length];
            device       my_device;
            my_device.setup_pin_diode(total_length,
                                      number_points,
                                      length_donor,
                                      length_intrinsic,
                                      doping_donor,
                                      doping_acceptor,
                                      doping_intrinsic);
            my_device.smooth_doping_profile(10);
            my_device.export_doping_profile("doping_profile.csv");
            double    target_anode_voltage = 100.0;
            double    tol                  = 1.0e-6;
            const int max_iter             = 100;
            double    voltage_step         = 0.01;
            my_device.solve_poisson(target_anode_voltage, tol, max_iter);
            // my_device.export_poisson_solution("poisson_solution", "poisson_solution_");

            const double stop_above_bv         = 3.0;
            double       mcintyre_voltage_step = 0.25;
            my_device.solve_mcintyre(mcintyre_voltage_step, stop_above_bv);
            const double brp_threshold = 1e-3;
            double       BV            = my_device.extract_breakdown_voltage(brp_threshold);
            BV_list[i][idx_length]     = BV;

            double BiasAboveBV               = 3.0;
            double BrP_at_Biasing            = my_device.get_brp_at_voltage(BV + BiasAboveBV);
            double DepletionWidth_at_Biasing = my_device.get_depletion_at_voltage(BV + BiasAboveBV);
            fmt::print("Acceptor : {:3e} \t Intrinsic : {:3e} \t BV : {:3e} \t BrP : {:3e} \t Depletion : {:3e} \n",
                       doping_acceptor,
                       length_intrinsic,
                       BV,
                       BrP_at_Biasing,
                       DepletionWidth_at_Biasing);
        }
    }
    // Export BV list
    std::ofstream BV_file("BV_list.csv");
    BV_file << "Acceptor,L_intrinsic,BV" << std::endl;
    for (int i = 0; i < number_acceptor_points; ++i) {
        for (int idx_length = 0; idx_length < nb_length_intrinsic; ++idx_length) {
            BV_file << list_doping_acceptor[i] << "," << list_length_intrisic[idx_length] << "," << BV_list[i][idx_length] << std::endl;
        }
    }
    BV_file.close();

    double poisson_time  = NewtonPoissonSolver::get_poisson_solver_time();
    double mcintyre_time = mcintyre::McIntyre::get_mcintyre_time();
    fmt::print("Total time spent in Poisson solver: {} s \n", poisson_time);
    fmt::print("Total time spent in McIntyre solver: {} s \n", mcintyre_time);
}
