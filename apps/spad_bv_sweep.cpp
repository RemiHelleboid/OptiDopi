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
    std::size_t number_points = 500;
    double      length_donor  = 0.5;
    double      doping_donor  = atof(argv[1]);
    // double length_intrinsic = atof(argv[2]);
    double doping_intrinsic = 1.0e10;

    double min_acceptor           = 1.0e15;
    double max_acceptor           = 1.0e19;
    int    number_acceptor_points = 32;
    auto   list_doping_acceptor   = utils::geomspace(min_acceptor, max_acceptor, number_acceptor_points);
    int    nb_length_intrinsic    = 30;
    auto   list_length_intrisic   = utils::linspace(0.0, 5.0, nb_length_intrinsic);

    std::vector<std::vector<double>> BV_list(number_acceptor_points);

#pragma omp parallel for schedule(dynamic) num_threads(16)
    for (int i = 0; i < number_acceptor_points; ++i) {
        BV_list[i].resize(nb_length_intrinsic);
        const double doping_acceptor = list_doping_acceptor[i];
        std::cout << "Acceptor : " << doping_acceptor << std::endl;
        for (int idx_length = 0; idx_length < nb_length_intrinsic; ++nb_length_intrinsic) {
            const double length_intrinsic = list_length_intrisic[idx_length];
            device       my_device;
            my_device.setup_pin_diode(total_length,
                                      number_points,
                                      length_donor,
                                      length_intrinsic,
                                      doping_donor,
                                      doping_acceptor,
                                      doping_intrinsic);
            my_device.export_doping_profile("doping_profile.csv");
            double    target_anode_voltage = 400.0;
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
            fmt::print("Acceptor : {:3e} Breakdown voltage: {} V \n", doping_acceptor, BV);
            BV_list[i][idx_length] = BV;
        }
    }
    // Export BV list
    std::ofstream BV_file("BV_list.csv");
    BV_file << "Acceptor,L_intrinsic,BV" << std::endl;
    for (int i = 0; i < number_acceptor_points; ++i) {
        for (int idx_length = 0; idx_length < nb_length_intrinsic; ++nb_length_intrinsic) {
            BV_file << list_doping_acceptor[i] << "," << list_length_intrisic[idx_length] << "," << BV_list[i][idx_length]
                    << std::endl;
        }
    }
    BV_file.close();
}
