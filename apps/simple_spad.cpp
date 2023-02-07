/**
 *
 */

#include <iostream>
#include <memory>
#include <random>
#include <fmt/core.h>

#include "PoissonSolver.hpp"
#include "device.hpp"
#include "doping_profile.hpp"
#include "McIntyre.hpp"

int main(int argc, char** argv) {
    const std::string OUTDIR = "output/";
    device my_device;
    my_device.setup_pin_diode(10.0, 500, 2.0, 0.0, 1.0e19, 5.0e17, 1.0e10);
    my_device.export_doping_profile("doping_profile.csv");
    double              target_anode_voltage = 40.0;
    double              tol                  = 1.0e-6;
    const int           max_iter             = 100;
    double              voltage_step         = 0.01;
    my_device.solve_poisson(target_anode_voltage, tol, max_iter);
    // my_device.export_poisson_solution("poisson_solution", "poisson_solution_");

    double mcintyre_voltage_step = 0.5;
    my_device.solve_mcintyre(mcintyre_voltage_step);
    // my_device.export_mcintyre_solution("mcintyre_solution", "MCI_");
    const double brp_threshold = 1e-3;
    double BV = my_device.extract_breakdown_voltage(brp_threshold);
    fmt::print("Breakdown voltage: {} V (threshold = {})\n", BV, brp_threshold);
    my_device.export_depletion_width(OUTDIR, "depletion_width.csv");

    double BiasAboveBV = 3.0;
    double BrP_at_Biasing = my_device.get_brp_at_voltage(BV + BiasAboveBV);
    double DepletionWidth_at_Biasing = my_device.get_depletion_at_voltage(BV + BiasAboveBV);

    fmt::print("Biasing voltage: {} V \n", BV + BiasAboveBV);
    fmt::print("\t- Breakdown Probability: {} \n", BrP_at_Biasing);
    fmt::print("\t- Depletion width: {} um \n", DepletionWidth_at_Biasing * 1e6);


    double poisson_time = NewtonPoissonSolver::get_poisson_solver_time();
    double mcintyre_time = mcintyre::McIntyre::get_mcintyre_time();
    fmt::print("Total time spent in Poisson solver: {} s \n", poisson_time);
    fmt::print("Total time spent in McIntyre solver: {} s \n", mcintyre_time);
}