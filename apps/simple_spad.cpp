/**
 *
 */

#include <fmt/core.h>

#include "PoissonSolver.hpp"
#include "device.hpp"
#include "doping_profile.hpp"

int main(int argc, char** argv) {
    device my_device;
    my_device.setup_pin_diode(10.0, 500, 2.0, 0.0, 1.0e19, 5.0e16, 1.0e10);
    my_device.export_doping_profile("doping_profile.csv");
    double              target_anode_voltage = 40.0;
    double              tol                  = 1.0e-6;
    const int           max_iter             = 100;
    double              voltage_step         = 0.01;
    my_device.solve_poisson(target_anode_voltage, tol, max_iter);
    my_device.export_poisson_solution("poisson_solution", "poisson_solution_");

    double mcintyre_voltage_step = 0.5;
    my_device.solve_mcintyre(voltage_step);
    my_device.export_mcintyre_solution("mcintyre_solution", "MCI_");
    const double brp_threshold = 1e-3;
    double BV = my_device.extract_breakdown_voltage(brp_threshold);
    fmt::print("Breakdown voltage: {} V (threshold = {})\n", BV, brp_threshold);

    fmt::print("End of the program {}.\n", argv[0]);
}