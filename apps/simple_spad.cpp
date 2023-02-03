/**
 *
 */

#include <fmt/core.h>

#include "PoissonSolver.hpp"
#include "device.hpp"
#include "doping_profile.hpp"

int main(int argc, char** argv) {
    device my_device;
    my_device.setup_pin_diode(10.0, 1000, 2.0, 0.0, 1.0e19, 1.0e16, 1.0e10);
    my_device.export_doping_profile("doping_profile.csv");
    NewtonPoissonSolver poisson_solver(my_device.get_doping_profile());
    double              target_anode_voltage = 20.0;
    double              tol                  = 1.0e-6;
    const int           max_iter             = 100;
    double              voltage_step         = 0.01;
    poisson_solver.newton_solver(target_anode_voltage, tol, max_iter, voltage_step);
    fmt::print("End of the program {}.\n", argv[0]);
}