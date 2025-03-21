/**
 *
 */

#include <fmt/core.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <random>

#include "AdvectionDiffusionMC.hpp"
#include "Device1D.hpp"
#include "McIntyre.hpp"
#include "PoissonSolver1D.hpp"

int main(int argc, char** argv) {
    std::cout << "Hello, world!" << std::endl;
    auto        start            = std::chrono::high_resolution_clock::now();
    std::size_t number_points    = 5000;
    double      total_length     = 5.0;
    double      length_donor     = 2.5;
    double      doping_intrinsic = 1.0e13;
    double      length_intrinsic = 0.0;

    double doping_donor    = 1.0e18;
    double doping_acceptor = 1.0e18;

    Device1D my_device;

    my_device.setup_pin_diode(total_length, number_points, length_donor, length_intrinsic, doping_donor, doping_acceptor, doping_intrinsic);
    // my_device.smooth_doping_profile(5);

    // Solve the Poisson and McIntyre equations
    double       target_anode_voltage  = 200.0;
    double       tol                   = 1.0e-9;
    const int    max_iter              = 1000;
    double       mcintyre_voltage_step = 0.25;
    const double stop_above_bv         = 15.0;
    double       BiasAboveBV           = 3.0;

    my_device.solve_poisson(target_anode_voltage, tol, max_iter);
    bool poisson_success = my_device.get_poisson_success();
    if (!poisson_success) {
        fmt::print("Poisson failed\n");
    }
    std::filesystem::create_directory("POISSON_NL");
    my_device.export_poisson_solution("POISSON_NL", "PN");
    
    auto                          end             = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    fmt::print("Total time : {:.3f} s \n\n", elapsed_seconds.count());
}