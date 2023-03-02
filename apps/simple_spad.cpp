/**
 *
 */

#include <fmt/core.h>

#include <iostream>
#include <filesystem>
#include <memory>
#include <random>

#include "McIntyre.hpp"
#include "PoissonSolver1D.hpp"
#include "Device1D.hpp"

#include "AdvectionDiffusionMC.hpp"

int main(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t number_points    = 1000;
    double      total_length     = 10.0;
    double      length_donor     = 1.0;
    double      doping_donor     = 5.0e19;
    double      doping_intrinsic = 1.0e13;
    double      length_intrinsic = 0.0;

    double doping_acceptor = 2.0e16;

    Device1D my_device;
    my_device.setup_pin_diode(total_length, number_points, length_donor, length_intrinsic, doping_donor, doping_acceptor, doping_intrinsic);
    my_device.smooth_doping_profile(5);

    // Solve the Poisson and McIntyre equations
    double       target_anode_voltage  = 100.0;
    double       tol                   = 1.0e-8;
    const int    max_iter              = 1000;
    double       mcintyre_voltage_step = 0.25;
    const double stop_above_bv         = 5.0;
    double       BiasAboveBV           = 3.0;

    my_device.solve_poisson_and_mcintyre(target_anode_voltage, tol, max_iter, mcintyre_voltage_step, stop_above_bv);
    bool poisson_success = my_device.get_poisson_success();
    if (!poisson_success) {
        fmt::print("Poisson failed\n");
    }
    cost_function_result cost_result = my_device.compute_cost_function(BiasAboveBV, 0.0);
    double               BV          = cost_result.result.BV;
    double               BRP         = cost_result.result.BrP;
    double               DW          = cost_result.result.DW;
    double               cost        = cost_result.total_cost;

    fmt::print("BV: {}, BRP: {}, DW: {}, cost: {}\n", BV, BRP, DW, cost);

    fmt::print("Start Advection Diffusion Monte Carlo\n");
    double       temperature = 300.0;
    double       time_step   = 1.0e-13;
    double       final_time  = 1.0e-9;

    ADMC::ParametersADMC parameters_admc;
    parameters_admc.m_time_step = time_step;
    parameters_admc.m_max_time  = final_time;
    parameters_admc.m_temperature = temperature;
    parameters_admc.m_max_particles = 100000;
    parameters_admc.m_output_file = "ADMC_0/ADMC_0_";
    std::filesystem::create_directory("ADMC_0");

    double voltage_AMDC = BV + BiasAboveBV;
    my_device.export_poisson_at_voltage_3D_emulation(voltage_AMDC, "ADMC_0/", "", 1.0, 1.0, 20, 20); 


    ADMC::SimulationADMC simulation_admc(parameters_admc, my_device);
    simulation_admc.set_electric_field(voltage_AMDC);
    simulation_admc.AddElectrons(1000, {3.25, 0.5, 0.5});
    simulation_admc.RunSimulation();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    fmt::print("Total time : {:.3f} s \n\n", elapsed_seconds.count());
}