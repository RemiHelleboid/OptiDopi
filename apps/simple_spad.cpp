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
    std::size_t number_points    = 25*25;
    double      total_length     = 2.0;
    double      length_donor     = 1.;
    double      doping_donor     = 1.0e18;
    double      doping_intrinsic = 1.0e12;
    double      length_intrinsic = 0.0;

    double doping_acceptor = 1.0e18;

    Device1D my_device;
    std::string filename_doping = "";
    if (argc > 1) {
        filename_doping = argv[1];
        doping_profile doping_profile;
        doping_profile.load_doping_profile(filename_doping);
        my_device.add_doping_profile(doping_profile);
    } else {
        my_device.setup_pin_diode(total_length, number_points, length_donor, length_intrinsic, doping_donor, doping_acceptor, doping_intrinsic);
        my_device.smooth_doping_profile(11);
    }


    // Solve the Poisson and McIntyre equations
    double       target_anode_voltage  = 50.0;
    double       tol                   = 1.0e-8;
    const int    max_iter              = 1000;
    double       mcintyre_voltage_step = 0.25;
    const double stop_above_bv         = 15.0;
    double       BiasAboveBV           = 3.0;

    my_device.solve_poisson_and_mcintyre(target_anode_voltage, tol, max_iter, mcintyre_voltage_step, stop_above_bv);
    bool poisson_success = my_device.get_poisson_success();
    if (!poisson_success) {
        fmt::print("Poisson failed\n");
    }
    // Dir is POISSON_NL_X where X is the number of already existing folders
    int nb_dir = 0;
    while (std::filesystem::exists(fmt::format("POISSON_NL{}", nb_dir))) {
        nb_dir++;
    }
    // std::filesystem::create_directory("POISSON_NL2");
    // my_device.export_mcintyre_solution("POISSON_NL2/", "McIntyre_");
    // my_device.export_poisson_solution("POISSON_NL2/", "Poisson_");
    std::filesystem::create_directory(fmt::format("POISSON_NL{}", nb_dir));
    my_device.export_mcintyre_solution(fmt::format("POISSON_NL{}", nb_dir), "McIntyre_");
    my_device.export_poisson_solution(fmt::format("POISSON_NL{}", nb_dir), "Poisson_");
    my_device.export_doping_profile(fmt::format("POISSON_NL{}/DopingProfile.csv", nb_dir));

    cost_function_result cost_result = my_device.compute_cost_function(BiasAboveBV, 0.0);
    double               BV          = cost_result.result.BV;
    double               BRP         = cost_result.result.BrP;
    double               DW          = cost_result.result.DW;
    double               cost        = cost_result.total_cost;

    fmt::print("BV: {}, BRP: {}, DW: {}, cost: {}\n", BV, BRP, DW, cost);

    exit(0);

    fmt::print("Start Advection Diffusion Monte Carlo\n");
    double temperature = 300.0;
    double time_step   = 5.0e-14;
    double final_time  = 10.0e-9;

    ADMC::ParametersADMC parameters_admc;
    parameters_admc.m_time_step                  = time_step;
    parameters_admc.m_max_time                   = final_time;
    parameters_admc.m_temperature                = temperature;
    parameters_admc.m_activate_impact_ionization = true;
    parameters_admc.m_activate_particle_creation = true;
    parameters_admc.m_max_particles              = 500;
    parameters_admc.m_avalanche_threshold        = parameters_admc.m_max_particles;
    parameters_admc.m_output_file                = "ADMC_0/ADMC_0_";

    double voltage_AMDC = BV + BiasAboveBV;
    my_device.export_poisson_at_voltage_3D_emulation(voltage_AMDC, ".", "", 1.0, 1.0, 20, 20);

    // int ctn = 0;
    // while (ctn < 10000){
    //     // Clear folder
    //     std::filesystem::remove_all("ADMC_0");
    //     std::filesystem::create_directory("ADMC_0");
    //     ADMC::SimulationADMC simulation_admc(parameters_admc, my_device, BV+3);
    //     // simulation_admc.set_electric_field(voltage_AMDC);
    //     simulation_admc.AddElectrons(1, {2.5, 0.5, 0.5});
    //     simulation_admc.RunSimulation();
    //     std::size_t nb_particles = simulation_admc.get_number_of_particles();
    //     // if (nb_particles > 50 && nb_particles < 100){
    //     //     break;
    //     // }
    // }

    std::size_t nb_simulation_per_point = 1000;
    std::size_t NbPointsX               = 100;
    std::cout << "Start ADMC simulation" << std::endl;
    // ADMC::MainFullADMCSimulation(parameters_admc, &my_device, voltage_AMDC, nb_simulation_per_point, NbPointsX, "test1");
    my_device.DeviceADMCSimulation(parameters_admc,
                                   voltage_AMDC,
                                   nb_simulation_per_point,
                                   NbPointsX,
                                   fmt::format("SimpleSPAD_{:.2f}_", voltage_AMDC));
    // my_device.DeviceADMCSimulationToMaxField(parameters_admc,
    //                                          voltage_AMDC,
    //                                          nb_simulation_per_point,
    //                                          NbPointsX,
    //                                          fmt::format("SimpleSPAD_MField_{:.2f}_", voltage_AMDC));

    auto                          end             = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    fmt::print("Total time : {:.3f} s \n\n", elapsed_seconds.count());
}