/**
 * @file TestADMCBulk.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-03-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <fmt/core.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>

#include "AdvectionDiffusionMC.hpp"
#include "Device1D.hpp"
#include "McIntyre.hpp"
#include "PoissonSolver1D.hpp"
#include "doctest/doctest.h"
#include "fill_vector.hpp"
#include "Statistics.hpp"
#include "Mobility.hpp"
#include "ImpactIonization.hpp"


TEST_CASE("Testing velocity vs field.") {
    double doping_level = 1.0e10;

    std::vector<double> electric_fields = utils::geomspace(1.0e2, 1.0e6, 20);
    std::vector<double> list_x_velocity_mean(electric_fields.size());
    std::vector<double> list_x_velocity_std(electric_fields.size());
    std::vector<double> list_x_velocity_theoretical(electric_fields.size());

    fmt::print("Number of electric fields: {}\n", electric_fields.size());


#pragma omp parallel for schedule(dynamic)
    for (std::size_t idx_electric_field = 0; idx_electric_field < electric_fields.size(); ++idx_electric_field) {
        std::cout << "idx_electric_field = " << idx_electric_field << std::endl;
        const double ElectricFieldX = electric_fields[idx_electric_field];
        ADMC::ParametersADMC parameters_admc;
        parameters_admc.m_time_step                  = 1.0e-14;
        parameters_admc.m_max_time                   = 1.0e-10;
        parameters_admc.m_temperature                = 300.0;
        parameters_admc.m_activate_impact_ionization = true;
        parameters_admc.m_activate_particle_creation = false;
        parameters_admc.m_max_particles              = 1e5;
        parameters_admc.m_avalanche_threshold        = parameters_admc.m_max_particles;
        parameters_admc.m_output_file                = "ADMC_0/ADMC_0_";

        std::size_t nb_electrons = 100;

        if (ElectricFieldX < 1.0e3) {
            parameters_admc.m_max_time = 1.0e-9;
            nb_electrons *= 10;
        }


        std::cout << "Number of : " << parameters_admc.m_max_particles << std::endl;
        ADMC::SimulationADMC simulation_admc(parameters_admc);
        // Add 1 electron
        simulation_admc.AddElectrons(nb_electrons, {0.0, 0.0, 0.0});
        std::cout << "Number of electrons: " << simulation_admc.get_number_of_electrons() << std::endl;

        simulation_admc.RunBULKSimulation(doping_level, ElectricFieldX);

        std::vector<double> x_final = simulation_admc.get_all_x_positions();

        std::vector<double> x_velocity(nb_electrons);
        constexpr double micron_to_cm    = 1.0e-4;
        for (std::size_t i = 0; i < nb_electrons; i++) {
            x_velocity[i] = micron_to_cm * fabs(x_final[i]) / parameters_admc.m_max_time;
        }
        double           x_velocity_mean = utils::mean(x_velocity);
        double           x_velocity_std  = utils::standard_deviation(x_velocity);

        list_x_velocity_mean[idx_electric_field] = fabs(x_velocity_mean);
        list_x_velocity_std[idx_electric_field]  = fabs(x_velocity_std);
        
        double theoretical_velocity = physic::model::electron_mobility_arora_canali(doping_level, ElectricFieldX, parameters_admc.m_temperature) * ElectricFieldX;
        list_x_velocity_theoretical[idx_electric_field] = fabs(theoretical_velocity);

        std::cout << "ElectricFieldX = " << ElectricFieldX << " V/cm -> x_velocity_mean = " << fabs(x_velocity_mean) << " cm/s -> x_velocity_std = " << x_velocity_std << std::endl;
    }
    // Export data
    std::ofstream file("x_velocity_mean_vs_electric_field.csv");
    file << "ElectricFieldX,x_velocity_mean,x_velocity_std,AroraCanali" << std::endl;
    for (std::size_t idx_electric_field = 0; idx_electric_field < electric_fields.size(); ++idx_electric_field) {
        file << electric_fields[idx_electric_field] << "," << list_x_velocity_mean[idx_electric_field] << "," << list_x_velocity_std[idx_electric_field] << "," << list_x_velocity_theoretical[idx_electric_field] << std::endl;
    }
    file.close();
}


