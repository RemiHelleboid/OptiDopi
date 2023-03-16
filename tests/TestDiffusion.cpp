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
#include "ImpactIonization.hpp"
#include "McIntyre.hpp"
#include "Mobility.hpp"
#include "PoissonSolver1D.hpp"
#include "Statistics.hpp"
#include "doctest/doctest.h"
#include "fill_vector.hpp"

TEST_CASE("Testing Diffusion at no field") {
    double               doping_level   = 1.0e10;
    const double         ElectricFieldX = 0.0;
    ADMC::ParametersADMC parameters_admc;
    parameters_admc.m_time_step                  = 1.0e-14;
    parameters_admc.m_max_time                   = 1.0e-9;
    parameters_admc.m_temperature                = 300.0;
    parameters_admc.m_activate_impact_ionization = true;
    parameters_admc.m_activate_particle_creation = false;
    parameters_admc.m_max_particles              = 1e5;
    parameters_admc.m_avalanche_threshold        = parameters_admc.m_max_particles;
    parameters_admc.m_output_file                = "ADMC_0/ADMC_0_";

    std::size_t nb_electrons = 250;

    ADMC::SimulationADMC simulation_admc(parameters_admc);
    simulation_admc.AddElectrons(nb_electrons, {0.0, 0.0, 0.0});
    std::cout << "Number of electrons: " << simulation_admc.get_number_of_electrons() << std::endl;

    simulation_admc.RunBULKSimulation(doping_level, ElectricFieldX);


    std::cout << "Mobility No Field Electron: " << physic::model::electron_mobility_arora_canali(1e10, 0.0, 300.0) << std::endl;
}

TEST_CASE("Testing Diffusion at no field holes") {
    double               doping_level   = 1.0e10;
    const double         ElectricFieldX = 0.0;
    ADMC::ParametersADMC parameters_admc;
    parameters_admc.m_time_step                  = 1.0e-14;
    parameters_admc.m_max_time                   = 1.0e-9;
    parameters_admc.m_temperature                = 300.0;
    parameters_admc.m_activate_impact_ionization = true;
    parameters_admc.m_activate_particle_creation = false;
    parameters_admc.m_max_particles              = 1e5;
    parameters_admc.m_avalanche_threshold        = parameters_admc.m_max_particles;
    parameters_admc.m_output_file                = "ADMC_0/ADMC_0_";

    std::size_t nb_electrons = 250;

    ADMC::SimulationADMC simulation_admc(parameters_admc);
    simulation_admc.AddHoles(nb_electrons, {0.0, 0.0, 0.0});

    simulation_admc.RunBULKSimulation(doping_level, ElectricFieldX);


    std::cout << "Mobility No Field Holes: " << physic::model::hole_mobility_arora_canali(1e10, 0.0, 300.0) << std::endl;
}