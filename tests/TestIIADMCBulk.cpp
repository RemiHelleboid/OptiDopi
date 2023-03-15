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

TEST_CASE("Testing Impact Ionization vs field.") {
    double doping_level = 1.0e10;

    std::vector<double> electric_fields = utils::geomspace(1.0e5, 1.0e6, 16);
    std::vector<double> list_impact_ionization_rate(electric_fields.size());
    std::vector<double> list_impact_ionization_rate_theoretical(electric_fields.size());

    fmt::print("Number of electric fields: {}\n", electric_fields.size());
    double GammaII = mcintyre::compute_gamma(300.0);
    double EgII    = mcintyre::compute_band_gap(300.0);

#pragma omp parallel for schedule(dynamic)
    for (std::size_t idx_electric_field = 0; idx_electric_field < electric_fields.size(); ++idx_electric_field) {
        const double         ElectricFieldX = electric_fields[idx_electric_field];
        ADMC::ParametersADMC parameters_admc;
        parameters_admc.m_time_step                  = 1.0e-14;
        parameters_admc.m_max_time                   = 1.0e-9;
        parameters_admc.m_temperature                = 300.0;
        parameters_admc.m_activate_impact_ionization = true;
        parameters_admc.m_activate_particle_creation = false;
        parameters_admc.m_max_particles              = 1e5;
        parameters_admc.m_avalanche_threshold        = parameters_admc.m_max_particles;
        parameters_admc.m_output_file                = "ADMC_0/ADMC_0_";

        std::size_t nb_electrons = 40;

        ADMC::SimulationADMC simulation_admc(parameters_admc);
        // Add 1 electron
        simulation_admc.AddElectrons(nb_electrons, {0.0, 0.0, 0.0});
        simulation_admc.RunBULKSimulation(doping_level, ElectricFieldX);
        std::vector<double> impact_ionization_rates                 = simulation_admc.compute_impact_ionization_coeff();
        double              impact_ionization_rate                  = utils::mean(impact_ionization_rates);
        list_impact_ionization_rate[idx_electric_field]             = impact_ionization_rate * 1.0e4;

        double ElectricFieldXVMicron = ElectricFieldX * 1.0e-4;
        const double theoretical_impact_ionization_rate             = mcintyre::alpha_DeMan(ElectricFieldXVMicron, GammaII, EgII);
        list_impact_ionization_rate_theoretical[idx_electric_field] = theoretical_impact_ionization_rate * 1.0e4;
        std::cout << "ElectricFieldX = " << ElectricFieldX << " V/cm -> impact_ionization_rate = " << impact_ionization_rate << std::endl;
    }
    // Export data
    std::ofstream file("impact_ionization_rate_vs_electric_field.csv");
    file << "ElectricFieldX,impact_ionization_rate,DeMan" << std::endl;
    for (std::size_t idx_electric_field = 0; idx_electric_field < electric_fields.size(); ++idx_electric_field) {
        file << electric_fields[idx_electric_field] << "," << list_impact_ionization_rate[idx_electric_field] << ","
             << list_impact_ionization_rate_theoretical[idx_electric_field] << std::endl;
    }
    file.close();
}

TEST_CASE("Testing Impact Ionization vs field for Holes.") {
    double doping_level = 1.0e10;

    std::vector<double> electric_fields = utils::geomspace(1.0e5, 1.0e6, 16);
    std::vector<double> list_impact_ionization_rate(electric_fields.size());
    std::vector<double> list_impact_ionization_rate_theoretical(electric_fields.size());

    fmt::print("Number of electric fields: {}\n", electric_fields.size());
    double GammaII = mcintyre::compute_gamma(300.0);
    double EgII    = mcintyre::compute_band_gap(300.0);

#pragma omp parallel for schedule(dynamic)
    for (std::size_t idx_electric_field = 0; idx_electric_field < electric_fields.size(); ++idx_electric_field) {
        const double         ElectricFieldX = electric_fields[idx_electric_field];
        ADMC::ParametersADMC parameters_admc;
        parameters_admc.m_time_step                  = 1.0e-14;
        parameters_admc.m_max_time                   = 1.0e-8;
        parameters_admc.m_temperature                = 300.0;
        parameters_admc.m_activate_impact_ionization = true;
        parameters_admc.m_activate_particle_creation = false;
        parameters_admc.m_max_particles              = 1e5;
        parameters_admc.m_avalanche_threshold        = parameters_admc.m_max_particles;
        parameters_admc.m_output_file                = "ADMC_0/ADMC_0_";

        std::size_t nb_holes = 40;

        ADMC::SimulationADMC simulation_admc(parameters_admc);
        // Add 1 electron
        simulation_admc.AddHoles(nb_holes, {0.0, 0.0, 0.0});
        simulation_admc.RunBULKSimulation(doping_level, ElectricFieldX);
        std::vector<double> impact_ionization_rates                 = simulation_admc.compute_impact_ionization_coeff();
        double              impact_ionization_rate                  = utils::mean(impact_ionization_rates);
        list_impact_ionization_rate[idx_electric_field]             = impact_ionization_rate * 1.0e4;

        double ElectricFieldXVMicron = ElectricFieldX * 1.0e-4;
        const double theoretical_impact_ionization_rate             = mcintyre::beta_DeMan(ElectricFieldXVMicron, GammaII, EgII);
        list_impact_ionization_rate_theoretical[idx_electric_field] = theoretical_impact_ionization_rate * 1.0e4;
        std::cout << "ElectricFieldX = " << ElectricFieldX << " V/cm -> impact_ionization_rate = " << impact_ionization_rate << std::endl;
    }
    // Export data
    std::ofstream file("hole_impact_ionization_rate_vs_electric_field.csv");
    file << "ElectricFieldX,impact_ionization_rate,DeMan" << std::endl;
    for (std::size_t idx_electric_field = 0; idx_electric_field < electric_fields.size(); ++idx_electric_field) {
        file << electric_fields[idx_electric_field] << "," << list_impact_ionization_rate[idx_electric_field] << ","
             << list_impact_ionization_rate_theoretical[idx_electric_field] << std::endl;
    }
    file.close();
}
