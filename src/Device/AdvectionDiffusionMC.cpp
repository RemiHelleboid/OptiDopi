/**
 * @file AdvectionDiffusionMC.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-03-01
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "AdvectionDiffusionMC.hpp"

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <filesystem>

#include "Device1D.hpp"
#include "Mobility.hpp"
#include "Statistics.hpp"
#include "fill_vector.hpp"
#include "interpolation.hpp"

namespace ADMC {

SimulationADMC::SimulationADMC(const ParametersADMC& parameters) : m_parameters(parameters), m_device{} {
    std::random_device rd;
    m_generator.seed(rd());
    m_distribution_uniform = std::uniform_real_distribution<double>(0.0, 1.0);
    m_distribution_normal  = std::normal_distribution<double>(0.0, 1.0);
}

SimulationADMC::SimulationADMC(const ParametersADMC& parameters, const Device1D& myDevice, double voltage)
    : m_parameters(parameters),
      m_device(myDevice),
      m_x_line(myDevice.get_doping_profile().get_x_line()),
      m_doping(myDevice.get_doping_profile().get_doping_concentration()),
      m_ElectricField(m_device.get_poisson_solution_at_voltage(voltage).m_electric_field),
      m_eVelocity(m_x_line.size()),
      m_hVelocity(m_x_line.size()) {
    m_particles.reserve(m_parameters.m_max_particles);

    if (m_ElectricField.size() != m_x_line.size()) {
        throw std::runtime_error("The size of the electric field vector is not the same as the size of the x_line vector.");
    }

    std::random_device rd;
    m_generator.seed(rd());
    m_distribution_uniform = std::uniform_real_distribution<double>(0.0, 1.0);
    m_distribution_normal  = std::normal_distribution<double>(0.0, 1.0);
}

void SimulationADMC::AddElectrons(std::size_t number_of_electrons) {
    double x_length = m_x_line.back() - m_x_line.front();
    for (std::size_t i = 0; i < number_of_electrons; ++i) {
        double      x = m_x_line.front() + x_length * m_distribution_uniform(m_generator);
        double      y = m_parameters.m_y_width * m_distribution_uniform(m_generator);
        double      z = m_parameters.m_z_width * m_distribution_uniform(m_generator);
        Vector3     position(x, y, z);
        std::size_t new_index    = m_particles.size();
        Particle    new_electron = Particle(new_index, ParticleType::electron, position);
        double      rpl_number   = m_distribution_uniform(m_generator);
        new_electron.set_rpl_number(rpl_number);
        m_particles.push_back(new_electron);
    }
}

void SimulationADMC::AddElectrons(std::size_t number_of_electrons, const Vector3& position) {
    for (std::size_t i = 0; i < number_of_electrons; ++i) {
        std::size_t new_index    = m_particles.size();
        Particle    new_electron = Particle(new_index, ParticleType::electron, position);
        double      rpl_number   = m_distribution_uniform(m_generator);
        new_electron.set_rpl_number(rpl_number);
        m_particles.push_back(new_electron);
    }
}

void SimulationADMC::AddHoles(std::size_t number_of_holes) {
    double x_length = m_x_line.back() - m_x_line.front();
    for (std::size_t i = 0; i < number_of_holes; ++i) {
        double      x = m_x_line.front() + x_length * m_distribution_uniform(m_generator);
        double      y = m_parameters.m_y_width * m_distribution_uniform(m_generator);
        double      z = m_parameters.m_z_width * m_distribution_uniform(m_generator);
        Vector3     position(x, y, z);
        std::size_t new_index  = m_particles.size();
        Particle    new_hole   = Particle(new_index, ParticleType::electron, position);
        double      rpl_number = m_distribution_uniform(m_generator);
        new_hole.set_rpl_number(rpl_number);
        m_particles.push_back(new_hole);
    }
}

void SimulationADMC::AddHoles(std::size_t number_of_holes, const Vector3& position) {
    for (std::size_t i = 0; i < number_of_holes; ++i) {
        std::size_t new_index  = m_particles.size();
        Particle    new_hole   = Particle(new_index, ParticleType::hole, position);
        double      rpl_number = m_distribution_uniform(m_generator);
        new_hole.set_rpl_number(rpl_number);
        m_particles.push_back(new_hole);
    }
}

// void SimulationADMC::SetDataFromDeviceStep() {
//     // Set doping and electric field
//     for (std::size_t idx_part = 0; idx_part < m_particles.size(); ++idx_part) {
//         std::size_t idx_x = 0;
//         while (m_particles[idx_part].position().x() > m_x_line[idx_x]) {
//             ++idx_x;
//         }
//         m_particles[idx_part].set_doping(m_doping[idx_x]);
//         m_particles[idx_part].set_electric_field({m_ElectricField[idx_x], 0.0, 0.0});
//         m_particles[idx_part].compute_mobility();
//         m_particles[idx_part].compute_velocity();
//     }
// }

void SimulationADMC::SetBulkData(double doping_level, double electric_field) {
    // Set doping and electric field
    for (std::size_t idx_part = 0; idx_part < m_particles.size(); ++idx_part) {
        m_particles[idx_part].set_doping(doping_level);
        m_particles[idx_part].set_electric_field({electric_field, 0.0, 0.0});
        m_particles[idx_part].compute_mobility();
        m_particles[idx_part].compute_velocity();
    }
}

void SimulationADMC::SetDataFromDeviceStep() {
    // Set doping and electric field
    std::vector<double> x_positions               = this->get_all_x_positions();
    std::vector<double> InterpolatedDoping        = Utils::interp1dSorted(m_x_line, m_doping, x_positions);
    std::vector<double> InterpolatedElectricField = Utils::interp1dSorted(m_x_line, m_ElectricField, x_positions);
    for (std::size_t idx_part = 0; idx_part < m_particles.size(); ++idx_part) {
        m_particles[idx_part].set_doping(InterpolatedDoping[idx_part]);
        m_particles[idx_part].set_electric_field({InterpolatedElectricField[idx_part], 0.0, 0.0});
        m_particles[idx_part].compute_mobility();
        m_particles[idx_part].compute_velocity();
    }
}

void SimulationADMC::PerformDriftDiffusionStep() {
    for (std::size_t idx_part = 0; idx_part < m_particles.size(); ++idx_part) {
        const Vector3 RandomGaussianVector(m_distribution_normal(m_generator),
                                           m_distribution_normal(m_generator),
                                           m_distribution_normal(m_generator));
        m_particles[idx_part].perform_transport_step(m_parameters.m_time_step, RandomGaussianVector);
    }
}

void SimulationADMC::PerformImpactIonizationStep() {
    std::size_t nb_particle = m_particles.size();
    for (std::size_t idx_part = 0; idx_part < nb_particle; ++idx_part) {
        m_particles[idx_part].perform_impact_ionization_step(m_parameters.m_time_step);
        if (m_particles[idx_part].has_impact_ionized()) {
            m_history.m_all_impact_ionization_positions.push_back(m_particles[idx_part].position());
            std::size_t index_new_particles   = m_particles.size();
            double      new_r_parent_particle = m_distribution_uniform(m_generator);
            m_particles[idx_part].set_rpl_number(new_r_parent_particle);
            m_particles[idx_part].set_cumulative_impact_ionization(0.0);
            m_particles[idx_part].add_impact_ionization_to_history();
            if (m_parameters.m_activate_particle_creation) {
                Particle new_particle = Particle(index_new_particles, ParticleType::electron, m_particles[idx_part].position());
                double   new_rpl      = m_distribution_uniform(m_generator);
                new_particle.set_rpl_number(new_rpl);
                m_particles.push_back(new_particle);
                Particle new_particle2 = Particle(index_new_particles + 1, ParticleType::hole, m_particles[idx_part].position());
                double   new_rpl2      = m_distribution_uniform(m_generator);
                new_particle2.set_rpl_number(new_rpl2);
                m_particles.push_back(new_particle2);
            }
        }
    }
}

void SimulationADMC::CheckContactCrossing() {
    for (std::size_t idx_part = 0; idx_part < m_particles.size(); ++idx_part) {
        if (m_particles[idx_part].position().x() < 0 || m_particles[idx_part].position().x() > m_x_line.back()) {
            m_particles[idx_part].set_crossed_contact(true);
        }
    }
    // Remove particles that have crossed the contact
    m_particles.erase(std::remove_if(m_particles.begin(), m_particles.end(), [](const Particle& p) { return p.crossed_contact(); }),
                      m_particles.end());
}

void SimulationADMC::RunSimulation() {
    m_time = 0.0;
    while (m_time < m_parameters.m_max_time && m_particles.size() > 0 && m_particles.size() < m_parameters.m_max_particles) {
        SetDataFromDeviceStep();
        PerformDriftDiffusionStep();
        PerformImpactIonizationStep();
        m_time += m_parameters.m_time_step;
        m_number_steps++;
        // ExportCurrentState();
    }
    if (m_particles.size() == 0) {
        std::cout << "No more particles, simulation stopped" << std::endl;
    } else if (m_particles.size() >= m_parameters.m_max_particles || m_particles.size() >= m_parameters.m_avalanche_threshold) {
        m_history.m_has_reached_avalanche = true;
        m_history.m_avalanche_time        = m_time;
        // std::cout << "Avalanche!" << std::endl;
    } else {
        // std::cout << "Maximum time reached, simulation stopped" << std::endl;
    }
}

std::vector<double> SimulationADMC::RunTransportSimulationToMaxField() {
    auto                it_max_field = std::max_element(m_ElectricField.begin(), m_ElectricField.end());
    double              x_max_field  = m_x_line[std::distance(m_ElectricField.begin(), it_max_field)];
    std::vector<double> times_to_max_field;
    const double tol_x = 5.0e-3;
    while (m_time < m_parameters.m_max_time && m_particles.size() > 0) {
        SetDataFromDeviceStep();
        PerformDriftDiffusionStep();
        m_time += m_parameters.m_time_step;
        m_number_steps++;
        std::vector<double>      x_positions = get_all_x_positions();
        std::vector<std::size_t> idx_part_to_remove;
        for (std::size_t idx_part = 0; idx_part < m_particles.size(); ++idx_part) {
            if (fabs(x_positions[idx_part] - x_max_field) < tol_x) {
                times_to_max_field.push_back(m_time);
                idx_part_to_remove.push_back(idx_part);
            }
        }
        std::size_t nb_particle = m_particles.size();
        for (std::size_t idx_part = 0; idx_part < nb_particle; ++idx_part) {
            if (std::find(idx_part_to_remove.begin(), idx_part_to_remove.end(), idx_part) != idx_part_to_remove.end()) {
                m_particles.erase(m_particles.begin() + idx_part);
            }
        }
    }
    return times_to_max_field;
}

void SimulationADMC::RunBULKSimulation(double doping_level, double electric_field) {
    m_time = 0.0;
    ExportCurrentState();
    std::vector<double> list_std_x;
    std::vector<double> times;
    list_std_x.push_back(0.0);
    times.push_back(0.0);
    while (m_time < m_parameters.m_max_time && m_particles.size() > 0 && m_particles.size() < m_parameters.m_max_particles) {
        SetBulkData(doping_level, electric_field);
        PerformDriftDiffusionStep();
        PerformImpactIonizationStep();
        m_time += m_parameters.m_time_step;
        m_number_steps++;

        double std_x = utils::standard_deviation(get_all_x_positions());
        list_std_x.push_back(std_x);
        times.push_back(m_time);
    }
    if (m_particles.size() == 0) {
        std::cout << "No more particles, simulation stopped" << std::endl;
    } else if (m_particles.size() >= m_parameters.m_max_particles || m_particles.size() >= m_parameters.m_avalanche_threshold) {
        m_history.m_has_reached_avalanche = true;
        m_history.m_avalanche_time        = m_time;
        // std::cout << "Avalanche!" << std::endl;
    } else {
        // std::cout << "Maximum time reached, simulation stopped" << std::endl;
    }
    std::ofstream my_file("std_xVStime.csv");
    my_file << "time,std_x" << std::endl;
    for (std::size_t i = 0; i < list_std_x.size(); ++i) {
        my_file << times[i] << "," << list_std_x[i] << std::endl;
    }
    my_file.close();
}

std::vector<double> SimulationADMC::compute_impact_ionization_coeff() const {
    std::vector<double>      x_positions = get_all_x_positions();
    std::vector<std::size_t> nb_impact_ionization_per_particle(x_positions.size(), 0);
    for (std::size_t idx_part = 0; idx_part < m_particles.size(); ++idx_part) {
        std::size_t nb_ii                           = m_particles[idx_part].get_history().m_number_of_impact_ionizations;
        nb_impact_ionization_per_particle[idx_part] = nb_ii;
    }
    std::vector<double> impact_ionization_rates(x_positions.size(), 0.0);
    for (std::size_t idx_part = 0; idx_part < m_particles.size(); ++idx_part) {
        double coeff                      = nb_impact_ionization_per_particle[idx_part] / fabs(x_positions[idx_part]);
        impact_ionization_rates[idx_part] = coeff;
    }
    return impact_ionization_rates;
}

void SimulationADMC::ExportCurrentState() const {
    std::string   filename = fmt::format("{}State.csv.{:05d}", m_parameters.m_output_file, m_number_steps);
    std::ofstream file(filename);
    // Export for each particle, its position, its velocity, and its type
    file << "X,Y,Z,Vx,Vy,Vz,Type,CumulativeIonizationCoeff\n";
    for (const auto& particle : m_particles) {
        fmt::print(file,
                   "{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:d},{:.3e} \n",
                   particle.position().x(),
                   particle.position().y(),
                   particle.position().z(),
                   particle.velocity().x(),
                   particle.velocity().y(),
                   particle.velocity().z(),
                   static_cast<int>(particle.type()),
                   particle.cumulative_impact_ionization());
    }
    file.close();
}

/**
 * @brief Run simulations on every x position of the device and gather the results.
 * Electron only simulation.
 *
 * @param parameters
 * @param device
 * @param nb_simulation_per_points
 */
void MainFullADMCSimulation(const ParametersADMC& parameters,
                            const Device1D&       device,
                            double                voltage,
                            std::size_t           nb_simulation_per_points,
                            std::size_t           nbPointsX,
                            const std::string&    export_name) {
    double x_max = device.get_doping_profile().get_x_line().back();

    std::vector<double> x_line = utils::linspace(0.0, x_max, nbPointsX);

    std::vector<double> all_avalanche_times;
    std::vector<double> eBreakdownRatio(x_line.size(), 0.0);

    std::size_t nb_avalanches = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : nb_avalanches)
    for (std::size_t idx_x = 0; idx_x < x_line.size(); ++idx_x) {
        // std::cout << "Running simulation on point " << x_line[idx_x] << std::endl;
        std::vector<double> avalanche_times_point;
        std::vector<double> eBreakdownRatio_point;
        int                 nb_avalanches_point = 0;
        for (std::size_t idx_sim = 0; idx_sim < nb_simulation_per_points; ++idx_sim) {
            SimulationADMC simulation(parameters, device, voltage);
            simulation.AddElectrons(1, {x_line[idx_x], 0.5, 0.5});
            simulation.RunSimulation();
            if (simulation.get_history().m_has_reached_avalanche) {
                avalanche_times_point.push_back(simulation.get_history().m_avalanche_time);
                nb_avalanches_point++;
                nb_avalanches++;
            }
        }
        double ratio           = static_cast<double>(nb_avalanches_point) / static_cast<double>(nb_simulation_per_points);
        eBreakdownRatio[idx_x] = ratio;
        // Add the avalanche times to the global vector
#pragma omp critical
        { all_avalanche_times.insert(all_avalanche_times.end(), avalanche_times_point.begin(), avalanche_times_point.end()); }
    }
    double ratio = static_cast<double>(nb_avalanches) / static_cast<double>(nb_simulation_per_points * x_line.size());
    std::cout << "Avalanche Breakdown ratio: " << ratio << std::endl;
    // Export the avalanche times to a file
    std::ofstream file_avalanche_times(export_name + "AvalancheTimes.csv");
    file_avalanche_times << "TimeAvalanche" << std::endl;
    for (const auto& time : all_avalanche_times) {
        file_avalanche_times << time << std::endl;
    }
    file_avalanche_times.close();

    // Export the breakdown ratio to a file
    std::ofstream file_breakdown_ratio(export_name + "BreakdownProbability.csv");
    file_breakdown_ratio << "X,BreakdownRatio" << std::endl;
    for (std::size_t idx_x = 0; idx_x < x_line.size(); ++idx_x) {
        file_breakdown_ratio << x_line[idx_x] << "," << eBreakdownRatio[idx_x] << std::endl;
    }
    file_breakdown_ratio.close();
}

/**
 * @brief Run simulations on every x position of the device and gather the results.
 * Electron only simulation.
 *
 * @param parameters
 * @param device
 * @param nb_simulation_per_points
 */
void MainFullADMCSimulationToMaxField(const ParametersADMC& parameters,
                                      const Device1D&       device,
                                      double                voltage,
                                      std::size_t           nb_simulation_per_points,
                                      std::size_t           nbPointsX,
                                      const std::string&    export_name) {

    std::filesystem::create_directory("TEST_FIELD");

    double              x_max  = device.get_doping_profile().get_x_line().back();
    std::vector<double> x_line = utils::linspace(0.0, x_max, nbPointsX);
    std::vector<double> all_transport_times;

#pragma omp parallel for schedule(dynamic)
    for (std::size_t idx_x = 0; idx_x < x_line.size(); ++idx_x) {
        // std::cout << "Running simulation on point " << x_line[idx_x] << std::endl;
        SimulationADMC      simulation(parameters, device, voltage);
        simulation.AddElectrons(nb_simulation_per_points, {x_line[idx_x], 0.5, 0.5});
        std::vector<double> time_to_max_field = simulation.RunTransportSimulationToMaxField();
        // Add the avalanche times to the global vector
#pragma omp critical
        { all_transport_times.insert(all_transport_times.end(), time_to_max_field.begin(), time_to_max_field.end()); }
        std::ofstream file_transport_times("TEST_FIELD/TimeToMaxField_" + std::to_string(idx_x) + "_.csv");
        file_transport_times << "TimeToMaxField" << std::endl;
        for (const auto& time : time_to_max_field) {
            file_transport_times << time << std::endl;
        }
        file_transport_times.close();
    }

    // Export the avalanche times to a file
    std::cout << "Exporting the time to max field" << std::endl;
    std::ofstream file_avalanche_times(export_name + "TimeToMaxField.csv");
    file_avalanche_times << "TimeToMaxField" << std::endl;
    for (const auto& time : all_transport_times) {
        file_avalanche_times << time << std::endl;
    }
    file_avalanche_times.close();
}

// void ExportAllParticlesHistory() const;

}  // namespace ADMC