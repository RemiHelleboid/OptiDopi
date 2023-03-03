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

#include "Mobility.hpp"
#include "Mobility.hpp"
#include "fill_vector.hpp"

namespace ADMC {

SimulationADMC::SimulationADMC(const ParametersADMC& parameters, const Device1D& myDevice)
    : m_parameters(parameters),
      m_device(myDevice),
      m_x_line(myDevice.get_doping_profile().get_x_line()),
      m_doping(myDevice.get_doping_profile().get_doping_concentration()),
      m_ElectricField(m_x_line.size()),
      m_eVelocity(m_x_line.size()),
      m_hVelocity(m_x_line.size()) {
    m_particles.reserve(m_parameters.m_max_particles);

    std::random_device rd;
    m_generator.seed(rd());
    m_distribution_uniform = std::uniform_real_distribution<double>(0.0, 1.0);
    m_distribution_normal  = std::normal_distribution<double>(0.0, 1.0);
}

void SimulationADMC::set_electric_field(double voltage) {
    m_ElectricField = m_device.get_poisson_solution_at_voltage(voltage).m_electric_field;
    if (m_ElectricField.size() != m_x_line.size()) {
        throw std::runtime_error("The size of the electric field vector is not the same as the size of the x_line vector.");
    }
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

void SimulationADMC::SetDataFromDeviceStep() {
    // Set doping and electric field
    for (std::size_t idx_part = 0; idx_part < m_particles.size(); ++idx_part) {
        std::size_t idx_x = 0;
        while (m_particles[idx_part].position().x() > m_x_line[idx_x]) {
            ++idx_x;
        }
        m_particles[idx_part].set_doping(m_doping[idx_x]);
        m_particles[idx_part].set_electric_field({m_ElectricField[idx_x], 0.0, 0.0});
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
    std::size_t nb_particle                    = m_particles.size();
    bool        at_least_one_impact_ionization = false;
    for (std::size_t idx_part = 0; idx_part < nb_particle; ++idx_part) {
        m_particles[idx_part].perform_impact_ionization_step(m_parameters.m_time_step);
        if (m_particles[idx_part].has_impact_ionized()) {
            at_least_one_impact_ionization = true;
            m_history.m_all_impact_ionization_positions.push_back(m_particles[idx_part].position());
            std::size_t index_new_particles   = m_particles.size();
            double      new_r_parent_particle = m_distribution_uniform(m_generator);
            m_particles[idx_part].set_rpl_number(new_r_parent_particle);
            m_particles[idx_part].set_cumulative_impact_ionization(0.0);
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
        if (m_particles[idx_part].position().x() < 0 || m_particles[idx_part].position().x() > m_device.get_doping_profile().get_x_line().back()) {
            m_particles[idx_part].set_crossed_contact(true);
        }
    }
    // Remove particles that have crossed the contact
    m_particles.erase(std::remove_if(m_particles.begin(), m_particles.end(), [](const Particle& p) { return p.crossed_contact();}), m_particles.end());
}


void SimulationADMC::RunSimulation() {
    m_time = 0.0;
    ExportCurrentState();
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
                            std::size_t           nb_simulation_per_points) {
    std::size_t         nb_points_x = device.get_doping_profile().get_x_line().size();
    double x_max = device.get_doping_profile().get_x_line().back();

    std::size_t NXpoints = 250;
    std::vector<double> x_line = utils::linspace(0.0, x_max, NXpoints);

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
            SimulationADMC simulation(parameters, device);
            simulation.AddElectrons(1, {x_line[idx_x], 0.5, 0.5});
            simulation.set_electric_field(voltage);
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
        {
            all_avalanche_times.insert(all_avalanche_times.end(), avalanche_times_point.begin(), avalanche_times_point.end());
        }
    }
    double ratio = static_cast<double>(nb_avalanches) / static_cast<double>(nb_simulation_per_points * x_line.size());
    std::cout << "Avalanche Breakdown ratio: " << ratio << std::endl;
    // Export the avalanche times to a file
    std::ofstream file_avalanche_times(parameters.m_output_file + "AvalancheTimes.csv");
    file_avalanche_times << "TimeAvalanche" << std::endl;
    for (const auto& time : all_avalanche_times) {
        file_avalanche_times << time << std::endl;
    }
    file_avalanche_times.close();
}

// void ExportAllParticlesHistory() const;

}  // namespace ADMC