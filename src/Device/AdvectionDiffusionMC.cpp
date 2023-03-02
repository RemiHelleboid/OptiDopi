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
#include "Vector3.hpp"

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
        m_particles.push_back(new_electron);
    }
}

void SimulationADMC::AddElectrons(std::size_t number_of_electrons, const Vector3& position) {
    for (std::size_t i = 0; i < number_of_electrons; ++i) {
        std::size_t new_index    = m_particles.size();
        Particle    new_electron = Particle(new_index, ParticleType::electron, position);
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
        std::size_t new_index = m_particles.size();
        Particle    new_hole  = Particle(new_index, ParticleType::electron, position);
        m_particles.push_back(new_hole);
    }
}

void SimulationADMC::AddHoles(std::size_t number_of_holes, const Vector3& position) {
    for (std::size_t i = 0; i < number_of_holes; ++i) {
        std::size_t new_index = m_particles.size();
        Particle    new_hole  = Particle(new_index, ParticleType::hole, position);
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

void SimulationADMC::RunSimulation() {

    m_time = 0.0;
    ExportCurrentState();
    while (m_time < m_parameters.m_max_time && m_particles.size() > 0 && m_particles.size() < m_parameters.m_max_particles) {
        SetDataFromDeviceStep();
        PerformDriftDiffusionStep();
        m_time += m_parameters.m_time_step;
        m_number_steps++;
        ExportCurrentState();
    }
}

void SimulationADMC::ExportCurrentState() const {
    std::string filename = fmt::format("{}State.csv.{:05d}", m_parameters.m_output_file, m_number_steps);
    std::ofstream file(filename);
    // Export for each particle, its position, its velocity, and its type
    file << "X,Y,Z,Vx,Vy,Vz,Type" << std::endl;
    for (const auto& particle : m_particles) {
        fmt::print(file, "{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:d}\n",
                   particle.position().x(), particle.position().y(), particle.position().z(),
                   particle.velocity().x(), particle.velocity().y(), particle.velocity().z(),
                   static_cast<int>(particle.type()));
    }
    file.close();
}



    // void ExportAllParticlesHistory() const;

}  // namespace ADMC