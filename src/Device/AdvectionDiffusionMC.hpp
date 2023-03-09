/**
 * @file AdvectionDiffusionMC.hpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-03-01
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "ImpactIonization.hpp"
#include "Mobility.hpp"
#include "ParticleAdvectionDiffusionMC.hpp"
#include "Vector3.hpp"

class Device1D;

namespace ADMC {

struct ADMCSimulationHistory {
    std::vector<double>  m_time;
    std::vector<double>  m_number_of_particles;
    std::vector<double>  m_number_of_electrons;
    std::vector<double>  m_number_of_holes;
    std::vector<double>  m_number_impact_ionizations;
    std::vector<Vector3> m_all_impact_ionization_positions;

    bool   m_has_reached_avalanche = false;
    double m_avalanche_time        = 0.0;
};

struct ParametersADMC {
    double m_time_step;
    double m_max_time;
    double m_temperature;

    // Psuedo 3D dimensions
    double m_y_width = 1.0;
    double m_z_width = 1.0;

    double m_x_anode;
    double m_x_cathode;

    std::size_t m_avalanche_threshold;
    std::size_t m_max_particles;

    bool m_activate_impact_ionization;
    bool m_activate_particle_creation;

    std::string m_output_file;

    bool keep_all_impact_ionization_positions = false;
};

class SimulationADMC {
 private:
    ParametersADMC        m_parameters;
    std::vector<Particle> m_particles;
    double                m_time         = 0.0;
    std::size_t           m_number_steps = 0;

    const Device1D&     m_device;
    std::vector<double> m_x_line;
    std::vector<double> m_doping;
    std::vector<double> m_ElectricField;
    std::vector<double> m_eVelocity;
    std::vector<double> m_hVelocity;

    std::mt19937                           m_generator;
    std::uniform_real_distribution<double> m_distribution_uniform;
    std::normal_distribution<double>       m_distribution_normal;

    ADMCSimulationHistory m_history;

 public:
    SimulationADMC(const ParametersADMC& parameters, const Device1D& device, double voltage);

    const ADMCSimulationHistory& get_history() const { return m_history; }
    double                       get_time() const { return m_history.m_time.back(); }

    ParametersADMC& parameters() { return m_parameters; }

    void ClearParticles() { m_particles.clear(); }

    void set_electric_field(double voltage);
    void set_electric_field(const std::vector<double>& electric_field) { m_ElectricField = electric_field; }

    /**
     * @brief Add electrons at random positions.
     *
     * @param number_of_electrons
     */
    void AddElectrons(std::size_t number_of_electrons);

    /**
     * @brief Add electrons at a given position
     *
     * @param number_of_electrons
     * @param position
     */
    void AddElectrons(std::size_t number_of_electrons, const Vector3& position);

    /**
     * @brief Add holes at random positions.
     *
     * @param number_of_holes
     */
    void AddHoles(std::size_t number_of_holes);

    /**
     * @brief Add holes at a given position.
     *
     * @param number_of_holes
     * @param position
     */
    void AddHoles(std::size_t number_of_holes, const Vector3& position);

    void SetDataFromDeviceStep();

    void PerformDriftDiffusionStep();

    void PerformImpactIonizationStep();

    void CheckContactCrossing();

    void FillSimulationHistory();

    void RunSimulation();

    bool has_reached_avalanche_threshold() const { return m_particles.size() > m_parameters.m_avalanche_threshold; }

    std::size_t get_number_of_particles() const { return m_particles.size(); }
    std::size_t get_number_of_electrons() const {
        return std::count_if(m_particles.begin(), m_particles.end(), [](const Particle& particle) {
            return particle.type() == ParticleType::electron;
        });
    }
    std::size_t get_number_of_holes() const {
        return std::count_if(m_particles.begin(), m_particles.end(), [](const Particle& particle) {
            return particle.type() == ParticleType::hole;
        });
    }

    std::vector<double> get_all_x_positions() const {
        std::vector<double> positions;
        positions.reserve(m_particles.size());
        for (const auto& particle : m_particles) {
            positions.push_back(particle.position().x());
        }
        return positions;
    }

    void ExportCurrentState() const;
    void ExportSimulationHistory() const;
    void ExportAllParticlesHistory() const;
};

void MainFullADMCSimulation(const ParametersADMC& parameters,
                            const Device1D&       device,
                            double                voltage,
                            std::size_t           nb_simulation_per_points,
                            std::size_t           nbPointsX,
                            const std::string&    export_name);

}  // namespace ADMC