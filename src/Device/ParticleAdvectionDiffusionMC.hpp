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
#include <vector>

#include "ImpactIonization.hpp"
#include "Mobility.hpp"
#include "Vector3.hpp"
#include "physical_constants.hpp"

namespace ADMC {

enum class ParticleType { electron, hole };

struct ParticleHistory {
    std::size_t          m_index;
    ParticleType         m_type;
    std::vector<Vector3> m_positions;
    std::vector<Vector3> m_velocities;
    std::vector<double>  m_times;
    std::vector<double>  m_cumulative_impact_ionizations;

    ParticleHistory(std::size_t index, ParticleType type) : m_index(index), m_type(type) {}
};

class Particle {
 protected:
    std::size_t  m_index;
    Vector3      m_position;
    double       m_mobility       = 0.0;
    Vector3      m_drift_velocity = Vector3(0.0, 0.0, 0.0);
    ParticleType m_type;

    Vector3 m_electric_field               = Vector3(0.0, 0.0, 0.0);
    double  m_doping                       = 0.0;
    double  m_time                         = 0.0;
    double  m_cumulative_impact_ionization = 0.0;
    double  m_rpl_number                   = 0.0;

    bool m_crossed_contact = false;

    ParticleHistory m_history;

 public:
    Particle(std::size_t index, ParticleType type) : m_index(index), m_type(type), m_history(index, type) {}
    Particle(std::size_t index, ParticleType type, const Vector3& position)
        : m_index(index),
          m_position(position),
          m_type(type),
          m_history(index, type) {}

    void set_position(const Vector3& position) { m_position = position; }
    void set_mobility(double mobility) { m_mobility = mobility; }
    void set_velocity(const Vector3& velocity) { m_drift_velocity = velocity; }
    void set_electric_field(const Vector3& electric_field) { m_electric_field = electric_field; }
    void set_doping(double doping) { m_doping = doping; }
    void set_time(double time) { m_time = time; }
    void set_cumulative_impact_ionization(double cumulative_impact_ionization) {
        m_cumulative_impact_ionization = cumulative_impact_ionization;
    }
    void set_rpl_number(double rpl_number) { m_rpl_number = rpl_number; }
    void set_crossed_contact(bool crossed_contact) { m_crossed_contact = crossed_contact; }

    std::size_t    index() const { return m_index; }
    const Vector3& position() const { return m_position; }
    double         mobility() const { return m_mobility; }
    const Vector3& velocity() const { return m_drift_velocity; }
    ParticleType   type() const { return m_type; }
    const Vector3& electric_field() const { return m_electric_field; }
    double         doping() const { return m_doping; }
    double         time() const { return m_time; }
    double         cumulative_impact_ionization() const { return m_cumulative_impact_ionization; }
    double         rpl_number() const { return m_rpl_number; }
    bool           crossed_contact() const { return m_crossed_contact; }
    double         get_probability_impact_ionization() const { return 1 - exp(-m_cumulative_impact_ionization); }
    double         charge_sign() const { return (m_type == ParticleType::electron) ? -1 : 1; }

    void compute_mobility(double temperature = 300.0) {
        if (m_type == ParticleType::electron) {
            m_mobility = physic::model::electron_mobility_arora_canali(m_doping, m_electric_field.norm(), temperature);
        } else {
            m_mobility = physic::model::hole_mobility_arora_canali(m_doping, m_electric_field.norm(), temperature);
        }
    }

    void compute_velocity() { m_drift_velocity = charge_sign() * m_mobility * m_electric_field; }

    void perform_transport_step(double time_step, Vector3 GaussianReducedCenter, double temperature = 300) {
        constexpr double cm_to_micron = 1e4;
        const double     particle_diffusion =
            m_mobility * physic::constant::boltzmann_constant * temperature / physic::constant::elementary_charge;
        const double particle_diffusion_sqrt = sqrt(2.0 * particle_diffusion);
        m_position +=
            m_drift_velocity * time_step * cm_to_micron + GaussianReducedCenter * particle_diffusion_sqrt * sqrt(time_step) * cm_to_micron;
        m_time += time_step;
    }

    void perform_impact_ionization_step(double time_step) {
        constexpr double cm_to_micron = 1e-4;
        double m_local_ionization_coeff = 0.0;
        if (m_type == ParticleType::electron) {
            m_local_ionization_coeff = mcintyre::alpha_DeMan(m_electric_field.norm() * cm_to_micron, 1.0, 1.12052);
        } else {
            m_local_ionization_coeff = mcintyre::beta_DeMan(m_electric_field.norm() * cm_to_micron, 1.0, 1.12052);
        }
        m_cumulative_impact_ionization += time_step * m_drift_velocity.norm() * m_local_ionization_coeff * 1.0/cm_to_micron;
    }

    bool has_impact_ionized() const {
        const double integral_path_length = 1 - exp(-m_cumulative_impact_ionization);
        return integral_path_length >= m_rpl_number;
    }

    void add_current_state_to_history() {
        m_history.m_positions.push_back(m_position);
        m_history.m_velocities.push_back(m_drift_velocity);
        m_history.m_times.push_back(m_time);
        m_history.m_cumulative_impact_ionizations.push_back(m_cumulative_impact_ionization);
    }
};

}  // namespace ADMC
