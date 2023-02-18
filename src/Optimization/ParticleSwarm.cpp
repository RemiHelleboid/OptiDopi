/**
 * @file ParticleSwarm.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-02-16
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "ParticleSwarm.hpp"

#include <fstream>
#include <iostream>
#include <filesystem>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include "omp.h"


namespace Optimization {

ParticleSwarm::ParticleSwarm(std::size_t                                       number_particles,
                             std::size_t                                       number_dimensions,
                             std::function<double(const std::vector<double>&)> fitness_function)
    : m_number_particles(number_particles),
      m_number_dimensions(number_dimensions),
      m_fitness_function(fitness_function) {
    m_particles.resize(m_number_particles);
    for (std::size_t i = 0; i < m_number_particles; ++i) {
        m_particles[i].set_dimensions(m_number_dimensions);
        m_particles[i].seed_random_engine();
    }

    m_best_position.resize(m_number_dimensions);
    this->set_up_export();
    m_random_engine.seed(m_random_device());
}

void ParticleSwarm::set_bounds(const std::vector<double>& bounds_min, const std::vector<double>& bounds_max) {
    m_bounds_min = bounds_min;
    m_bounds_max = bounds_max;
}

void ParticleSwarm::set_bounds(std::vector<std::pair<double, double>> bounds) {
    m_bounds_min.resize(m_number_dimensions);
    m_bounds_max.resize(m_number_dimensions);
    for (std::size_t i = 0; i < m_number_dimensions; ++i) {
        m_bounds_min[i] = bounds[i].first;
        m_bounds_max[i] = bounds[i].second;
    }
}

void ParticleSwarm::initialize_particles() {
    std::cout << "Initializing particles..." << std::endl;
#pragma omp parallel for 
    for (std::size_t idx_particle = 0; idx_particle < m_number_particles; ++idx_particle) {
        for (std::size_t i = 0; i < m_number_dimensions; ++i) {
            double range = m_bounds_max[i] - m_bounds_min[i];
            m_particles[idx_particle].position[i] = m_bounds_min[i] + range * m_uniform_distribution(m_particles[idx_particle].get_random_engine());
            // Velocity is initialized between -abs(max - min) and abs(max - min)
            m_particles[idx_particle].velocity[i] = -range + 2.0 * range * m_uniform_distribution(m_particles[idx_particle].get_random_engine());
        }
        m_particles[idx_particle].best_position = m_particles[idx_particle].position;
        m_particles[idx_particle].best_fitness  = m_fitness_function(m_particles[idx_particle].position);
    }

    m_best_position = m_particles[0].position;
    m_best_fitness  = m_particles[0].best_fitness;
    for (std::size_t i = 1; i < m_number_particles; ++i) {
        if (m_particles[i].best_fitness < m_best_fitness) {
            m_best_position = m_particles[i].position;
            m_best_fitness  = m_particles[i].best_fitness;
        }
    }
    this->export_current_state();
}

void ParticleSwarm::clip_particles() {
    for (auto& particle : m_particles) {
        for (std::size_t i = 0; i < m_number_dimensions; ++i) {
            if (particle.position[i] < m_bounds_min[i]) {
                particle.position[i] = m_bounds_min[i];
            } else if (particle.position[i] > m_bounds_max[i]) {
                particle.position[i] = m_bounds_max[i];
            }
        }
    }
}

void ParticleSwarm::update_particles() {
#pragma omp parallel for 
    for (auto& particle : m_particles) {
        for (std::size_t i = 0; i < m_number_dimensions; ++i) {
            double r1 = m_uniform_distribution(m_random_engine);
            double r2 = m_uniform_distribution(m_random_engine);
            particle.velocity[i] = m_inertia_weight * particle.velocity[i] + m_cognitive_weight * r1 * (particle.best_position[i] - particle.position[i]) +
                                   m_social_weight * r2 * (m_best_position[i] - particle.position[i]);
            particle.position[i] += particle.velocity[i] * m_velocity_scaling;
        }
        double fitness = m_fitness_function(particle.position);
        if (fitness < particle.best_fitness) {
            particle.best_position = particle.position;
            particle.best_fitness  = fitness;
        }
    }
    for (std::size_t i = 0; i < m_number_particles; ++i) {
        if (m_particles[i].best_fitness < m_best_fitness) {
            m_best_position = m_particles[i].position;
            m_best_fitness  = m_particles[i].best_fitness;
        }
    }
}

void ParticleSwarm::optimize(std::size_t number_iterations) {
    initialize_particles();
    for (std::size_t i = 0; i < number_iterations; ++i) {
        std::cout << fmt::format("Iteration {:d}/{:d}", i, number_iterations) << std::endl;
        m_current_iteration = i;
        update_particles();
        clip_particles();
        this->export_current_state();
        std::cout << fmt::format("\rIteration {:d}/{:d} -> Best fitness: {:.3f}", i, number_iterations, m_best_fitness) << std::flush;
    }
}

void ParticleSwarm::set_up_export() {
    std::cout << "Setting up export..." << std::endl;
    // Create directory for output
    if (std::filesystem::exists(m_dir_export)) {
        std::filesystem::remove_all(m_dir_export);
    }
    std::filesystem::create_directories(m_dir_export);
    
    int dimension = m_number_dimensions;
    // Create one file for each particle to store its history
    for (std::size_t idx_particle = 0; idx_particle < m_number_particles; ++idx_particle) {
        std::ofstream file;
        file.open(fmt::format("{}/particle_{:04d}.csv", m_dir_export, idx_particle));
        // Write header (iteration, position, velocity, best_position, best_fitness)
        file << "iteration";
        for (int i = 0; i < dimension; ++i) {
            file << fmt::format(",x_{:02d}", i);
        }
        for (int i = 0; i < dimension; ++i) {
            file << fmt::format(",v_{:02d}", i);
        }
        for (int i = 0; i < dimension; ++i) {
            file << fmt::format(",best_x_{:02d}", i);
        }
        file << ",best_fitness" << std::endl;
        file.close();
    }
    // Create one file for the swarm to store its history
    std::ofstream file(m_dir_export + "/global_swarm.csv");
    // Write header (iteration, best_position, best_fitness)
    file << "iteration";
    for (int i = 0; i < dimension; ++i) {
        file << fmt::format(",best_x_{:02d}", i);
    }
    file << ",best_fitness" << std::endl;
    file.close();
}

void ParticleSwarm::export_current_state() {
    int dimension = m_bounds_min.size();
    // Export particle history
    for (std::size_t idx_particle = 0; idx_particle < m_number_particles; ++idx_particle) {
        std::ofstream file;
        file.open(fmt::format("{}/particle_{:04d}.csv", m_dir_export, idx_particle), std::ios_base::app);
        // Write iteration, position, velocity, best_position, best_fitness
        file << m_current_iteration;
        for (int i = 0; i < dimension; ++i) {
            file << fmt::format(",{:.6f}", m_particles[idx_particle].position[i]);
        }
        for (int i = 0; i < dimension; ++i) {
            file << fmt::format(",{:.6f}", m_particles[idx_particle].velocity[i]);
        }
        for (int i = 0; i < dimension; ++i) {
            file << fmt::format(",{:.6f}", m_particles[idx_particle].best_position[i]);
        }
        file << fmt::format(",{:.6f}", m_particles[idx_particle].best_fitness) << std::endl;
        file.close();
    }
    // Export swarm history
    std::ofstream file(m_dir_export + "/global_swarm.csv", std::ios_base::app);
    // Write iteration, best_position, best_fitness
    file << m_current_iteration;
    for (int i = 0; i < dimension; ++i) {
        file << fmt::format(",{:.6f}", m_best_position[i]);
    }
    file << fmt::format(",{:.6f}", m_best_fitness) << std::endl;
    file.close();
}


}  // namespace Optimization