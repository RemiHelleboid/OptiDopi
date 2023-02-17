/**
 * @file ParticleSwarm.hpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-02-16
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

namespace Optimization {

struct Particle {
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> best_position;
    double              best_fitness;
    std::mt19937        m_random_engine;

    Particle() = default;

    Particle(std::size_t number_dimensions) {
        position.resize(number_dimensions);
        velocity.resize(number_dimensions);
        best_position.resize(number_dimensions);
        std::random_device random_device;
        m_random_engine.seed(random_device());
    }

    void set_dimensions(std::size_t number_dimensions) {
        position.resize(number_dimensions);
        velocity.resize(number_dimensions);
        best_position.resize(number_dimensions);
    }

    void seed_random_engine() {
        std::random_device random_device;
        m_random_engine.seed(random_device());
    }

    std::mt19937& get_random_engine() { return m_random_engine; }
};

class ParticleSwarm {
 private:
    std::size_t           m_current_iteration = 0;
    std::vector<Particle> m_particles;
    std::vector<double>   m_bounds_min;
    std::vector<double>   m_bounds_max;
    double                m_velocity_scaling = 1.0e-1;

    std::vector<double> m_best_position;
    double              m_best_fitness;

    std::size_t m_number_particles;
    std::size_t m_number_dimensions;

    double m_inertia_weight;
    double m_cognitive_weight;
    double m_social_weight;

    std::function<double(const std::vector<double>&)> m_fitness_function;

    std::random_device                     m_random_device;
    std::mt19937                           m_random_engine;
    std::uniform_real_distribution<double> m_uniform_distribution;

    std::string m_dir_export = "ParticleSwarmResults/";

 public:
    ParticleSwarm(std::size_t                                       number_particles,
                  std::size_t                                       number_dimensions,
                  std::function<double(const std::vector<double>&)> fitness_function);
    ~ParticleSwarm() = default;

    void set_inertia_weight(double inertia_weight) { m_inertia_weight = inertia_weight; }
    void set_cognitive_weight(double cognitive_weight) { m_cognitive_weight = cognitive_weight; }
    void set_social_weight(double social_weight) { m_social_weight = social_weight; }

    void set_bounds(const std::vector<double>& bounds_min, const std::vector<double>& bounds_max);
    void set_bounds(std::vector<std::pair<double, double>> bounds);

    void initialize_particles();
    void update_particles();
    void clip_particles();

    void optimize(std::size_t number_iterations);

    std::vector<double> get_best_position() const { return m_best_position; }
    double              get_best_fitness() const { return m_best_fitness; }

    void set_file_export(const std::string& dir_name) { m_dir_export = dir_name; }

    void set_up_export();
    void export_current_state();
};
}  // namespace Optimization