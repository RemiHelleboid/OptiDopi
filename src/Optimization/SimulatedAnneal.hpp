/**
 *
 */

#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "fmt/format.h"
#include "fmt/ostream.h"

struct SimulatedAnnealHistory {
    std::vector<std::vector<double>> solutions;
    std::vector<double>              costs;
    std::vector<double>              temperatures;

    void export_to_csv(const std::string& filename) {
        std::ofstream file(filename);
        // Check if file is open
        if (!file.is_open()) {
            throw std::runtime_error(fmt::format("Could not open file {} for writing", filename));
        }
        file << "iteration,cost,temperature";
        for (std::size_t i = 0; i < solutions[0].size(); ++i) {
            file << fmt::format(",x{}", i);
        }
        file << std::endl;
        for (std::size_t i = 0; i < solutions.size(); ++i) {
            file << fmt::format("{},{},{},", i, costs[i], temperatures[i]);
            fmt::print(file, "{:e}", fmt::join(solutions[i], ","));
            file << std::endl;
        }
    }
};

enum class CoolingSchedule {
    Linear,
    Geometrical,
    Exponential,
    Logarithmic,
    Boltzmann,
};

struct SimulatedAnnealOptions {
    std::size_t     m_nb_variables;
    std::size_t     m_max_iterations;
    CoolingSchedule m_cooling_schedule;
    double          m_initial_temperature;
    double          m_final_temperature;
    double          m_alpha_cooling;
    double          m_beta_cooling;

    std::size_t       m_log_frequency   = 100;
    std::string m_prefix_name_log = "";

    SimulatedAnnealOptions() = default;
    SimulatedAnnealOptions(const SimulatedAnnealOptions&) = default;

};

class SimulatedAnnealing {
 private:
    SimulatedAnnealOptions m_options;

    double      m_temperature;
    std::size_t m_current_iteration;

    std::random_device                     m_random_device;
    std::mt19937                           m_generator;
    std::uniform_real_distribution<double> m_distribution;

    /**
     * @brief Cost function
     * First argument is the variables vector.
     *
     */
    std::function<double(std::vector<double>, const std::vector<double>&)> m_cost_function;

    /**
     * @brief Cooling function, returns the new temperature based on the current temperature.
     *
     */
    std::function<double(double)> m_cooling_function;

    double m_alpha_cooling;

    /**
     * @brief Acceptance function, returns true if the new solution is accepted.
     * First argument is the current cost.
     * Second argument is the new cost.
     *
     */
    std::function<double(double, double)> m_acceptance_function;

    /**
     * @brief Bounds of the variables vector.
     *
     */
    std::vector<std::pair<double, double>> m_bounds;

    std::vector<double> m_current_solution;
    std::vector<double> m_best_solution;

    double m_current_cost;
    double m_best_cost;

    SimulatedAnnealHistory m_history;

 public:
    SimulatedAnnealing() = delete;
    SimulatedAnnealing(const SimulatedAnnealOptions&                                          sa_options,
                       std::function<double(std::vector<double>, const std::vector<double>&)> cost_function);

    SimulatedAnnealOptions&       options() { return m_options; }
    const SimulatedAnnealOptions& get_options() const { return m_options; }

    void                set_bounds(std::vector<std::pair<double, double>> bounds);
    void                set_bounds(std::vector<double> bounds_min, std::vector<double> bounds_max);
    std::vector<double> clip_variables(const std::vector<double>& variables) const;

    void set_initial_solution(std::vector<double> initial_solution);
    void create_random_initial_solution();
    void set_max_iterations(std::size_t max_iterations);
    void set_initial_temperature(double initial_temperature);
    void set_final_temperature(double final_temperature);
    void set_alpha_cooling(double alpha_cooling) { m_alpha_cooling = alpha_cooling; }

    const std::vector<double>& get_best_solution() const { return m_best_solution; }
    const std::vector<double>& get_current_solution() const { return m_current_solution; }
    double                     get_best_cost() const { return m_best_cost; }
    double                     get_current_cost() const { return m_current_cost; }

    void add_current_solution_to_history();

    void linear_cooling();
    void geometrical_cooling(double alpha);
    void exponential_cooling(double alpha);
    void logarithmic_cooling();

    std::vector<double> neighbour_function();

    void run();
    void restart();

    SimulatedAnnealHistory get_history() const { return m_history; }
    void                   export_history();
};
