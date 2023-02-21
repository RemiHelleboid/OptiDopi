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
        fmt::print("Exporting to {} ...", filename);
        std::ofstream file(filename);
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

class SimulatedAnnealing {
 private:
    std::size_t     m_nb_variables;
    std::size_t     m_max_iterations;
    std::size_t     m_current_iteration;
    CoolingSchedule m_cooling_schedule;
    double          m_initial_temperature;
    double          m_final_temperature;
    double          m_temperature;

    std::random_device                     m_random_device;
    std::mt19937                           m_generator;
    std::uniform_real_distribution<double> m_distribution;

    std::size_t m_frequency_print = 100;

    /**
     * @brief Cost function
     * First argument is the variables vector.
     *
     */
    std::function<double(std::vector<double>)> m_cost_function;

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

    std::string m_prefix_name = "";

 public:
    SimulatedAnnealing() = delete;
    SimulatedAnnealing(std::size_t                                m_nb_variables,
                       CoolingSchedule                            cooling_schedule,
                       std::size_t                                max_iterations,
                       double                                     initial_temperature,
                       double                                     final_temperature,
                       std::function<double(std::vector<double>)> cost_function);
    const std::string&  get_prefix_name() const { return m_prefix_name; }
    void                set_prefix_name(const std::string& prefix_name) { m_prefix_name = prefix_name; }
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

    void set_cooling_schedule(CoolingSchedule cooling_schedule) { m_cooling_schedule = cooling_schedule; }
    void set_frequency_print(std::size_t frequency_print) { m_frequency_print = frequency_print; }

    void linear_cooling();
    void geometrical_cooling(double alpha);
    void exponential_cooling(double alpha);
    void logarithmic_cooling();

    std::vector<double> neighbour_function();

    void run();

    SimulatedAnnealHistory get_history() const { return m_history; }
    void                   export_history();
};
