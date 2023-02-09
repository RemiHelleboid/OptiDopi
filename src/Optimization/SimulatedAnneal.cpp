/**
 *
 */
#include "SimulatedAnneal.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "fmt/format.h"
#include "fmt/ostream.h"

SimulatedAnnealing::SimulatedAnnealing(std::size_t                                             m_nb_variables,
                                       CoolingSchedule                                         cooling_schedule,
                                       std::size_t                                             max_iterations,
                                       double                                                  initial_temperature,
                                       double                                                  final_temperature,
                                       std::function<double(std::vector<double>)>              cost_function,
                                       std::function<std::vector<double>(std::vector<double>)> neighbour_function)
    : m_nb_variables(m_nb_variables),
      m_cooling_schedule(cooling_schedule),
      m_max_iterations(max_iterations),
      m_current_iteration(0),
      m_initial_temperature(initial_temperature),
      m_final_temperature(final_temperature),
      m_temperature(initial_temperature),
      m_random_device(),
      m_generator(m_random_device()),
      m_distribution(0.0, 1.0),
      m_cost_function(cost_function),
      m_neighbour_function(neighbour_function),
      m_history(std::make_unique<SimulatedAnnealHistory>()) {
    m_bounds.resize(m_nb_variables);
    m_current_solution.resize(m_nb_variables);
}

void SimulatedAnnealing::set_bounds(std::vector<std::pair<double, double>> bounds) { m_bounds = bounds; }

void SimulatedAnnealing::set_initial_solution(std::vector<double> initial_solution) { m_current_solution = initial_solution; }

void SimulatedAnnealing::create_random_initial_solution() {
    m_current_solution.clear();
    for (auto bound : m_bounds) {
        m_current_solution.push_back(m_distribution(m_generator) * (bound.second - bound.first) + bound.first);
    }
}

void SimulatedAnnealing::set_max_iterations(std::size_t max_iterations) { m_max_iterations = max_iterations; }

void SimulatedAnnealing::set_initial_temperature(double initial_temperature) { m_initial_temperature = initial_temperature; }

void SimulatedAnnealing::set_final_temperature(double final_temperature) { m_final_temperature = final_temperature; }

std::vector<double> SimulatedAnnealing::clip_variables(const std::vector<double>& variables) {
    std::vector<double> clipped_variables;
    for (std::size_t i = 0; i < variables.size(); ++i) {
        clipped_variables.push_back(std::min(std::max(variables[i], m_bounds[i].first), m_bounds[i].second));
    }
    return clipped_variables;
}

void SimulatedAnnealing::add_current_solution_to_history() {
    m_history->solutions.push_back(m_current_solution);
    m_history->costs.push_back(m_current_cost);
    m_history->temperatures.push_back(m_temperature);
}

void SimulatedAnnealing::linear_cooling() {
    m_temperature = m_initial_temperature * (1.0 - (double)m_current_iteration / (double)m_max_iterations) +
                    m_final_temperature * (double)m_current_iteration / (double)m_max_iterations;
}

void SimulatedAnnealing::geometrical_cooling(double alpha) {
    m_temperature = m_initial_temperature * std::pow(alpha, (double)m_current_iteration);
}

void SimulatedAnnealing::exponential_cooling(double alpha) {
    m_temperature = m_initial_temperature * std::exp(-alpha * (double)m_current_iteration);
}

void SimulatedAnnealing::logarithmic_cooling() { m_temperature = m_initial_temperature / std::log(1.0 + (double)m_current_iteration); }

void SimulatedAnnealing::run() {
    m_current_cost  = m_cost_function(m_current_solution);
    m_best_solution = m_current_solution;
    m_best_cost     = m_current_cost;

    while (m_current_iteration < m_max_iterations && m_temperature > m_final_temperature) {
        std::vector<double> new_solution = m_neighbour_function(m_current_solution);
        new_solution                     = clip_variables(new_solution);
        double new_cost                  = m_cost_function(new_solution);

        if (new_cost < m_current_cost) {
            m_current_solution = new_solution;
            m_current_cost     = new_cost;
        } else {
            double probability = std::exp(-(new_cost - m_current_cost) / m_temperature);
            if (m_distribution(m_generator) < probability) {
                m_current_solution = new_solution;
                m_current_cost     = new_cost;
            }
        }

        if (m_current_cost < m_best_cost) {
            m_best_solution = m_current_solution;
            m_best_cost     = m_current_cost;
        }

        double alpha = 0.99;
        switch (m_cooling_schedule) {
            case CoolingSchedule::Linear:
                linear_cooling();
                break;
            case CoolingSchedule::Geometrical:
                geometrical_cooling(alpha);
                break;
            case CoolingSchedule::Exponential:
                exponential_cooling(0.0001);
                break;
            case CoolingSchedule::Logarithmic:
                logarithmic_cooling();
                break;
        }

        add_current_solution_to_history();
        // Print log with format: iteration, temperature, cost, with fmt::print
        if (m_current_iteration % 50 == 0) {
            fmt::print("{}, {}, {}\n", m_current_iteration, m_temperature, m_current_cost);
        }
        ++m_current_iteration;
    }
    fmt::print("{}, {}, {}\n", m_current_iteration, m_temperature, m_current_cost);
    // fmt::print("Best solution: {}, cost: {}\n", m_best_solution[0], m_best_cost);

    m_history->export_to_csv("history.csv");
}