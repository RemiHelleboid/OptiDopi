/**
 *
 */

#include "PoissonSolver.hpp"

#include <fmt/core.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

#include "Physics.hpp"
#include "doping_profile.hpp"
#include "gradient.hpp"
#include "smoother.hpp"


double NewtonPoissonSolver::m_poisson_solver_time = 0.0;

double compute_depletion_width(const Eigen::VectorXd& electron_density,
                               const Eigen::VectorXd& x_line,
                               const Eigen::VectorXd& hole_density,
                               const Eigen::VectorXd& doping_concentration,
                               const double           epsilon) {
    constexpr double m3_to_cm3 = 1.0e6;
    Eigen::VectorXd depletion_shape     = electron_density - hole_density;
    std::size_t     sum_depleted_points = 0;
    for (unsigned int idx_x = 0; idx_x < x_line.size(); ++idx_x) {
        depletion_shape(idx_x) /= doping_concentration(idx_x) * m3_to_cm3;
        depletion_shape(idx_x) = fabs(depletion_shape(idx_x));
        if (depletion_shape(idx_x) < epsilon) {
            sum_depleted_points++;
        }
    }
    return 0.0;
}
double compute_depletion_width(const std::vector<double>& x_line,
                               const std::vector<double>& electron_density,
                               const std::vector<double>& hole_density,
                               const std::vector<double>& doping_concentration,
                               const double               epsilon) {
    std::size_t         sum_depleted_points = 0;
    std::vector<double> depletion_shape(x_line.size());
    constexpr double    m3_to_cm3 = 1.0e6;
    for (unsigned int idx_x = 0; idx_x < x_line.size(); ++idx_x) {
        depletion_shape[idx_x] = electron_density[idx_x] - hole_density[idx_x];
        depletion_shape[idx_x] /= (doping_concentration[idx_x] * m3_to_cm3);
        depletion_shape[idx_x] = fabs(depletion_shape[idx_x]);
        if (depletion_shape[idx_x] < epsilon) {
            sum_depleted_points++;
        }
    }
    double dx = x_line[1] - x_line[0];
    return sum_depleted_points * dx;
}

PoissonSolution::PoissonSolution(double              voltage,
                                 std::vector<double> x_line,
                                 std::vector<double> potential,
                                 std::vector<double> electron_density,
                                 std::vector<double> hole_density,
                                 std::vector<double> electric_field)
    : m_voltage(voltage),
      m_x_line(x_line),
      m_potential(potential),
      m_electron_density(electron_density),
      m_hole_density(hole_density),
      m_electric_field(electric_field) {}

PoissonSolution::PoissonSolution(double              voltage,
                                 Eigen::VectorXd     x_line,
                                 Eigen::VectorXd     potential,
                                 Eigen::VectorXd     electron_density,
                                 Eigen::VectorXd     hole_density,
                                 std::vector<double> electric_field)
    : m_voltage(voltage) {
    m_x_line           = std::vector<double>(x_line.data(), x_line.data() + x_line.size());
    m_potential        = std::vector<double>(potential.data(), potential.data() + potential.size());
    m_electron_density = std::vector<double>(electron_density.data(), electron_density.data() + electron_density.size());
    m_hole_density     = std::vector<double>(hole_density.data(), hole_density.data() + hole_density.size());
    m_electric_field   = electric_field;
}

void PoissonSolution::export_to_file(const std::string& filename) const {
    std::ofstream file(filename);
    file << "X,Potential,eDensity,hDensity,ElectricField" << std::endl;
    for (std::size_t i = 0; i < m_x_line.size(); ++i) {
        file << m_x_line[i] << "," << m_potential[i] << "," << m_electron_density[i] << "," << m_hole_density[i] << ","
             << m_electric_field[i] << std::endl;
    }
    file.close();
}

NewtonPoissonSolver::NewtonPoissonSolver(const Eigen::VectorXd& doping_concentration, const Eigen::VectorXd& x_line)
    : m_doping_concentration(doping_concentration),
      m_x_line(x_line) {
    m_matrix.resize(m_x_line.size(), m_x_line.size());
    m_vector_rhs.resize(m_x_line.size());
    m_solution.resize(m_x_line.size());
    m_electron_density.resize(m_x_line.size());
    m_hole_density.resize(m_x_line.size());
    m_total_charge.resize(m_x_line.size());
    m_derivative_total_charge.resize(m_x_line.size());
}

NewtonPoissonSolver::NewtonPoissonSolver(const doping_profile& my_doping_profile) {
    constexpr double microns_to_meters = 1e-6;
    constexpr double m3_to_cm3         = 1e6;
    std::size_t      number_of_points  = my_doping_profile.get_x_line().size();
    m_doping_concentration.resize(number_of_points);
    m_x_line.resize(number_of_points);
    for (std::size_t i = 0; i < number_of_points; ++i) {
        m_doping_concentration(i) = my_doping_profile.get_doping_concentration()[i] * m3_to_cm3;
        m_x_line(i)               = my_doping_profile.get_x_line()[i] * microns_to_meters;
    }

    m_matrix.resize(m_x_line.size(), m_x_line.size());
    m_vector_rhs.resize(m_x_line.size());
    m_solution.resize(m_x_line.size());
    m_electron_density.resize(m_x_line.size());
    m_hole_density.resize(m_x_line.size());
    m_total_charge.resize(m_x_line.size());
    m_derivative_total_charge.resize(m_x_line.size());
}

void NewtonPoissonSolver::set_doping_profile(const doping_profile& my_doping_profile) {
    constexpr double microns_to_meters = 1e-6;
    constexpr double m3_to_cm3         = 1e6;
    std::size_t      number_of_points  = my_doping_profile.get_x_line().size();
    m_doping_concentration.resize(number_of_points);
    m_x_line.resize(number_of_points);
    for (std::size_t i = 0; i < number_of_points; ++i) {
        m_doping_concentration(i) = my_doping_profile.get_doping_concentration()[i] * m3_to_cm3;
        m_x_line(i)               = my_doping_profile.get_x_line()[i] * microns_to_meters;
    }
    m_matrix.resize(m_x_line.size(), m_x_line.size());
    m_vector_rhs.resize(m_x_line.size());
    m_solution.resize(m_x_line.size());
    m_electron_density.resize(m_x_line.size());
    m_hole_density.resize(m_x_line.size());
    m_total_charge.resize(m_x_line.size());
    m_derivative_total_charge.resize(m_x_line.size());
}

void NewtonPoissonSolver::compute_electron_density(const double applied_voltage) {
    for (int i = 0; i < m_x_line.size(); ++i) {
        m_electron_density(i) = m_intrisinc_carrier_concentration * std::exp((m_solution(i) - applied_voltage) / m_thermal_voltage);
    }
}

void NewtonPoissonSolver::compute_hole_density(const double applied_voltage) {
    for (int i = 0; i < m_x_line.size(); ++i) {
        m_hole_density(i) = m_intrisinc_carrier_concentration * std::exp((applied_voltage - m_solution(i)) / m_thermal_voltage);
    }
}

void NewtonPoissonSolver::compute_total_charge(const double cathode_voltage, const double anode_voltage) {
    compute_electron_density(anode_voltage);
    compute_hole_density(cathode_voltage);
    for (int i = 0; i < m_x_line.size(); ++i) {
        m_total_charge(i) = m_hole_density(i) - m_electron_density(i);
    }
}

void NewtonPoissonSolver::compute_derivative_total_charge(const double cathode_voltage, const double anode_voltage) {
    for (int i = 0; i < m_x_line.size(); ++i) {
        m_derivative_total_charge(i) = (1.0 / m_thermal_voltage) * (-m_hole_density(i) - m_electron_density(i));
    }
}

void NewtonPoissonSolver::compute_initial_guess() {
    for (std::size_t index_x = 0; index_x < m_x_line.size(); ++index_x) {
        const double doping = m_doping_concentration(index_x);
        if (doping > 0.0) {
            const double value_inside_log =
                doping / (2.0 * m_intrisinc_carrier_concentration) +
                sqrt(1 + (doping * doping / (4.0 * m_intrisinc_carrier_concentration * m_intrisinc_carrier_concentration)));
            m_solution(index_x) = m_thermal_voltage * log(value_inside_log);
        } else {
            const double inverse_value_inside_log =
                sqrt(1 + ((doping * doping) / (4.0 * m_intrisinc_carrier_concentration * m_intrisinc_carrier_concentration))) -
                doping / (2.0 * m_intrisinc_carrier_concentration);
            m_solution(index_x) = m_thermal_voltage * log(1.0 / inverse_value_inside_log);
        }
    }
    // Take the opposite of the solution
    m_solution = -m_solution;
    // Export the initial guess to a file
    std::ofstream file("initial_guess.csv");
    for (std::size_t index_x = 0; index_x < m_x_line.size(); ++index_x) {
        file << m_x_line(index_x) << "," << m_solution(index_x) << std::endl;
    }
    file.close();
}

double NewtonPoissonSolver::compute_boundary_conditions(const double applied_voltage, const double doping) {
    double       v_contact      = applied_voltage;
    const double doping_contact = doping;
    if (doping_contact >= 0) {
        v_contact += -m_thermal_voltage * log(doping_contact / (2.0 * m_intrisinc_carrier_concentration) +
                                              sqrt(1 + std::pow(doping_contact / (2.0 * m_intrisinc_carrier_concentration), 2.0)));
    } else {
        v_contact +=
            -m_thermal_voltage * log(1.0 / (sqrt(1 + (doping_contact * doping_contact /
                                                      (4.0 * m_intrisinc_carrier_concentration * m_intrisinc_carrier_concentration))) -
                                            doping_contact / (2.0 * m_intrisinc_carrier_concentration)));
    }
    return v_contact;
}

void NewtonPoissonSolver::compute_matrix() {
    m_matrix.setZero();
    double dx_square = m_x_line(1) - m_x_line(0);
    dx_square *= dx_square;
    const double q_over_eps = electronic_charge / (vacuum_permittivity * silicon_premitivity);
    // Build the triplet list
    std::vector<Eigen::Triplet<double>> triplet_list;
    triplet_list.reserve(m_x_line.size() * 3);
    for (std::size_t idx_x = 1; idx_x < m_x_line.size() - 1; ++idx_x) {
        triplet_list.emplace_back(idx_x, idx_x - 1, -1.0 / dx_square);
        triplet_list.emplace_back(idx_x, idx_x + 1, -1.0 / dx_square);
        double diag_value = 2.0 / dx_square - q_over_eps * m_derivative_total_charge(idx_x);
        triplet_list.emplace_back(idx_x, idx_x, diag_value);
    }
    // Boundary conditions
    triplet_list.emplace_back(0, 0, 1.0);
    triplet_list.emplace_back(m_x_line.size() - 1, m_x_line.size() - 1, 1.0);
    m_matrix.setFromTriplets(triplet_list.begin(), triplet_list.end());
    m_matrix.makeCompressed();
}

void NewtonPoissonSolver::update_matrix() {
    double dx_square = m_x_line(1) - m_x_line(0);
    dx_square *= dx_square;
    const double q_over_eps = electronic_charge / (vacuum_permittivity * silicon_premitivity);
    for (std::size_t idx_x = 0; idx_x < m_x_line.size(); ++idx_x) {
        double diag_value               = 2.0 / dx_square - q_over_eps * m_derivative_total_charge(idx_x);
        m_matrix.coeffRef(idx_x, idx_x) = diag_value;
    }
}

void NewtonPoissonSolver::compute_right_hand_side() {
    m_vector_rhs.setZero();
    double       dx         = m_x_line(1) - m_x_line(0);
    const double q_over_eps = electronic_charge / (vacuum_permittivity * silicon_premitivity);
    for (std::size_t idx_x = 1; idx_x < m_x_line.size() - 1; ++idx_x) {
        m_vector_rhs(idx_x) = -q_over_eps * m_doping_concentration(idx_x) -
                              (1.0 / (dx * dx)) * (-m_solution(idx_x + 1) + 2.0 * m_solution(idx_x) - m_solution(idx_x - 1)) +
                              q_over_eps * m_total_charge(idx_x);
    }
    // Boundary conditions
    m_vector_rhs(0)                   = 0.0;
    m_vector_rhs(m_x_line.size() - 1) = 0.0;
}

void NewtonPoissonSolver::newton_solver(const double final_anode_voltage,
                                        const double tolerance,
                                        const int    max_iterations,
                                        double       voltage_step) {
    auto start = std::chrono::high_resolution_clock::now();

    double       anode_voltage   = 0.0;
    const double cathode_voltage = 0.0;
    const double doping_anode    = m_doping_concentration(0);
    const double doping_cathode  = m_doping_concentration(m_x_line.size() - 1);
    voltage_step                 = 3.0 * m_thermal_voltage;
    // fmt::print("thermal voltage: {:.3e}\n", m_thermal_voltage);

    std::vector<double> m_xline_vector(m_x_line.data(), m_x_line.data() + m_x_line.size());

    compute_initial_guess();
    compute_total_charge(cathode_voltage, anode_voltage);
    compute_derivative_total_charge(cathode_voltage, anode_voltage);
    compute_matrix();
    update_matrix();
    compute_right_hand_side();
    m_solver.analyzePattern(m_matrix);

    Eigen::VectorXd m_new_solution(m_x_line.size());
    const double    lambda = 1.0;

    std::size_t index_voltage_step = 0;
    while (anode_voltage <= final_anode_voltage) {
        // std::cout << "\rVoltage anode: " << anode_voltage << std::flush;
        double      residual        = 1.0e10;
        std::size_t index_iteration = 0;
        while (residual > tolerance && index_iteration < max_iterations) {
            index_iteration++;
            compute_total_charge(cathode_voltage, anode_voltage);
            compute_derivative_total_charge(cathode_voltage, anode_voltage);
            update_matrix();
            compute_right_hand_side();
            m_solver.factorize(m_matrix);
            m_new_solution = m_solver.solve(m_vector_rhs);
            residual       = m_new_solution.norm();
            m_solution += lambda * m_new_solution;
            m_solution(0)                   = compute_boundary_conditions(anode_voltage, doping_anode);
            m_solution(m_x_line.size() - 1) = compute_boundary_conditions(cathode_voltage, doping_cathode);
        }
        if (index_iteration == max_iterations) {
            std::cout << "Maximum number of iterations reached" << std::endl;
            throw std::runtime_error("Maximum number of iterations reached. Increase the number of iterations");
        }
        // Transform solution into std::vector
        std::vector<double> solution_vector(m_solution.data(), m_solution.data() + m_solution.size());
        std::vector<double> electric_field_vector = Utils::gradient(m_xline_vector, solution_vector);
        constexpr double    cm_to_m               = 1.0e-2;
        std::for_each(electric_field_vector.begin(), electric_field_vector.end(), [](double& value) { value *= cm_to_m; });
        // Take absolute value of electric field
        std::for_each(electric_field_vector.begin(), electric_field_vector.end(), [](double& value) { value = std::abs(value); });
        PoissonSolution solution(anode_voltage, m_x_line, m_solution, m_electron_density, m_hole_density, electric_field_vector);
        m_list_voltages.push_back(anode_voltage);
        m_list_poisson_solutions.push_back(solution);

        index_voltage_step++;
        anode_voltage += voltage_step;
    }
    auto                          end             = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    m_poisson_solver_time += elapsed_seconds.count();
}

double NewtonPoissonSolver::get_depletion_width(const double epsilon) const {
    return compute_depletion_width(m_x_line, m_electron_density, m_hole_density, m_doping_concentration, epsilon);
}

std::vector<double> NewtonPoissonSolver::get_list_depletion_width(const double epsilon) const {
    std::vector<double> vect_doping_concentration(m_doping_concentration.data(),
                                                  m_doping_concentration.data() + m_doping_concentration.size());
    std::vector<double> list_depletion_width;
    for (const auto& solution : m_list_poisson_solutions) {
        list_depletion_width.push_back(compute_depletion_width(solution.m_x_line,
                                                               solution.m_electron_density,
                                                               solution.m_hole_density,
                                                               vect_doping_concentration,
                                                               epsilon));
    }
    std::vector<double> smoothed_depletions = Utils::convol_square(list_depletion_width, 5);
    return smoothed_depletions;
}

void NewtonPoissonSolver::reset_all() {
    m_list_voltages.clear();
    m_list_poisson_solutions.clear();
    m_solution.resize(0);
    m_electron_density.resize(0);
    m_hole_density.resize(0);
    m_total_charge.resize(0);
    m_derivative_total_charge.resize(0);
    m_matrix.resize(0, 0);
    m_vector_rhs.resize(0);
}

void NewtonPoissonSolver::export_current_solution(const std::string& filename) const {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "X,V,eDensity,hDensity" << std::endl;
        for (std::size_t idx_x = 0; idx_x < m_x_line.size(); ++idx_x) {
            file << m_x_line(idx_x) << "," << m_solution(idx_x) << "," << m_electron_density(idx_x) << "," << m_hole_density(idx_x)
                 << std::endl;
        }
        file.close();
    }
}
