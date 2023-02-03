/**
 *
 */

#include "PoissonSolver.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <random>
#include <vector>

#include <fmt/core.h>

#include "Physics.hpp"
#include "doping_profile.hpp"

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
    std::size_t number_of_points = my_doping_profile.get_x_line().size();
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
        m_hole_density(i) =
            m_intrisinc_carrier_concentration * std::exp((applied_voltage - m_solution(i)) / m_thermal_voltage);
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

    // Export the right hand side to a file
    std::ofstream file("rhs.csv");
    for (std::size_t index_x = 0; index_x < m_x_line.size(); ++index_x) {
        file << m_x_line(index_x) << "," << m_vector_rhs(index_x) << std::endl;
    }
    file.close();
}

void NewtonPoissonSolver::newton_solver(const double final_anode_voltage,
                                        const double tolerance,
                                        const int    max_iterations,
                                        double       voltage_step) {
    double       anode_voltage   = 0.0;
    const double cathode_voltage = 0.0;
    const double doping_anode    = m_doping_concentration(0);
    const double doping_cathode  = m_doping_concentration(m_x_line.size() - 1);

    compute_initial_guess();
    compute_total_charge(cathode_voltage, anode_voltage);
    // Export the total charge to a file
    std::ofstream file("total_charge.csv");
    for (std::size_t index_x = 0; index_x < m_x_line.size(); ++index_x) {
        file << m_x_line(index_x) << "," << m_total_charge(index_x) << std::endl;
    }
    file.close();
    compute_derivative_total_charge(cathode_voltage, anode_voltage);
    compute_matrix();
    update_matrix();
    compute_right_hand_side();
    // Export the right hand side to a file
    std::ofstream file_rhs("rhs000.csv");
    for (std::size_t index_x = 0; index_x < m_x_line.size(); ++index_x) {
        file_rhs << m_x_line(index_x) << "," << m_vector_rhs(index_x) << std::endl;
    }
    file_rhs.close();
    m_solver.analyzePattern(m_matrix);

    Eigen::VectorXd m_new_solution(m_x_line.size());
    const double    lambda = 1.0;


    std::size_t index_voltage_step = 0;
    while (anode_voltage <= final_anode_voltage) {
        std::cout << "Voltage anode: " << anode_voltage << std::endl;
        double residual = 1.0e10;
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
        index_voltage_step++;
        anode_voltage += voltage_step;
        // Fill index with zeros at the begining for filename
        std::ostringstream index_paded;
        index_paded << std::setw(6) << std::setfill('0') << index_voltage_step;
        std::string filename = "RES/solution_" + index_paded.str() + ".csv";
        export_solution(filename);
    }
}

void NewtonPoissonSolver::export_solution(const std::string& filename) const {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "X,V,eDensity,hDensity" << std::endl;
        for (std::size_t idx_x = 0; idx_x < m_x_line.size(); ++idx_x) {
            file << m_x_line(idx_x) << "," << m_solution(idx_x) << "," << m_electron_density(idx_x) << ","
                 << m_hole_density(idx_x) << std::endl;
        }
    file.close();
    }
}
