/**
 * @file device.cpp
 */

#include "device.hpp"

#include <fmt/core.h>

#include <filesystem>
#include <iostream>

#include "gradient.hpp"

void device::add_doping_profile(doping_profile& doping_profile) { m_doping_profile = doping_profile; }

void device::setup_pin_diode(double      xlenght,
                             std::size_t number_points,
                             double      length_donor,
                             double      length_intrinsic,
                             double      donor_level,
                             double      acceptor_level,
                             double      intrisic_level) {
    m_doping_profile
        .set_up_pin_diode(0.0, xlenght, number_points, length_donor, length_intrinsic, donor_level, acceptor_level, intrisic_level);
}

void device::solve_poisson(const double final_anode_voltage, const double tolerance, const int max_iterations) {
    m_poisson_solver.set_doping_profile(m_doping_profile);
    const double voltage_step = 0.01;
    m_poisson_solver.newton_solver(final_anode_voltage, tolerance, max_iterations, voltage_step);

    m_list_voltages          = m_poisson_solver.get_list_voltages();
    m_list_poisson_solutions = m_poisson_solver.get_list_poisson_solutions();
}

void device::export_poisson_solution(const std::string& directory_name, const std::string& prefix) const {
    std::cout << "Exporting the poisson solution to the directory " << directory_name << std::endl;
    std::filesystem::create_directories(directory_name);
    for (std::size_t idx_voltage = 0; idx_voltage < m_list_voltages.size(); ++idx_voltage) {
        const std::string filename = fmt::format("{}/{}{:03.5f}.csv", directory_name, prefix, m_list_voltages[idx_voltage]);
        m_list_poisson_solutions[idx_voltage].export_to_file(filename);
    }
    std::cout << "Exporting the poisson solution to the directory " << directory_name << " done." << std::endl;
}

void device::solve_mcintyre(const double voltage_step) {
    std::size_t index_step = (m_list_voltages[1] - m_list_voltages[0]) / voltage_step;
    if (index_step == 0) {
        index_step = 1;
    }
    std::cout << "index_step = " << index_step << std::endl;
    const double micron_to_cm = 1.0e-4;
    std::vector<double> x_line_cm(m_doping_profile.get_x_line());
    for (auto& x : x_line_cm) {
        x *= micron_to_cm;
    }
    std::cout << "total length = " << x_line_cm.back() << " cm" << std::endl;
    m_mcintyre_solver.set_xline(x_line_cm);
    double tol = 1e-9;
    for (std::size_t idx_voltage = 0; idx_voltage < m_list_voltages.size(); idx_voltage += index_step) {
        std::cout << "Solving McIntyre for voltage = " << m_list_voltages[idx_voltage] << std::endl;
        m_mcintyre_solver.set_electric_field(m_list_poisson_solutions[idx_voltage].m_electric_field);
        m_mcintyre_solver.ComputeDampedNewtonSolution(tol);
        m_list_mcintyre_voltages.push_back(m_list_voltages[idx_voltage]);
        m_list_mcintyre_solutions.push_back(m_mcintyre_solver.get_solution());
    }
}

void device::export_mcintyre_solution(const std::string& directory_name, const std::string& prefix) const {
    std::cout << "Exporting the McIntyre solution to the directory " << directory_name << std::endl;
    std::filesystem::create_directories(directory_name);
    auto x_line = m_doping_profile.get_x_line();
    for (std::size_t idx_voltage = 0; idx_voltage < m_list_mcintyre_voltages.size(); ++idx_voltage) {
        const std::string filename = fmt::format("{}/{}{:03.5f}.csv", directory_name, prefix, m_list_mcintyre_voltages[idx_voltage]);
        std::ofstream     file(filename);
        file << "X,eBreakdownProbability,hBreakdownProbability,totalBreakdownProbability" << std::endl;
        for (std::size_t idx_x = 0; idx_x < x_line.size(); ++idx_x) {
            file << x_line[idx_x] << "," << m_list_mcintyre_solutions[idx_voltage].e_breakdown_probability[idx_x] << ","
                 << m_list_mcintyre_solutions[idx_voltage].h_breakdown_probability[idx_x] << ","
                 << m_list_mcintyre_solutions[idx_voltage].total_breakdown_probability[idx_x] << std::endl;
        }
        file.close();
    }
    std::string   filename_global_brp = fmt::format("{}/{}_GlobalBRP_.csv", directory_name, prefix);
    std::ofstream file(filename_global_brp);
    file << "Voltage,MeanBreakdownProbability" << std::endl;
    for (std::size_t idx_voltage = 0; idx_voltage < m_list_mcintyre_voltages.size(); ++idx_voltage) {
        file << m_list_mcintyre_voltages[idx_voltage] << "," << m_list_mcintyre_solutions[idx_voltage].m_mean_breakdown_probability
             << std::endl;
    }
    std::cout << "Exporting the McIntyre solution to the directory " << directory_name << " done." << std::endl;
}
