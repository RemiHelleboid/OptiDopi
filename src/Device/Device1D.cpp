/**
 * @file device.cpp
 */

#include "Device1D.hpp"

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/xchar.h>

#include <filesystem>
#include <iostream>

#include "fill_vector.hpp"
#include "gradient.hpp"
#include "interpolation.hpp"
#include "smoother.hpp"

void Device1D::add_doping_profile(doping_profile& doping_profile) { m_doping_profile = doping_profile; }

void Device1D::setup_constant_device(double x_length, std::size_t number_points, double doping_acceptor, double doping_donor) {}

void Device1D::setup_pin_diode(double      xlenght,
                               std::size_t number_points,
                               double      length_donor,
                               double      length_intrinsic,
                               double      donor_level,
                               double      acceptor_level,
                               double      intrisic_level) {
    m_doping_profile
        .set_up_pin_diode(0.0, xlenght, number_points, length_donor, length_intrinsic, donor_level, acceptor_level, intrisic_level);
}

void Device1D::set_up_complex_diode(double              xlength,
                                    std::size_t         number_points,
                                    double              length_donor,
                                    double              length_intrinsic,
                                    double              donor_level,
                                    double              intrisic_level,
                                    std::vector<double> list_x_acceptor,
                                    std::vector<double> list_acceptor_level) {
    m_doping_profile.set_up_advanced_pin(xlength,
                                         number_points,
                                         length_donor,
                                         length_intrinsic,
                                         donor_level,
                                         intrisic_level,
                                         list_x_acceptor,
                                         list_acceptor_level);
}

void Device1D::smooth_doping_profile(int window_size) { m_doping_profile.smooth_doping_profile(window_size); }

void Device1D::solve_poisson(const double final_anode_voltage, const double tolerance, const int max_iterations) {
    m_poisson_solver.set_doping_profile(m_doping_profile);
    const double voltage_step = 0.01;
    m_poisson_solver.newton_solver(final_anode_voltage, tolerance, max_iterations, voltage_step);
    m_list_voltages          = m_poisson_solver.get_list_voltages();
    m_list_poisson_solutions = m_poisson_solver.get_list_poisson_solutions();
}

PoissonSolution Device1D::get_poisson_solution_at_voltage(double voltage) const {
    auto it = std::lower_bound(m_list_voltages.begin(), m_list_voltages.end(), voltage);
    if (it == m_list_voltages.end()) {
        std::cout << "Voltage " << voltage << " not found in the list of voltages. Max voltage = " << m_list_voltages.back() << std::endl;
        throw std::runtime_error("Voltage not found");
    }
    std::size_t idx_voltage = std::distance(m_list_voltages.begin(), it);
    return m_list_poisson_solutions[idx_voltage];
}

void Device1D::export_poisson_solution(const std::string& directory_name, const std::string& prefix, double voltage_step) const {
    std::size_t freq_voltage = voltage_step / ((m_list_voltages.back() - m_list_voltages.front()) / m_list_voltages.size());
    if (freq_voltage < 1) {
        freq_voltage = 1;
    }
    std::filesystem::create_directories(directory_name);
    for (std::size_t idx_voltage = 0; idx_voltage < m_list_voltages.size(); idx_voltage += freq_voltage) {
        const std::string filename = fmt::format("{}/{}{:03.5f}.csv", directory_name, prefix, m_list_voltages[idx_voltage]);
        m_list_poisson_solutions[idx_voltage].export_to_file(filename);
    }
}

void Device1D::export_poisson_solution(const std::string& directory_name, const std::string& prefix) const {
    std::filesystem::create_directories(directory_name);
    for (std::size_t idx_voltage = 0; idx_voltage < m_list_voltages.size(); ++idx_voltage) {
        const std::string filename = fmt::format("{}/{}{:03.5f}.csv", directory_name, prefix, m_list_voltages[idx_voltage]);
        m_list_poisson_solutions[idx_voltage].export_to_file(filename);
    }
}

void Device1D::export_poisson_solution_at_voltage(double voltage, const std::string& directory_name, const std::string& prefix) const {
    // Find nearest voltage
    auto it = std::lower_bound(m_list_voltages.begin(), m_list_voltages.end(), voltage);
    if (it == m_list_voltages.end()) {
        std::cout << "Voltage " << voltage << " not found in the list of voltages. Max voltage = " << m_list_voltages.back() << std::endl;
        return;
    }
    std::size_t idx_voltage = std::distance(m_list_voltages.begin(), it);
    std::filesystem::create_directories(directory_name);
    const std::string filename = fmt::format("{}/{}{:03.5f}.csv", directory_name, prefix, m_list_voltages[idx_voltage]);
    m_list_poisson_solutions[idx_voltage].export_to_file(filename);
}

void Device1D::export_poisson_at_voltage_3D_emulation(double             voltage,
                                                      const std::string& directory_name,
                                                      const std::string& prefix,
                                                      double             DY,
                                                      double             DZ,
                                                      std::size_t        NY,
                                                      std::size_t        NZ) const {
    std::vector<double> x_line_micron(m_doping_profile.get_x_line());
    std::vector<double> y_line_micron = utils::linspace(0.0, DY, NY);
    std::vector<double> z_line_micron = utils::linspace(0.0, DZ, NZ);
    // Find nearest voltage
    auto it = std::lower_bound(m_list_voltages.begin(), m_list_voltages.end(), voltage);
    if (it == m_list_voltages.end()) {
        std::cout << "Voltage " << voltage << " not found in the list of voltages. Max voltage = " << m_list_voltages.back() << std::endl;
        return;
    }
    std::size_t           idx_voltage      = std::distance(m_list_voltages.begin(), it);
    const PoissonSolution poisson_solution = m_list_poisson_solutions[idx_voltage];
    std::vector<double>   phi              = poisson_solution.m_potential;
    std::vector<double>   n                = poisson_solution.m_electron_density;
    std::vector<double>   p                = poisson_solution.m_hole_density;
    std::vector<double>   electric_field   = poisson_solution.m_electric_field;

    const std::string filename = fmt::format("{}/{}{:03.5f}.csv", directory_name, prefix, m_list_voltages[idx_voltage]);
    std::ofstream     file(filename);
    file << "X,Y,Z,ElectrostaticPotential,ElectronDensity,HoleDensity,ElectricField,DopingConcentration" << std::endl;
    for (std::size_t idx_x = 0; idx_x < x_line_micron.size(); ++idx_x) {
        for (std::size_t idx_y = 0; idx_y < y_line_micron.size(); ++idx_y) {
            for (std::size_t idx_z = 0; idx_z < z_line_micron.size(); ++idx_z) {
                file << x_line_micron[idx_x] << "," << y_line_micron[idx_y] << "," << z_line_micron[idx_z] << "," << phi[idx_x] << ","
                     << n[idx_x] << "," << p[idx_x] << "," << electric_field[idx_x] << ","
                     << m_doping_profile.get_doping_concentration()[idx_x] << std::endl;
            }
        }
    }
    file.close();
}

void Device1D::solve_mcintyre(const double voltage_step, double stop_at_bv_plus) {
    double index_step     = (voltage_step / double(m_list_voltages[1] - m_list_voltages[0]));
    int    index_step_int = int(index_step);
    // std::cout << "index_step = " << index_step << std::endl;
    if (index_step == 0) {
        index_step = 1;
    }
    std::vector<double> x_line_micron(m_doping_profile.get_x_line());
    m_mcintyre_solver.set_xline(x_line_micron);
    // std::cout << "Max x = " << x_line_micron.back() << std::endl;
    double tol               = 1e-6;
    double breakdown_voltage = 0.0;
    double cm_to_micron      = 1.0e-4;
    for (std::size_t idx_voltage = 0; idx_voltage < m_list_voltages.size(); idx_voltage += index_step_int) {
        // std::cout << "Voltage = " << m_list_voltages[idx_voltage] << std::endl;
        m_mcintyre_solver.set_electric_field(m_list_poisson_solutions[idx_voltage].m_electric_field, true, cm_to_micron);
        m_mcintyre_solver.ComputeDampedNewtonSolution(tol);
        m_list_mcintyre_voltages.push_back(m_list_voltages[idx_voltage]);
        m_list_mcintyre_solutions.push_back(m_mcintyre_solver.get_solution());
        if (m_list_mcintyre_solutions.back().m_mean_breakdown_probability > 1e-3 && breakdown_voltage == 0.0) {
            breakdown_voltage = m_list_voltages[idx_voltage];
        }
        if (m_list_voltages[idx_voltage] > breakdown_voltage + stop_at_bv_plus && breakdown_voltage != 0.0) {
            break;
        }
    }
}

void Device1D::solve_poisson_and_mcintyre(const double final_anode_voltage,
                                          const double tolerance,
                                          const int    max_iterations,
                                          double       mcintyre_voltage_step,
                                          double       stop_at_bv_plus) {
    m_poisson_solver.set_doping_profile(m_doping_profile);
    const double voltage_step = 0.01;

    m_poisson_solver.newton_solver_with_mcintyre(final_anode_voltage,
                                                 tolerance,
                                                 max_iterations,
                                                 voltage_step,
                                                 mcintyre_voltage_step,
                                                 true,
                                                 stop_at_bv_plus);
    m_list_voltages          = m_poisson_solver.get_list_voltages();
    m_list_poisson_solutions = m_poisson_solver.get_list_poisson_solutions();

    m_list_mcintyre_voltages  = m_poisson_solver.get_list_mcintyre_voltages();
    m_list_mcintyre_solutions = m_poisson_solver.get_list_mcintyre_solutions();
}

void Device1D::export_depletion_width(const std::string& directory_name, const std::string& prefix) const {
    const double        epsilon                          = 1e-9;
    std::vector<double> list_total_breakdown_probability = m_poisson_solver.get_list_depletion_width(epsilon);
    std::filesystem::create_directories(directory_name);
    std::ofstream file;
    file.open(fmt::format("{}/{}.csv", directory_name, prefix));

    fmt::print(file, "Voltage,DepletionWidth\n");
    for (std::size_t idx_voltage = 0; idx_voltage < m_list_voltages.size(); ++idx_voltage) {
        fmt::print(file, "{:.3f},{:.3e}\n", m_list_voltages[idx_voltage], list_total_breakdown_probability[idx_voltage]);
    }
    file.close();
}

std::vector<double> Device1D::get_list_total_breakdown_probability() const {
    std::vector<double> list_total_breakdown_probability;
    for (const auto& mcintyre_solution : m_list_mcintyre_solutions) {
        list_total_breakdown_probability.push_back(mcintyre_solution.total_breakdown_probability.back());
    }
    return list_total_breakdown_probability;
}

double Device1D::extract_breakdown_voltage(double brp_threshold) const {
    std::vector<double> list_total_breakdown_probability = get_list_total_breakdown_probability();
    auto it_bv = std::find_if(list_total_breakdown_probability.begin(), list_total_breakdown_probability.end(), [&](const double& voltage) {
        return voltage > brp_threshold;
    });
    if (it_bv == list_total_breakdown_probability.end()) {
        // std::cout << "NaN BV" << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
    }
    // Interpolate the BV
    std::size_t idx_voltage  = std::distance(list_total_breakdown_probability.begin(), it_bv);
    double      voltage      = m_list_mcintyre_voltages[idx_voltage];
    double      brp          = list_total_breakdown_probability[idx_voltage];
    double      brp_prev     = list_total_breakdown_probability[idx_voltage - 1];
    double      voltage_prev = m_list_mcintyre_voltages[idx_voltage - 1];
    double      m_slope      = (brp - brp_prev) / (voltage - voltage_prev);
    double      m_intercept  = brp - m_slope * voltage;
    // std::cout << "Original BV: " << voltage << std::endl;
    // std::cout << "Interpolated BV: " << (brp_threshold - m_intercept) / m_slope << std::endl << std::endl;
    return (brp_threshold - m_intercept) / m_slope;
}

void Device1D::export_mcintyre_solution(const std::string& directory_name, const std::string& prefix) const {
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
    std::string   filename_global_brp = fmt::format("{}/Glob_{}_GlobalBRP_.csv", directory_name, prefix);
    std::ofstream file(filename_global_brp);
    file << "Voltage,MeanBreakdownProbability" << std::endl;
    for (std::size_t idx_voltage = 0; idx_voltage < m_list_mcintyre_voltages.size(); ++idx_voltage) {
        file << m_list_mcintyre_voltages[idx_voltage] << "," << m_list_mcintyre_solutions[idx_voltage].m_mean_breakdown_probability
             << std::endl;
    }
    std::cout << "Exporting the McIntyre solution to the directory " << directory_name << " done." << std::endl;
}

double Device1D::get_brp_at_voltage(double voltage) const {
    double interpolated_brp = Utils::interp1d(m_list_mcintyre_voltages, get_list_total_breakdown_probability(), voltage);
    return interpolated_brp;
}

double Device1D::get_depletion_at_voltage(double voltage) const {
    double epsilon                = 1e-7;
    double interpolated_depletion = Utils::interp1d(m_list_voltages, m_poisson_solver.get_list_depletion_width(epsilon), voltage);
    return interpolated_depletion;
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

cost_function_result Device1D::compute_cost_function(double voltage_above_breakdown, double time) const {
    const double alpha_BV  = 20.0;
    double       BV_TOL    = 2.0;
    double       BV_Target = 20.0;

    double alpha_BP = 5.0;
    double alpha_DW = 200.0;

    if (time > 0.8) {
        alpha_BP = 200.0;
        alpha_DW = 5.0;
    }

    double BreakdownVoltage     = extract_breakdown_voltage(1.0e-6);
    double BreakdownProbability = get_brp_at_voltage(BreakdownVoltage + voltage_above_breakdown);
    double DepletionWidth       = get_depletion_at_voltage(BreakdownVoltage);
    double BV_cost              = alpha_BV * std::pow(fabs((BreakdownVoltage - BV_Target) / BV_TOL), 10);
    double BP_cost              = -alpha_BP * BreakdownProbability;
    double meter_to_micron      = 1.0e6;
    double DW_cost              = -alpha_DW * (DepletionWidth / m_doping_profile.get_x_line().back()) * meter_to_micron;

    // If time == 0, we show the initial FoM.
    if (time == 0.0) {
        fmt::print("Initial FoM: BV = {:5.2f} V, BP = {:5.2f} %, DW = {:5.2f} um\n",
                   BreakdownVoltage,
                   BreakdownProbability * 100.0,
                   DepletionWidth * meter_to_micron);
    }

    if (std::isnan(BV_cost)) {
        BV_cost = 1.0e6;
    }
    double cost = BV_cost + BP_cost + DW_cost;
    cost        = 100.0 * cost;

    result_simu result;
    result.BV  = BreakdownVoltage;
    result.BrP = BreakdownProbability;
    result.DW  = DepletionWidth;
    return {result, BV_cost, BP_cost, DW_cost, cost};
}

void Device1D::DeviceADMCSimulation(const ADMC::ParametersADMC& parameters,
                                    double                      voltage,
                                    std::size_t                 nb_simulation_per_points,
                                    std::size_t                 nbXPoints,
                                    const std::string&          export_name) {
    ADMC::MainFullADMCSimulation(parameters, *this, voltage, nb_simulation_per_points, nbXPoints, export_name);
}

void Device1D::DeviceADMCSimulationToMaxField(const ADMC::ParametersADMC& parameters,
                                              double                      voltage,
                                              std::size_t                 nb_simulation_per_points,
                                              std::size_t                 nbXPoints,
                                              const std::string&          export_name) {
    ADMC::MainFullADMCSimulationToMaxField(parameters, *this, voltage, nb_simulation_per_points, nbXPoints, export_name);
}
