/**
 * @file device.hpp
 * @author remzerrr (remi.helleboid@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-05
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "AdvectionDiffusionMC.hpp"
#include "DopingProfile1D.hpp"
#include "McIntyre.hpp"
#include "PoissonSolver1D.hpp"

struct result_simu {
    double BV  = 0.0;
    double BrP = 0.0;
    double DW  = 0.0;

    void set_NaN() {
        BV  = std::numeric_limits<double>::quiet_NaN();
        BrP = std::numeric_limits<double>::quiet_NaN();
        DW  = std::numeric_limits<double>::quiet_NaN();
    }

    void set_very_high() {
        BV  = 1.0e6;
        BrP = 1.0e6;
        DW  = 1.0e6;
    }
};

struct cost_function_result {
    result_simu result;
    double      BV_cost;
    double      BP_cost;
    double      DW_cost;
    double      total_cost;

    void set_NaN() {
        result.set_NaN();
        BV_cost    = -std::numeric_limits<double>::quiet_NaN();
        BP_cost    = std::numeric_limits<double>::quiet_NaN();
        DW_cost    = std::numeric_limits<double>::quiet_NaN();
        total_cost = std::numeric_limits<double>::quiet_NaN();
    }

    void set_very_high() {
        BV_cost    = -1.0e6;
        BP_cost    = 1.0e6;
        DW_cost    = 1.0e6;
        total_cost = 1.0e6;
    }
};

class Device1D {
 private:
    std::string    m_name;
    doping_profile m_doping_profile;

    NewtonPoissonSolver m_poisson_solver;

    std::vector<double>          m_list_voltages;
    std::vector<PoissonSolution> m_list_poisson_solutions;

    mcintyre::McIntyre                      m_mcintyre_solver;
    std::vector<double>                     m_list_mcintyre_voltages;
    std::vector<mcintyre::McIntyreSolution> m_list_mcintyre_solutions;

 public:
    Device1D()                           = default;
    Device1D(const Device1D&)            = default;
    Device1D(Device1D&&)                 = default;
    Device1D& operator=(const Device1D&) = default;
    ~Device1D()                          = default;

    void               set_name(const std::string& name) { m_name = name; }
    const std::string& get_name() const { return m_name; }

    const doping_profile& get_doping_profile() const { return m_doping_profile; }

    void add_doping_profile(doping_profile& doping_profile);



    void setup_constant_device(double x_length, std::size_t number_points, double doping_acceptor, double doping_donor);
    void setup_pin_diode(double      x_length,
                         std::size_t number_points,
                         double      length_donor,
                         double      length_intrinsic,
                         double      donor_level,
                         double      acceptor_level,
                         double      intrinsic_level);

    void set_up_complex_diode(double              x_length,
                              std::size_t         number_points,
                              double              length_donor,
                              double              length_intrinsic,
                              double              donor_level,
                              double              intrinsic_level,
                              std::vector<double> list_x_acceptor,
                              std::vector<double> list_acceptor_level);

    void export_doping_profile(const std::string& filename) const { m_doping_profile.export_doping_profile(filename); }
    void smooth_doping_profile(int window_size);

    void            solve_poisson(const double final_anode_voltage, const double tolerance, const int max_iterations);
    bool            get_poisson_success() const { return m_poisson_solver.get_solver_success(); }
    void            export_poisson_solution(const std::string& directory_name, const std::string& prefix) const;
    void            export_poisson_solution(const std::string& directory_name, const std::string& prefix, double voltage_step) const;
    void            export_poisson_solution_at_voltage(double voltage, const std::string& directory_name, const std::string& prefix) const;
    void            export_poisson_at_voltage_3D_emulation(double             voltage,
                                                           const std::string& directory_name,
                                                           const std::string& prefix,
                                                           double             DY,
                                                           double             DZ,
                                                           std::size_t        NY,
                                                           std::size_t        NZ) const;
    PoissonSolution get_poisson_solution_at_voltage(double voltage) const;

    const std::vector<double>&          get_list_voltages() const { return m_list_voltages; }
    const std::vector<PoissonSolution>& get_list_poisson_solutions() const { return m_list_poisson_solutions; }
    std::vector<double>                 get_list_depletion_width() const;
    void                                export_depletion_width(const std::string& directory_name, const std::string& prefix) const;

    void                solve_mcintyre(const double voltage_step, double stop_at_bv_plus = 1e10);
    std::vector<double> get_list_total_breakdown_probability() const;
    void                export_mcintyre_solution(const std::string& directory_name, const std::string& prefix) const;
    double              extract_breakdown_voltage(double brp_threshold) const;

    void solve_poisson_and_mcintyre(const double final_anode_voltage,
                                    const double tolerance,
                                    const int    max_iterations,
                                    double       mcintyre_voltage_step,
                                    double       stop_at_bv_plus = 1e10);

    double get_brp_at_voltage(double voltage) const;
    double get_depletion_at_voltage(double voltage) const;

    cost_function_result compute_cost_function(double voltage_above_breakdown, double time) const;

    void DeviceADMCSimulation(const ADMC::ParametersADMC& parameters,
                              double                      voltage,
                              std::size_t                 nb_simulation_per_points,
                              std::size_t                 nbXPoints,
                              const std::string&          export_name);
};