
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "DopingProfile1D.hpp"
#include "McIntyre.hpp"
#include "Physics.hpp"

typedef Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> SolverSparseLU;

struct PoissonSolution {
    double              m_voltage;
    std::vector<double> m_x_line;
    std::vector<double> m_potential;
    std::vector<double> m_electron_density;
    std::vector<double> m_hole_density;
    std::vector<double> m_electric_field;

    PoissonSolution(double              voltage,
                    std::vector<double> x_line,
                    std::vector<double> potential,
                    std::vector<double> electron_density,
                    std::vector<double> hole_density,
                    std::vector<double> electric_field);

    PoissonSolution(double              voltage,
                    Eigen::VectorXd     x_line,
                    Eigen::VectorXd     potential,
                    Eigen::VectorXd     electron_density,
                    Eigen::VectorXd     hole_density,
                    std::vector<double> electric_field);

    PoissonSolution() = default;

    double get_max_electric_field() const { return *std::max_element(m_electric_field.begin(), m_electric_field.end()); }

    void export_to_file(const std::string& filename) const;
};

class NewtonPoissonSolver {
 private:
    Eigen::VectorXd              m_x_line;
    Eigen::VectorXd              m_doping_concentration;
    std::size_t                  m_number_points;
    const double                 m_temperature                     = 300.0;
    const double                 m_thermal_voltage                 = boltzmann_constant_eV * m_temperature;
    const double                 m_intrisinc_carrier_concentration = 1.1e16;
    Eigen::SparseMatrix<double>  m_matrix;
    Eigen::VectorXd              m_vector_rhs;
    Eigen::VectorXd              m_solution;
    SolverSparseLU               m_solver;
    Eigen::VectorXd              m_electron_density;
    Eigen::VectorXd              m_hole_density;
    Eigen::VectorXd              m_total_charge;
    Eigen::VectorXd              m_derivative_total_charge;
    std::vector<double>          m_list_voltages;
    std::vector<PoissonSolution> m_list_poisson_solutions;
    bool                         m_solver_success = false;

    mcintyre::McIntyre                      m_mcintyre_solver;
    std::vector<double>                     m_list_mcintyre_voltages;
    std::vector<mcintyre::McIntyreSolution> m_list_mcintyre_solutions;
    bool                                    m_solve_with_mcintyre   = false;
    double                                  m_step_voltage_mcintyre = 0.0;

    static double m_poisson_solver_time;

 public:
    NewtonPoissonSolver() = default;
    NewtonPoissonSolver(const Eigen::VectorXd& doping_concentration, const Eigen::VectorXd& x_line);
    NewtonPoissonSolver(const doping_profile& my_doping_profile);

    static double get_poisson_solver_time() { return m_poisson_solver_time; }

    void set_doping_profile(const doping_profile& my_doping_profile);

    void compute_electron_density(const double applied_voltage);
    void compute_hole_density(const double applied_voltage);
    void compute_total_charge(const double cathode_voltage, const double anode_voltage);
    void compute_derivative_total_charge();

    void compute_initial_guess();

    double compute_boundary_conditions(const double applied_voltage, const double doping);
    void   compute_matrix();
    void   update_matrix();
    void   compute_right_hand_side();

    void newton_solver(const double final_anode_voltage, const double tolerance, const std::size_t max_iterations, double voltage_step);
    void newton_solver_with_mcintyre(const double      final_anode_voltage,
                                     const double      tolerance,
                                     const std::size_t max_iterations,
                                     double            voltage_step,
                                     const double      step_voltage_mcintyre,
                                     bool              stop_at_bvPlus = false,
                                     double            bvPlus         = 3.0);

    bool                get_solver_success() const { return m_solver_success; }
    double              get_depletion_width(const double epsilon) const;
    std::vector<double> get_list_depletion_width(const double epsilon) const;

    std::vector<double>          get_list_voltages() const { return m_list_voltages; }
    std::vector<PoissonSolution> get_list_poisson_solutions() const { return m_list_poisson_solutions; }
    void                         export_current_solution(const std::string& filename) const;

    std::vector<double>                     get_list_mcintyre_voltages() const { return m_list_mcintyre_voltages; }
    std::vector<mcintyre::McIntyreSolution> get_list_mcintyre_solutions() const { return m_list_mcintyre_solutions; }

    void reset_all();
};