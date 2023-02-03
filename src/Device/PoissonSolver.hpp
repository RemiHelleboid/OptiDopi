
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "Physics.hpp"
#include "doping_profile.hpp"

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

    void export_to_file(const std::string& filename) const;
};

class NewtonPoissonSolver {
 private:
    Eigen::VectorXd              m_x_line;
    Eigen::VectorXd              m_doping_concentration;
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

 public:
    NewtonPoissonSolver() = default;
    NewtonPoissonSolver(const Eigen::VectorXd& doping_concentration, const Eigen::VectorXd& x_line);
    NewtonPoissonSolver(const doping_profile& my_doping_profile);

    void set_doping_profile(const doping_profile& my_doping_profile);

    void compute_electron_density(const double applied_voltage);
    void compute_hole_density(const double applied_voltage);
    void compute_total_charge(const double cathode_voltage, const double anode_voltage);
    void compute_derivative_total_charge(const double cathode_voltage, const double anode_voltage);

    void compute_initial_guess();

    double compute_boundary_conditions(const double applied_voltage, const double doping);
    void   compute_matrix();
    void   update_matrix();
    void   compute_right_hand_side();

    void newton_solver(const double final_anode_voltage, const double tolerance, const int max_iterations, double voltage_step);

    std::vector<double>          get_list_voltages() const { return m_list_voltages; }
    std::vector<PoissonSolution> get_list_poisson_solutions() const { return m_list_poisson_solutions; }
    void                         export_current_solution(const std::string& filename) const;

    void reset_all();
};