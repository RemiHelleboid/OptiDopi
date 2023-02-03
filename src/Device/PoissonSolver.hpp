
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

class NewtonPoissonSolver {
 private:
    Eigen::SparseMatrix<double> m_matrix;
    Eigen::VectorXd             m_vector_rhs;
    Eigen::VectorXd             m_x_line;
    Eigen::VectorXd             m_solution;
    Eigen::VectorXd             m_doping_concentration;
    Eigen::VectorXd             m_electron_density;
    Eigen::VectorXd             m_hole_density;
    Eigen::VectorXd             m_total_charge;
    Eigen::VectorXd             m_derivative_total_charge;
    const double                m_temperature                     = 300.0;
    const double                m_thermal_voltage                 = boltzmann_constant_eV * m_temperature;
    const double                m_intrisinc_carrier_concentration = 1.1e16;
    SolverSparseLU              m_solver;

 public:
    NewtonPoissonSolver(const Eigen::VectorXd& doping_concentration, const Eigen::VectorXd& x_line);
    NewtonPoissonSolver(const doping_profile& my_doping_profile);

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

    void export_solution(const std::string& filename) const;
};