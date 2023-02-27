/**
 *
 */

#pragma once

#include <iostream>
#include <memory>
#include <random>
#include <vector>

/*! \file McIntyre.h
    \brief Header file of Mc Intyre implementation.
*/
#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <vector>

#include "ImpactIonization.hpp"

typedef Eigen::Triplet<double> T;

namespace mcintyre {

struct McIntyreSolution {
    double              m_voltage;
    double              m_mean_breakdown_probability;
    std::vector<double> total_breakdown_probability;
    std::vector<double> e_breakdown_probability;
    std::vector<double> h_breakdown_probability;
    bool                solver_has_converged;

    void export_to_file(const std::string& filename) const;
};

class McIntyre {
 private:
    const double        m_temperature = 300.0;
    std::vector<double> m_xline;
    std::vector<double> m_electric_field;
    Eigen::VectorXd     m_eRateImpactIonization;
    Eigen::VectorXd     m_hRateImpactIonization;
    //! The solution vector.
    /*! The vector is filled in the following way : \n BreakdownP[2k] = Breakdown probability for electron at
       StreamLine->GlobalLine[k] \n BreakdownP[2k+1] = Breakdown probability for hole at
       StreamLine->GlobalLine[k] */
    Eigen::VectorXd mBreakdownP;

    // Same format
    Eigen::VectorXd m_InitialGuessBreakdownP;

    // Final breakdown probability
    std::vector<double> m_totalBreakdownProbability;
    std::vector<double> m_eBreakdownProbability;
    std::vector<double> m_hBreakdownProbability;
    //! Flag to know the status of the convergence
    bool mSolverHasConverged = false;

    static double      m_McIntyre_time;
    static std::size_t m_total_number_sim;
    static std::size_t m_converged_sim;

 public:
    McIntyre() = default;
    McIntyre(std::vector<double> x_line, std::vector<double> electric_field, double temperature = 300.0);
    McIntyre(std::vector<double> x_line, double temperature = 300.0);
    ~McIntyre(){};

    static double      get_mcintyre_time() { return m_McIntyre_time; }
    static std::size_t get_total_number_sim() { return m_total_number_sim; }
    static std::size_t get_converged_sim() { return m_converged_sim; }
    static double      get_ratio_converged_sim() { return static_cast<double>(m_converged_sim) / static_cast<double>(m_total_number_sim); }

    void set_xline(std::vector<double> x_line);
    void set_xline(const Eigen::VectorXd& x_line);
    void set_electric_field(std::vector<double> electric_field, bool recompute_initial_guess = false, double conv_factor = 1.0);
    double get_max_electric_field() const;

    double                      dPe_func(double Pe, double Ph, int index);
    double                      dPh_func(double Pe, double Ph, int index);
    std::vector<double>         compute_N_pi(int i);
    std::vector<double>         computeA(int i, double Pe, double Ph);
    Eigen::SparseMatrix<double> assembleMat();
    Eigen::VectorXd             assembleSecondMemberNewton();
    Eigen::VectorXd             assembleSecondMemberNewton(Eigen::VectorXd breakdownP);
    void                        initial_guess(double eBrP = 0.8, double hBrP = 0.4);
    void                        ComputeDampedNewtonSolution(double tolerance);
    bool                        get_Solver_Has_Converged() const { return (mSolverHasConverged); };

    std::vector<double> get_total_breakdown_probability() const { return m_totalBreakdownProbability; }
    std::vector<double> get_electron_breakdown_probability() const { return m_eBreakdownProbability; }
    std::vector<double> get_hole_breakdown_probability() const { return m_hBreakdownProbability; }
    double              get_mean_total_breakdown_probability() const;

    McIntyreSolution get_solution() const {
        McIntyreSolution solution;
        solution.m_voltage                    = m_temperature;
        solution.m_mean_breakdown_probability = get_mean_total_breakdown_probability();
        solution.total_breakdown_probability  = get_total_breakdown_probability();
        solution.e_breakdown_probability      = get_electron_breakdown_probability();
        solution.h_breakdown_probability      = get_hole_breakdown_probability();
        solution.solver_has_converged         = get_Solver_Has_Converged();
        return solution;
    }
};

}  // namespace mcintyre
