/*! \file McIntyre.h
 *  \brief Source file of Mc Intyre implementation.
 *
 *  Details.
 */

#include "McIntyre.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>

#include "ImpactIonization.hpp"
#include "smoother.hpp"

namespace mcintyre {

double      McIntyre::m_McIntyre_time    = 0.0;
std::size_t McIntyre::m_total_number_sim = 0;
std::size_t McIntyre::m_converged_sim    = 0;

McIntyre::McIntyre(std::vector<double> x_line, double temperature) : m_temperature(temperature), m_xline(x_line) {
    mBreakdownP              = Eigen::VectorXd::Zero(2 * m_xline.size());
    m_InitialGuessBreakdownP = Eigen::VectorXd::Zero(2 * m_xline.size());
    mSolverHasConverged      = false;
    m_eRateImpactIonization.resize(m_xline.size());
    m_hRateImpactIonization.resize(m_xline.size());
    const double Gamma = compute_gamma(m_temperature);
    const double E_g   = compute_band_gap(m_temperature);
    for (std::size_t idx_x = 0; idx_x < m_xline.size(); idx_x++) {
        m_eRateImpactIonization[idx_x] = alpha_DeMan(m_electric_field[idx_x], Gamma, E_g);
        m_hRateImpactIonization[idx_x] = beta_DeMan(m_electric_field[idx_x], Gamma, E_g);
    }
}

// McIntyre::McIntyre(std::vector<double> x_line, std::vector<double> electric_field, double temperature) : m_temperature(temperature) {
//     m_xline                  = x_line;
//     m_electric_field         = electric_field;
//     mBreakdownP              = Eigen::VectorXd::Zero(2 * m_xline.size());
//     m_InitialGuessBreakdownP = Eigen::VectorXd::Zero(2 * m_xline.size());
//     mSolverHasConverged      = false;
//     m_eRateImpactIonization.resize(m_xline.size());
//     m_hRateImpactIonization.resize(m_xline.size());
//     const double Gamma = compute_gamma(m_temperature);
//     const double E_g   = compute_band_gap(m_temperature);
//     for (std::size_t idx_x = 0; idx_x < m_xline.size(); idx_x++) {
//         m_eRateImpactIonization[idx_x] = alpha_DeMan(m_electric_field[idx_x], Gamma, E_g);
//         m_hRateImpactIonization[idx_x] = beta_DeMan(m_electric_field[idx_x], Gamma, E_g);
//     }
//     this->initial_guess();
// }

void McIntyre::set_xline(std::vector<double> x_line) {
    m_xline                  = x_line;
    mBreakdownP              = Eigen::VectorXd::Zero(2 * m_xline.size());
    m_InitialGuessBreakdownP = Eigen::VectorXd::Zero(2 * m_xline.size());
    mSolverHasConverged      = false;
    m_eRateImpactIonization.resize(m_xline.size());
    m_hRateImpactIonization.resize(m_xline.size());
}

void McIntyre::set_xline(const Eigen::VectorXd& x_line) {
    m_xline = std::vector<double>(x_line.data(), x_line.data() + x_line.size());
    this->set_xline(m_xline);
}

void McIntyre::set_electric_field(std::vector<double> electric_field, bool recompute_initial_guess, double conv_factor) {
    if (electric_field.size() != m_xline.size()) {
        throw std::invalid_argument("The size of the electric field vector is not the same as the size of the xline vector.");
    }
    // m_electric_field = electric_field;
    m_electric_field = Utils::convol_square(electric_field, 5);
    std::transform(m_electric_field.begin(), m_electric_field.end(), m_electric_field.begin(), [conv_factor](double x) {
        return x * conv_factor;
    });
    // std::cout << "Max electric field: " << *std::max_element(m_electric_field.begin(), m_electric_field.end()) << std::endl;
    // std::cout << "Max xline: " << *std::max_element(m_xline.begin(), m_xline.end()) << std::endl;
    const double Gamma = compute_gamma(m_temperature);
    const double E_g   = compute_band_gap(m_temperature);
    for (std::size_t idx_x = 0; idx_x < m_xline.size(); idx_x++) {
        // std::cout << "Electric field: " << m_electric_field[idx_x] << std::endl;
        m_eRateImpactIonization[idx_x] = alpha_DeMan(m_electric_field[idx_x], Gamma, E_g);
        m_hRateImpactIonization[idx_x] = beta_DeMan(m_electric_field[idx_x], Gamma, E_g);
    }
    double total_brp = mBreakdownP.norm();
    if (!mSolverHasConverged || total_brp <= 0.5) {
        std::cout << "Recomputing initial guess" << std::endl;
        this->initial_guess();
    }
}

/*! \brief Computes the derivate of Pe over the x variable.
 *  The formula is given by the McIntyre model.
 *  \return double dpe : The derivative over x of Pe. */
double McIntyre::dPe_func(double Pe, double Ph, int index) {
    double dpe = (1 - Pe) * m_eRateImpactIonization[index] * (Pe + Ph - Pe * Ph);
    return dpe;
}

/*! \brief Computes the derivate of Ph over the x variable.
 *  The formula is given by the McIntyre model.
 *  \return double dph : The derivative over x of Ph. */
double McIntyre::dPh_func(double Pe, double Ph, int index) {
    double dph = -(1 - Ph) * m_hRateImpactIonization[index] * (Pe + Ph - Pe * Ph);
    return dph;
}

/*! \brief Computes the i-st elementary N_pi vector of dimension 2x1.
 *  It will be used to assemble the global problem to solve.
 *  \return std::vector<double> : The i-st N_pi vector. */
std::vector<double> McIntyre::compute_N_pi(int i) {
    std::vector<double> N_pi(2);
    double              h   = m_xline[i + 1] - m_xline[i];
    double              y_e = (1.0 / h) * (mBreakdownP[2 * i + 2] - mBreakdownP[2 * i]);
    double              y_h = (1.0 / h) * (mBreakdownP[2 * i + 3] - mBreakdownP[2 * i + 1]);
    double              f_e =
        0.5 * (dPe_func(mBreakdownP[2 * i], mBreakdownP[2 * i + 1], i) + dPe_func(mBreakdownP[2 * i + 2], mBreakdownP[2 * i + 3], i + 1));
    double f_h =
        0.5 * (dPh_func(mBreakdownP[2 * i], mBreakdownP[2 * i + 1], i) + dPh_func(mBreakdownP[2 * i + 2], mBreakdownP[2 * i + 3], i + 1));

    N_pi[0] = y_e - f_e;
    N_pi[1] = y_h - f_h;
    return N_pi;
}

/*! \brief Computes the i-st elementary A matrix of dimension 2x2.
 *  It will be used to assemble the global problem to solve.
 *  \return std::vector<double> : The i-st A matrix. */
std::vector<double> McIntyre::computeA(int i, double Pe, double Ph) {
    std::vector<double> A(4);
    double              alpha_e = m_eRateImpactIonization[i];
    double              alpha_h = m_hRateImpactIonization[i];

    A[0] = alpha_e * (1 - 2 * Pe - 2 * Ph + 2 * Pe * Ph);   // A00
    A[1] = alpha_e * (1 - 2 * Pe + Pe * Pe);                // A01
    A[2] = alpha_h * (-1 + 2 * Ph - Ph * Ph);               // A10
    A[3] = alpha_h * (-1 + 2 * Pe + 2 * Ph - 2 * Pe * Ph);  // A11
    return A;
}

/*! \brief Computes the global matrix of the problem.
 *
 *  \return Eigen::SparseMatrix MatA. */
Eigen::SparseMatrix<double> McIntyre::assembleMat() {
    std::size_t                 N = m_xline.size();
    Eigen::SparseMatrix<double> MatA(2 * N, 2 * N);
    std::vector<T>              tripletList;
    const std::size_t           NNZ_estim = 8 * N;

    tripletList.reserve(NNZ_estim);

    for (std::size_t i = 0; i < N - 1; ++i) {
        double              x    = m_xline[i];
        double              xpp  = m_xline[i + 1];
        double              h    = xpp - x;
        std::vector<double> Ai   = computeA(i, mBreakdownP[2 * i], mBreakdownP[2 * i + 1]);
        std::vector<double> Aipp = computeA(i + 1, mBreakdownP[2 * i + 2], mBreakdownP[2 * i + 3]);

        tripletList.push_back(T(2 * i, 2 * i, -(1.0 / h) - 0.5 * Ai[0]));
        tripletList.push_back(T(2 * i, 2 * i + 1, -0.5 * Ai[1]));
        tripletList.push_back(T(2 * i, 2 * i + 2, (1.0 / h) - 0.5 * Aipp[0]));
        tripletList.push_back(T(2 * i, 2 * i + 3, -0.5 * Aipp[1]));

        tripletList.push_back(T(2 * i + 1, 2 * i, -0.5 * Ai[2]));
        tripletList.push_back(T(2 * i + 1, 2 * i + 1, -(1.0 / h) - 0.5 * Ai[3]));
        tripletList.push_back(T(2 * i + 1, 2 * i + 2, -0.5 * Aipp[2]));
        tripletList.push_back(T(2 * i + 1, 2 * i + 3, (1.0 / h) - 0.5 * Aipp[3]));
    }
    tripletList.push_back(T(2 * N - 1, 2 * N - 1, 1.0));
    tripletList.push_back(T(2 * N - 2, 0, 1.0));
    MatA.setFromTriplets(tripletList.begin(), tripletList.end());
    // Check if the matrix is symmetric

    return MatA;
}

/*! \brief Computes the global second member of the linear problem to solve.
 *
 *  \return Eigen::VectorXd B. */
Eigen::VectorXd McIntyre::assembleSecondMemberNewton() {
    std::size_t     N = m_xline.size();
    Eigen::VectorXd B(2 * N);

    for (std::size_t i = 0; i < N - 1; i++) {
        std::vector<double> q_i = compute_N_pi(i);
        B[2 * i]                = -q_i[0];
        B[2 * i + 1]            = -q_i[1];
    }

    B[2 * N - 2] = mBreakdownP[0];
    B[2 * N - 1] = mBreakdownP[2 * N - 1];
    return (B);
}

/*! \brief Computes an initial guess for the Newton solver algorithm.
 *
 *  \return std::vector<double> Y the initial guess. */
void McIntyre::initial_guess(double eBrP, double hBrP) {
    int                 NbPoints                      = m_xline.size();
    auto                index_max_electric_field      = std::max_element(m_electric_field.begin(), m_electric_field.end());
    double              length_peack_triggering_force = m_xline[std::distance(m_electric_field.begin(), index_max_electric_field)];
    std::vector<double> eInitialBrP(NbPoints);
    std::vector<double> hInitialBrP(NbPoints);
    for (int i = 0; i < NbPoints; i++) {
        eInitialBrP[i] = eBrP * (m_xline[i] >= length_peack_triggering_force);
        hInitialBrP[i] = hBrP * (m_xline[i] < length_peack_triggering_force);
    }
    // Smooth the initial guess
    eInitialBrP = Utils::convol_square(eInitialBrP, 5);
    hInitialBrP = Utils::convol_square(hInitialBrP, 5);
    m_InitialGuessBreakdownP.resize(2 * NbPoints);
    for (int i = 0; i < NbPoints; i++) {
        m_InitialGuessBreakdownP[2 * i]     = eInitialBrP[i];
        m_InitialGuessBreakdownP[2 * i + 1] = hInitialBrP[i];
    }
}

/*! \brief Computes the solution of the coupled non-linear McIntyre system.
 *
 *  \return std::vector<double> The computed solution of the system. */
void McIntyre::ComputeDampedNewtonSolution(double tolerance) {
    auto            start  = std::chrono::high_resolution_clock::now();
    std::size_t     N      = m_xline.size();
    double          Norm_w = 1e10;
    int             epoch  = 0;
    Eigen::VectorXd W(2 * N);
    Eigen::VectorXd W_old(2 * N);
    mBreakdownP                                       = m_InitialGuessBreakdownP;
    Eigen::SparseMatrix<double>                   MAT = assembleMat();
    Eigen::VectorXd                               B   = assembleSecondMemberNewton();
    Eigen::SparseLU<Eigen::SparseMatrix<double> > EigenSolver;
    EigenSolver.analyzePattern(MAT);
    int                              MaxEpoch = 2500;
    double                           factor   = 1.0;
    double                           lambda   = 1.0;
    std::random_device               rd;
    std::mt19937                     gen(rd());
    std::uniform_real_distribution<> dis(-0.05, 0.05);
    while (Norm_w > tolerance && epoch < MaxEpoch) {
        double lambda = 1.0;
        MAT           = assembleMat();
        B             = assembleSecondMemberNewton();
        EigenSolver.factorize(MAT);
        if (EigenSolver.info() != Eigen::Success) {
            mSolverHasConverged = false;
            factor              = factor * 1.05;
            double random_e     = dis(gen);
            initial_guess(1.0 + random_e, 1.0 + 0.99 * random_e);
            mBreakdownP = m_InitialGuessBreakdownP;
            // std::cerr << "Inter error\n";
            std::cout << "NO CONVERGENCE OF MCINTYRE DURING COMPUTE, NB EPOCH = " << epoch << std::endl;
            // lambda *= 0.5;
            epoch++;
        } else {
            W           = EigenSolver.solve(B);
            W_old       = W;
            Norm_w      = W.norm() / sqrt(2 * N);
            mBreakdownP = mBreakdownP + lambda * W;
            epoch++;
        }
        // std::cout << "epoch = " << epoch << " Norm_w = " << Norm_w << std::endl;
    }
    if (Norm_w > tolerance && epoch == MaxEpoch) {
        mSolverHasConverged = false;
        std::cerr << "NO CONVERGENCE OF MCINTYRE, NB EPOCH = " << epoch << std::endl;
        mBreakdownP                                   = Eigen::VectorXd::Zero(2 * N);
        m_eBreakdownProbability                       = std::vector<double>(N, 0.0);
        m_hBreakdownProbability                       = std::vector<double>(N, 0.0);
        m_totalBreakdownProbability                   = std::vector<double>(N, 0.0);
        auto                          end             = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        m_McIntyre_time += elapsed_seconds.count();
        m_total_number_sim++;
        return;
    } else {
        mSolverHasConverged = true;
        m_total_number_sim++;
        m_converged_sim++;
        m_InitialGuessBreakdownP = mBreakdownP;
        std::vector<double> eBrPVector(N);
        std::vector<double> hBrPVector(N);
        std::vector<double> BrPVector(N);
        for (std::size_t k = 0; k < N; ++k) {
            eBrPVector[k] = mBreakdownP[2 * k];
            hBrPVector[k] = mBreakdownP[2 * k + 1];
            if ((eBrPVector[k] > 1) || (eBrPVector[k] < 1e-8)) {
                eBrPVector[k] = 0.0;
            }
            if ((hBrPVector[k] > 1) || (hBrPVector[k] < 1e-8)) {
                hBrPVector[k] = 0.0;
            }
            BrPVector[k] = eBrPVector[k] + hBrPVector[k] - eBrPVector[k] * hBrPVector[k];
        }
        m_eBreakdownProbability     = eBrPVector;
        m_hBreakdownProbability     = hBrPVector;
        m_totalBreakdownProbability = BrPVector;
    }
    auto                          end             = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    m_McIntyre_time += elapsed_seconds.count();
}

void McIntyre::ComputeDampedNewtonSolutionIterative(double tolerance) {
    auto            start  = std::chrono::high_resolution_clock::now();
    std::size_t     N      = m_xline.size();
    double          Norm_w = 1e10;
    int             epoch  = 0;
    Eigen::VectorXd W(2 * N);
    Eigen::VectorXd W_new(2 * N);
    mBreakdownP                                       = m_InitialGuessBreakdownP;
    Eigen::SparseMatrix<double>                   MAT = assembleMat();
    Eigen::VectorXd                               B   = assembleSecondMemberNewton();
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double> > solver;
    int                                           MaxEpoch = 500;
    double                                        factor   = 1.0;
    while (Norm_w > tolerance && epoch < MaxEpoch) {
        MAT = assembleMat();
        B   = assembleSecondMemberNewton();
        solver.compute(MAT);
        W             = solver.solve(B);
        Norm_w        = W.norm();
        double lambda = 1.0;
        // double g_0    = (1.0 / 2) * Norm_w * Norm_w;
        mBreakdownP = mBreakdownP + lambda * W;
        epoch++;
    }
    if (Norm_w > tolerance && epoch == MaxEpoch) {
        mSolverHasConverged = false;
        std::cerr << "NO CONVERGENCE OF MCINTYRE, NB EPOCH = " << epoch << std::endl;
        mBreakdownP                                   = Eigen::VectorXd::Zero(2 * N);
        m_eBreakdownProbability                       = std::vector<double>(N, 0.0);
        m_hBreakdownProbability                       = std::vector<double>(N, 0.0);
        m_totalBreakdownProbability                   = std::vector<double>(N, 0.0);
        auto                          end             = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        m_McIntyre_time += elapsed_seconds.count();
        return;
    } else {
        mSolverHasConverged      = true;
        m_InitialGuessBreakdownP = mBreakdownP;
        std::vector<double> eBrPVector(N);
        std::vector<double> hBrPVector(N);
        std::vector<double> BrPVector(N);
        for (std::size_t k = 0; k < N; ++k) {
            eBrPVector[k] = mBreakdownP[2 * k];
            hBrPVector[k] = mBreakdownP[2 * k + 1];
            if ((eBrPVector[k] > 1) || (eBrPVector[k] < 1e-8)) {
                eBrPVector[k] = 0.0;
            }
            if ((hBrPVector[k] > 1) || (hBrPVector[k] < 1e-8)) {
                hBrPVector[k] = 0.0;
            }
            BrPVector[k] = eBrPVector[k] + hBrPVector[k] - eBrPVector[k] * hBrPVector[k];
        }
        m_eBreakdownProbability     = eBrPVector;
        m_hBreakdownProbability     = hBrPVector;
        m_totalBreakdownProbability = BrPVector;
    }
    auto                          end             = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    m_McIntyre_time += elapsed_seconds.count();
}

double McIntyre::get_mean_total_breakdown_probability() const {
    double mean = 0.0;
    for (std::size_t i = 0; i < m_totalBreakdownProbability.size(); ++i) {
        mean += m_totalBreakdownProbability[i];
    }
    mean /= static_cast<double>(m_totalBreakdownProbability.size());
    return mean;
}

}  // namespace mcintyre
