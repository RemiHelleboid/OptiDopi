#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/xchar.h>
#include <omp.h>  // For OpenMP

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>  // For directory creation
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "argparse.hpp"  // Include the argparse library

namespace fs = std::filesystem;  // Alias for filesystem

// SPAD class with space charge effect
class SPAD {
 private:
    // Physical constants (configurable at runtime)
    double m_q;        // Electron charge (C)
    double m_v_e;      // Saturation velocity of electrons (cm/s)
    double m_v_h;      // Saturation velocity of holes (cm/s)
    double m_W;        // Width of the multiplication region (cm)
    double m_V_BD;     // Breakdown voltage (V)
    double m_V_ex;     // Excess voltage (V)
    double m_C;        // Capacitance (F)
    double m_R;        // Quenching resistance (Ohms)
    double m_alpha_0;  // Impact ionization coefficient for electrons (cm^-1)
    double m_beta_0;   // Impact ionization coefficient for holes (cm^-1)
    double m_alpha_p;  // Impact ionization coefficient for electrons (cm^-1)
    double m_beta_p;   // Impact ionization coefficient for holes (cm^-1)
    double m_eta;      // Bias parameter for electron generation locations
    double m_k_sc;     // Space charge proportionality constant (V·cm/C)

    // Simulation parameters (configurable at runtime)
    double      m_dt;         // Time step (s)
    std::size_t m_num_steps;  // Number of time steps

    // Internal state
    std::vector<double> m_electrons;  // Positions of electrons in the MR (cm)
    std::vector<double> m_holes;      // Positions of holes in the MR (cm)
    double              m_N_e;        // Number of electrons accumulated at the cathode node
    double              m_V;          // Voltage across the MR (V)
    double              m_E;          // Electric field in the MR (V/cm)

    // Random number generation (each instance has its own engine)
    std::mt19937                           m_rng;
    std::uniform_real_distribution<double> m_dist{0.0, 1.0};

    // Results
    bool m_avalanche             = false;
    bool m_succesufull_quenching = false;

 public:
    // Constructor with configurable parameters, including m_k_sc for space charge effect
    SPAD(double q,
         double v_e,
         double v_h,
         double W,
         double V_BD,
         double V_ex,
         double C,
         double R,
         double alpha_0,
         double beta_0,
         double alpha_p,
         double beta_p,
         double eta,  // Parameter name remains unchanged
         double dt,
         int    num_steps,
         double k_sc)
        : m_q(q),
          m_v_e(v_e),
          m_v_h(v_h),
          m_W(W),
          m_V_BD(V_BD),
          m_V_ex(V_ex),
          m_C(C),
          m_R(R),
          m_alpha_0(alpha_0),
          m_beta_0(beta_0),
          m_alpha_p(alpha_p),
          m_beta_p(beta_p),
          m_eta(eta),  // Initialize renamed member variable using parameter "eta"
          m_k_sc(k_sc),
          m_dt(dt),
          m_num_steps(static_cast<std::size_t>(num_steps)),
          m_electrons({W / 2}),
          m_holes({W / 2}),
          m_N_e(0),
          m_V(V_BD + V_ex),
          m_E((V_BD + V_ex) / W),
          m_rng(std::random_device{}()) {
        // Validate parameters
        if (W <= 0) throw std::invalid_argument("Width (W) must be positive.");
        if (dt <= 0) throw std::invalid_argument("Time step (dt) must be positive.");
        if (num_steps <= 0) throw std::invalid_argument("Number of time steps must be positive.");
        if (V_ex < 0) throw std::invalid_argument("Excess voltage (V_ex) must be non-negative.");
    }

    bool hasAvalanche() const { return m_avalanche; }
    bool hasSuccesufullQuenching() const { return m_succesufull_quenching; }

    // Calculate impact ionization rates based on the effective electric field
    std::pair<double, double> calculateImpactIonizationRates(double E) const {
        double alpha = m_alpha_0 * std::exp(-m_alpha_p / E);
        double beta  = m_beta_0 * std::exp(-m_beta_p / E);
        return {alpha, beta};
    }

    // Simulate impact ionization for a given carrier type.
    // (New carriers are generated at the same positions as their parent carriers.)
    std::vector<double> simulateImpactIonization(const std::vector<double>& carriers, double ionizationRate, double velocity) {
        std::vector<double> newCarriers;
        for (const auto pos : carriers) {
            if (m_dist(m_rng) < ionizationRate * velocity * m_dt) {
                newCarriers.push_back(pos);
            }
        }
        return newCarriers;
    }

    // Update carrier positions based on their velocity.
    void updateCarrierPositions(std::vector<double>& carriers, double velocity) {
        for (auto& pos : carriers) {
            pos += velocity * m_dt;
        }
    }

    // Remove carriers that have drifted outside the multiplication region.
    void removeDriftedCarriers(std::vector<double>& carriers, const std::function<bool(double)>& isOutOfBounds) {
        carriers.erase(std::remove_if(carriers.begin(), carriers.end(), isOutOfBounds), carriers.end());
    }

    // Simulate one time step with space charge effect.
    void simulateTimeStep() {
        // Update voltage based on accumulated electrons at the cathode
        m_V = (m_V_BD + m_V_ex) - (m_q / m_C) * m_N_e;

        // Compute charge density in the MR and the voltage drop due to space charge
        const double totalCarriers = m_electrons.size() + m_holes.size();
        const double rho           = m_q * totalCarriers / m_W;  // Charge density (C/cm)
        const double deltaV_sc     = m_k_sc * rho;
        const double V_eff         = std::max(m_V - deltaV_sc, 0.0);
        m_E                        = V_eff / m_W;

        // Get impact ionization rates based on the effective electric field
        const auto [alpha, beta] = calculateImpactIonizationRates(m_E);

        // Simulate impact ionization for electrons
        const auto newElectrons = simulateImpactIonization(m_electrons, alpha, m_v_e);
        // New holes are generated at the same positions as the newly created electrons
        const auto newHolesFromElectrons = newElectrons;
        m_electrons.insert(m_electrons.end(), newElectrons.begin(), newElectrons.end());
        m_holes.insert(m_holes.end(), newHolesFromElectrons.begin(), newHolesFromElectrons.end());

        // Simulate impact ionization for holes
        const auto newElectronsFromHoles = simulateImpactIonization(m_holes, beta, m_v_h);
        const auto newHolesFromHoles     = newElectronsFromHoles;
        m_electrons.insert(m_electrons.end(), newElectronsFromHoles.begin(), newElectronsFromHoles.end());
        m_holes.insert(m_holes.end(), newHolesFromHoles.begin(), newHolesFromHoles.end());

        // Update positions of electrons and holes
        updateCarrierPositions(m_electrons, m_v_e);
        updateCarrierPositions(m_holes, -m_v_h);

        // Remove electrons that have drifted out of the MR
        const auto   isDriftedElectron    = [this](double pos) { return pos >= m_W; };
        const double driftedElectronCount = std::count_if(m_electrons.begin(), m_electrons.end(), isDriftedElectron);
        removeDriftedCarriers(m_electrons, isDriftedElectron);
        m_N_e += driftedElectronCount;  // Accumulate electrons at the cathode

        // Remove holes that have drifted out of the MR
        const auto isDriftedHole = [](double pos) { return pos < 0; };
        removeDriftedCarriers(m_holes, isDriftedHole);

        // Discharge electrons through the quenching resistor
        m_N_e -= m_N_e * m_dt / (m_R * m_C);
    }

    // Run the full simulation and write results to a CSV file.
    void runSimulation(const std::string& outputFile) {
        std::vector<double> voltageHistory(m_num_steps);
        std::vector<double> electronCountHistory(m_num_steps);
        std::vector<double> electricFieldHistory(m_num_steps);
        const std::size_t   MAX_NB_ELECTRONS = 10e6;  // Threshold for stopping simulation

        double       Vbias               = m_V_BD + m_V_ex;
        const double RATIO_QUECH_SUCCESS = 0.999;

        for (std::size_t step = 0; step < m_num_steps; ++step) {
            simulateTimeStep();
            voltageHistory[step]       = m_V;
            electronCountHistory[step] = m_electrons.size();
            electricFieldHistory[step] = m_E;

            if (m_N_e > MAX_NB_ELECTRONS) {
                fmt::print("Simulation stopped: Maximum number of electrons in MR exceeded the threshold.\n");
                voltageHistory.resize(step + 1);
                electronCountHistory.resize(step + 1);
                electricFieldHistory.resize(step + 1);
                break;
            }
            // Check for avalanche
            if (!m_avalanche && (Vbias - m_V) >= 0.95 * m_V_ex) {
                m_avalanche = true;
            }
            if (!m_avalanche && step >= static_cast<std::size_t>(m_num_steps * 0.1)) {
                voltageHistory.resize(step + 1);
                electronCountHistory.resize(step + 1);
                electricFieldHistory.resize(step + 1);
                break;
            }
            // Check for quenching success (voltage back to V_BD)
            if (m_avalanche && m_V >= RATIO_QUECH_SUCCESS * Vbias) {
                m_succesufull_quenching = true;
                voltageHistory.resize(step + 1);
                electronCountHistory.resize(step + 1);
                electricFieldHistory.resize(step + 1);
                break;
            }
        }

        // Write results to a CSV file
        std::ofstream outFile(outputFile);
        if (!outFile) {
            throw std::runtime_error("Failed to open output file: " + outputFile);
        }

        outFile << "time,voltage,electron_count,electric_field\n";
        for (std::size_t i = 0; i < voltageHistory.size(); ++i) {
            outFile << i * m_dt << "," << voltageHistory[i] << "," << electronCountHistory[i] << "," << electricFieldHistory[i] << "\n";
        }
        outFile.close();

        // fmt::print("Max number of electrons in MR: {}\n", maxElectronCount);
    }
};

// Function to create a directory if it doesn't exist
void createDirectory(const std::string& dirName) {
    if (!fs::exists(dirName)) {
        fs::create_directory(dirName);
        fmt::print("Created directory: {}\n", dirName);
    }
}

int main(int argc, char* argv[]) {
    // Parse command-line arguments
    argparse::ArgumentParser parser("SPAD Simulation");

    // Add arguments for all configurable parameters, including m_k_sc for space charge effect
    parser.add_argument("--q").default_value(1.6e-19).help("Electron charge (C)").scan<'g', double>();
    parser.add_argument("--v_e").default_value(1.02e7).help("Saturation velocity of electrons (cm/s)").scan<'g', double>();
    parser.add_argument("--v_h").default_value(8.31e6).help("Saturation velocity of holes (cm/s)").scan<'g', double>();
    parser.add_argument("--W").default_value(0.8e-4).help("Width of the multiplication region (cm)").scan<'g', double>();
    parser.add_argument("--V_BD").default_value(29.55).help("Breakdown voltage (V)").scan<'g', double>();
    parser.add_argument("--V_ex").default_value(0.45).help("Excess voltage (V)").scan<'g', double>();
    parser.add_argument("--C").default_value(30e-15).help("Capacitance (F)").scan<'g', double>();
    parser.add_argument("--R").default_value(12e3).help("Quenching resistance (Ohms)").scan<'g', double>();
    parser.add_argument("--alpha_0").default_value(3.80e6).help("Impact ionization coefficient for electrons (cm^-1)").scan<'g', double>();
    parser.add_argument("--beta_0").default_value(2.25e7).help("Impact ionization coefficient for holes (cm^-1)").scan<'g', double>();
    parser.add_argument("--alpha_p").default_value(1.75e6).help("Impact ionization coefficient for electrons (cm^-1)").scan<'g', double>();
    parser.add_argument("--beta_p").default_value(3.26e6).help("Impact ionization coefficient for holes (cm^-1)").scan<'g', double>();
    parser.add_argument("--eta").default_value(3.0).help("Bias parameter for electron generation locations").scan<'g', double>();
    parser.add_argument("--dt").default_value(0.2e-12).help("Time step (s)").scan<'g', double>();
    parser.add_argument("--num_steps").default_value(50000).help("Number of time steps").scan<'i', int>();
    parser.add_argument("--output_dir").default_value("simulation_results").help("Output directory name");
    parser.add_argument("--num_simulations").default_value(1).help("Number of simulations to run").scan<'i', int>();
    parser.add_argument("--k_sc").default_value(5e11).help("Space charge proportionality constant (V·cm/C)").scan<'g', double>();
    // Add argument to plot or not the results (plot if the argument is present)
    parser.add_argument("-p", "--plot").default_value(false).implicit_value(true).help("Plot the results using Python");

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << "\n";
        return 1;
    }

    try {
        // Extract simulation parameters from command line
        const std::string baseOutputDir = parser.get<std::string>("--output_dir");

        const double q         = parser.get<double>("--q");
        const double v_e       = parser.get<double>("--v_e");
        const double v_h       = parser.get<double>("--v_h");
        const double W         = parser.get<double>("--W");
        const double V_BD      = parser.get<double>("--V_BD");
        const double V_ex      = parser.get<double>("--V_ex");
        const double C         = parser.get<double>("--C");
        const double R         = parser.get<double>("--R");
        const double alpha_0   = parser.get<double>("--alpha_0");
        const double beta_0    = parser.get<double>("--beta_0");
        const double alpha_p   = parser.get<double>("--alpha_p");
        const double beta_p    = parser.get<double>("--beta_p");
        const double eta       = parser.get<double>("--eta");
        const double dt        = parser.get<double>("--dt");
        const int    num_steps = parser.get<int>("--num_steps");
        double       k_sc      = parser.get<double>("--k_sc");
        k_sc                   = 0.0;

        // Format the output directory name using fmt::format for clarity.
        std::string outputDir = fmt::format("{}_C_{:.2e}_R_{:.2e}_Vex_{:.2f}_VB_{:.2f}_W_{:.2e}", baseOutputDir, C, R, V_ex, V_BD, W);
        createDirectory(outputDir);

        // Run simulations (can be parallelized with OpenMP if desired)
        const int numSimulations = parser.get<int>("--num_simulations");

        int              nb_avalanche             = 0;
        int              nb_succesufull_quenching = 0;
        std::atomic<int> progressCounter{0};

#pragma omp parallel for num_threads(8) reduction(+ : nb_avalanche) reduction(+ : nb_succesufull_quenching)
        for (int i = 0; i < numSimulations; ++i) {
            const std::string outputFile = outputDir + "/simulation_" + std::to_string(i + 1) + ".csv";
            SPAD              spad(q, v_e, v_h, W, V_BD, V_ex, C, R, alpha_0, beta_0, alpha_p, beta_p, eta, dt, num_steps, k_sc);
            spad.runSimulation(outputFile);
            nb_avalanche += spad.hasAvalanche();

            if (spad.hasAvalanche()) {
                nb_succesufull_quenching += spad.hasSuccesufullQuenching();
            }
            // Atomically update the progress counter
            int currentProgress = progressCounter.fetch_add(1) + 1;

            // Print progress every 10 iterations or at the end
            if (currentProgress % 10 == 0 || currentProgress == numSimulations) {
#pragma omp critical
                {
                    fmt::print("\rProgress: {}/{} simulations completed.", currentProgress, numSimulations);
                    std::cout << std::flush;
                }
            }
        }

        double proba_avalanche = static_cast<double>(nb_avalanche) / numSimulations;
        double proba_quenching = static_cast<double>(nb_succesufull_quenching) / nb_avalanche;

        std::cout << "\n\n";

        double RC_ns = R * C * 1e9;  // Time constant in ns
        fmt::print("C = {:.2e}, R = {:.2e}, RC = {:.2f} ns, V_bias = {} V, W = {:.2e} cm\n", C, R, RC_ns, V_BD + V_ex, W);
        fmt::print("Results saved in directory: {}\n", outputDir);

        fmt::print("Probability of avalanche: {:.2f}\n", proba_avalanche);
        fmt::print("Probability of succesufull quenching: {:.2f} ({}/{})\n", proba_quenching, nb_succesufull_quenching, nb_avalanche);

        // Save global results to a file
        std::string   globalResultsFile = outputDir + "/global_results.txt";
        std::ofstream outFile(globalResultsFile);
        if (!outFile) {
            throw std::runtime_error("Failed to open output file: " + globalResultsFile);
        }
        // outFile << "C (F) = " << C << "\n";
        // outFile << "R (Ohms) = " << R << "\n";
        // outFile << "RC (ns) = " << RC_ns << "\n";
        // outFile << "V_bias (V) = " << V_BD + V_ex << "\n";
        // outFile << "W (cm) = " << W << "\n";
        // outFile << "Probability of avalanche = " << proba_avalanche << "\n";
        // outFile << "Probability of succesufull quenching = " << proba_quenching << "\n";
        // outFile.close();
        std::string globalResults = fmt::format(
            "C (F) = {:.5e}\nR (Ohms) = {:.5e}\nRC (ns) = {:.2f}\nV_bias (V) = {}\nW (cm) = {:.2e}\nProbability of avalanche = {:.2f}\nProbability "
            "of succesufull quenching = {:.2f}\n",
            C,
            R,
            RC_ns,
            V_BD + V_ex,
            W,
            proba_avalanche,
            proba_quenching);
        outFile << globalResults;
        outFile.close();

        std::cout << "\n -------------------------------- \n";

        if (parser["--plot"] == true) {
            // Run python script to plot the results
            std::string pythonScriptFile = std::string(CMAKE_SOURCE_DIR) + "/python/plot_quencher.py";
            std::string command          = fmt::format("python {} {}", pythonScriptFile, outputDir);
            fmt::print("Running command: {}\n", command);
            int status = std::system(command.c_str());
            if (status != 0) {
                fmt::print("Failed to run the Python script.\n");
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
