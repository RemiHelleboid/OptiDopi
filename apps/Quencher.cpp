#include <omp.h>  // For OpenMP

#include <algorithm>
#include <cmath>
#include <filesystem>  // For directory creation
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "argparse.hpp"  // Include the argparse library

namespace fs = std::filesystem;  // Alias for filesystem

// Random number generator
std::random_device               rd;
std::mt19937                     gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

// SPAD class with space charge effect
class SPAD {
 private:
    // Physical constants (configurable at runtime)
    double q;        // Electron charge (C)
    double v_e;      // Saturation velocity of electrons (cm/s)
    double v_h;      // Saturation velocity of holes (cm/s)
    double W;        // Width of the multiplication region (cm)
    double V_BD;     // Breakdown voltage (V)
    double V_ex;     // Excess voltage (V)
    double C;        // Capacitance (F)
    double R;        // Quenching resistance (Ohms)
    double alpha_0;  // Impact ionization coefficient for electrons (cm^-1)
    double beta_0;   // Impact ionization coefficient for holes (cm^-1)
    double alpha_p;  // Impact ionization coefficient for electrons (cm^-1)
    double beta_p;   // Impact ionization coefficient for holes (cm^-1)
    double eta;      // Bias parameter for electron generation locations
    double k_sc;     // Space charge proportionality constant (V·cm/C)

    // Simulation parameters (configurable at runtime)
    double dt;         // Time step (s)
    int    num_steps;  // Number of time steps

    // Internal state
    std::vector<double> electrons;  // Positions of electrons in the MR (cm)
    std::vector<double> holes;      // Positions of holes in the MR (cm)
    double              N_e;        // Number of electrons accumulated at the cathode node
    double              V;          // Voltage across the MR (V)
    double              E;          // Electric field in the MR (V/cm)

 public:
    // Constructor with configurable parameters, including k_sc for space charge effect
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
         double eta,
         double dt,
         int    num_steps,
         double k_sc)
        : q(q),
          v_e(v_e),
          v_h(v_h),
          W(W),
          V_BD(V_BD),
          V_ex(V_ex),
          C(C),
          R(R),
          alpha_0(alpha_0),
          beta_0(beta_0),
          alpha_p(alpha_p),
          beta_p(beta_p),
          eta(eta),
          dt(dt),
          num_steps(num_steps),
          k_sc(k_sc),
          electrons({W / 2}),
          holes({W / 2}),
          N_e(0),
          V(V_BD + V_ex),
          E(V / W) {
        // Validate parameters
        if (W <= 0) throw std::invalid_argument("Width of the multiplication region (W) must be positive.");
        if (dt <= 0) throw std::invalid_argument("Time step (dt) must be positive.");
        if (num_steps <= 0) throw std::invalid_argument("Number of time steps (num_steps) must be positive.");
        if (V_ex < 0) throw std::invalid_argument("Excess voltage (V_ex) must be non-negative.");
    }

    // Function to calculate impact ionization rates
    std::pair<double, double> calculate_impact_ionization_rates(double E) const {
        double alpha = alpha_0 * std::exp(-alpha_p / E);  // Electron impact ionization rate
        double beta  = beta_0 * std::exp(-beta_p / E);    // Hole impact ionization rate
        return {alpha, beta};
    }

    // Function to simulate impact ionization
    std::vector<double> simulate_impact_ionization(const std::vector<double>& carriers, double ionization_rate, double velocity) const {
        std::vector<double> new_carriers;
        for (double pos : carriers) {
            if (dis(gen) < ionization_rate * velocity * dt) {
                new_carriers.push_back(pos);  // New carrier generated at the same position
            }
        }
        return new_carriers;
    }

    // Function to update carrier positions
    void update_carrier_positions(std::vector<double>& carriers, double velocity) {
        for (double& pos : carriers) {
            pos += velocity * dt;  // Update position based on velocity
        }
    }

    // Function to remove carriers that have drifted out of the MR
    void remove_drifted_carriers(std::vector<double>& carriers, const std::function<bool(double)>& boundary_condition) {
        carriers.erase(std::remove_if(carriers.begin(), carriers.end(), boundary_condition), carriers.end());
    }

    // Function to simulate one time step with space charge effect
    void simulate_time_step() {
        // Update voltage based on accumulated electrons at the cathode
        V = (V_BD + V_ex) - (q / C) * N_e;

        // Compute the total number of carriers in the MR and the corresponding charge density
        double total_carriers = electrons.size() + holes.size();
        double rho            = q * total_carriers / W;  // Charge density (C/cm)

        // Compute a voltage drop due to space charge
        double deltaV_sc = k_sc * rho;
        std::cout << "Space charge voltage drop: " << deltaV_sc << " V for N_e = " << N_e << "\n";
        double V_eff = V - deltaV_sc;  // Effective voltage across MR after space charge drop
        if (V_eff < 0) V_eff = 0;      // Avoid negative effective voltage

        // Update effective electric field
        E = V_eff / W;  // Effective electric field in the MR

        // Calculate impact ionization rates based on the effective electric field
        auto [alpha, beta] = calculate_impact_ionization_rates(E);

        // Simulate impact ionization for electrons
        auto new_electrons = simulate_impact_ionization(electrons, alpha, v_e);
        auto new_holes     = new_electrons;  // New holes generated at the same positions as electrons
        electrons.insert(electrons.end(), new_electrons.begin(), new_electrons.end());
        holes.insert(holes.end(), new_holes.begin(), new_holes.end());

        // Simulate impact ionization for holes
        auto new_electrons_from_holes = simulate_impact_ionization(holes, beta, v_h);
        auto new_holes_from_holes     = new_electrons_from_holes;  // New holes generated at the same positions as electrons from holes
        electrons.insert(electrons.end(), new_electrons_from_holes.begin(), new_electrons_from_holes.end());
        holes.insert(holes.end(), new_holes_from_holes.begin(), new_holes_from_holes.end());

        // Update positions of electrons and holes
        update_carrier_positions(electrons, v_e);
        update_carrier_positions(holes, -v_h);

        // Remove electrons that have drifted out of the MR
        auto   drifted_electrons = [this](double pos) { return pos >= W; };
        double delta_n_e         = std::count_if(electrons.begin(), electrons.end(), drifted_electrons);
        remove_drifted_carriers(electrons, drifted_electrons);
        N_e += delta_n_e;  // Accumulate electrons at the cathode node

        // Remove holes that have drifted out of the MR
        auto drifted_holes = [](double pos) { return pos < 0; };
        remove_drifted_carriers(holes, drifted_holes);

        // Discharge electrons through the quenching resistor
        N_e -= N_e * dt / (R * C);
    }

    // Function to run the full simulation
    void run_simulation(const std::string& output_file) {
        // Arrays to store results
        std::vector<double> voltage_history(num_steps);
        std::vector<double> electron_count_history(num_steps);
        std::vector<double> electric_field_history(num_steps);
        std::size_t MAX_NB_ELECTRONS = 10e6;  // Maximum number of electrons in the MR
        // Run simulation loop
        for (int i = 0; i < num_steps; ++i) {
            simulate_time_step();
            voltage_history[i]        = V;
            electron_count_history[i] = electrons.size();
            electric_field_history[i] = E;

            if (N_e > MAX_NB_ELECTRONS) {
                std::cout << "Simulation stopped: Maximum number of electrons in MR exceeded the threshold.\n";
                // Resize arrays to the current step
                voltage_history.resize(i + 1);
                electron_count_history.resize(i + 1);
                electric_field_history.resize(i + 1);
                break;
            }
        }

        // Check if the maximum number of electrons exceeds the threshold
        std::size_t max_nb_electrons = *std::max_element(electron_count_history.begin(), electron_count_history.end());
        if (max_nb_electrons < 100) {
            std::cout << "Simulation skipped: Maximum number of electrons in MR (" << max_nb_electrons << ") is below the threshold.\n";
            return;
        }

        // Write results to a CSV file
        std::ofstream out_file(output_file);
        if (!out_file) {
            throw std::runtime_error("Failed to open output file: " + output_file);
        }

        num_steps = voltage_history.size();  // Update the number of steps based on the actual size
        out_file << "time,voltage,electron_count,electric_field\n";
        for (int i = 0; i < num_steps; ++i) {
            out_file << i * dt << "," << voltage_history[i] << "," << electron_count_history[i] << "," << electric_field_history[i];
            out_file << "\n";
        }
        out_file.close();

        std::cout << "Max number of electrons in MR: " << max_nb_electrons << "\n";
    }
};

// Function to create a directory if it doesn't exist
void create_directory(const std::string& dir_name) {
    if (!fs::exists(dir_name)) {
        fs::create_directory(dir_name);
        std::cout << "Created directory: " << dir_name << "\n";
    }
}

int main(int argc, char* argv[]) {
    // Parse command-line arguments
    argparse::ArgumentParser parser("SPAD Simulation");

    // Add arguments for all configurable parameters, including k_sc for space charge effect
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

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << "\n";
        return 1;
    }

    try {
        // Create output directory
        std::string output_dir = parser.get<std::string>("--output_dir");
        create_directory(output_dir);

        // Run multiple simulations
        int num_simulations = parser.get<int>("--num_simulations");
        // #pragma omp parallel for
        for (int i = 0; i < num_simulations; ++i) {
            std::string output_file = output_dir + "/simulation_" + std::to_string(i + 1) + ".csv";

            // Create SPAD object with configurable parameters, including k_sc
            SPAD spad(parser.get<double>("--q"),
                      parser.get<double>("--v_e"),
                      parser.get<double>("--v_h"),
                      parser.get<double>("--W"),
                      parser.get<double>("--V_BD"),
                      parser.get<double>("--V_ex"),
                      parser.get<double>("--C"),
                      parser.get<double>("--R"),
                      parser.get<double>("--alpha_0"),
                      parser.get<double>("--beta_0"),
                      parser.get<double>("--alpha_p"),
                      parser.get<double>("--beta_p"),
                      parser.get<double>("--eta"),
                      parser.get<double>("--dt"),
                      parser.get<int>("--num_steps"),
                      parser.get<double>("--k_sc"));

            // Run simulation
            std::cout << "Starting simulation " << i + 1 << " of " << num_simulations << "...\n";
            spad.run_simulation(output_file);
            std::cout << "Simulation " << i + 1 << " completed. Results saved to '" << output_file << "'.\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
