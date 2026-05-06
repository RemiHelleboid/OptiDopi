#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/xchar.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "argparse.hpp"

namespace fs = std::filesystem;

#pragma omp declare reduction(merge_double_vector : std::vector<double> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

struct SimulationParameters {
    double      q                            = 1.6e-19;
    double      electron_velocity            = 1.02e7;
    double      hole_velocity                = 8.31e6;
    double      width                        = 0.8e-4;
    double      breakdown_voltage            = 29.55;
    double      excess_voltage               = 0.45;
    double      capacitance                  = 30e-15;
    double      resistance                   = 12e3;
    double      alpha_0                      = 3.80e6;
    double      beta_0                       = 2.25e7;
    double      alpha_p                      = 1.75e6;
    double      beta_p                       = 3.26e6;
    double      eta                          = 3.0;
    double      dt                           = 0.1e-12;
    double      space_charge_coefficient     = 5e11;
    std::size_t num_steps                    = 150000;
    std::size_t avalanche_carrier_threshold  = 1000;
    std::size_t extinction_carrier_threshold = 0;
    std::size_t max_carriers                 = 10'000'000;
    double      recharge_ratio               = 0.999;
};

struct SimulationResult {
    bool   avalanche                = false;
    bool   quenched                 = false;
    bool   recharged                = false;
    bool   stopped_by_carrier_limit = false;
    double avalanche_time           = std::numeric_limits<double>::quiet_NaN();
    double quench_time              = std::numeric_limits<double>::quiet_NaN();
    double recharge_time            = std::numeric_limits<double>::quiet_NaN();
};

class SPAD {
 public:
    explicit SPAD(SimulationParameters parameters, std::uint32_t seed = std::random_device{}())
        : m_parameters(parameters),
          m_bias_voltage(parameters.breakdown_voltage + parameters.excess_voltage),
          m_voltage(m_bias_voltage),
          m_electric_field(m_bias_voltage / parameters.width),
          m_rng(seed) {
        validateParameters();
        const double initial_position = sampleInitialPosition();
        m_electrons.push_back(initial_position);
        m_holes.push_back(initial_position);
    }

    const SimulationResult& result() const { return m_result; }

    void run(const std::string& output_file) {
        std::vector<double> voltage_history;
        std::vector<double> electron_history;
        std::vector<double> hole_history;
        std::vector<double> electric_field_history;
        std::vector<double> avalanche_current_history;

        voltage_history.reserve(m_parameters.num_steps);
        electron_history.reserve(m_parameters.num_steps);
        hole_history.reserve(m_parameters.num_steps);
        electric_field_history.reserve(m_parameters.num_steps);
        avalanche_current_history.reserve(m_parameters.num_steps);

        for (std::size_t step = 0; step < m_parameters.num_steps; ++step) {
            const StepResult step_result = simulateStep();
            const double     time        = static_cast<double>(step) * m_parameters.dt;

            voltage_history.push_back(m_voltage);
            electron_history.push_back(static_cast<double>(m_electrons.size()));
            hole_history.push_back(static_cast<double>(m_holes.size()));
            electric_field_history.push_back(m_electric_field);
            avalanche_current_history.push_back(step_result.avalanche_current);

            updateEventState(step, time);

            if (m_electrons.size() + m_holes.size() > m_parameters.max_carriers) {
                m_result.stopped_by_carrier_limit = true;
                break;
            }

            if (m_result.recharged) {
                break;
            }

            if (!m_result.avalanche && step >= static_cast<std::size_t>(0.3 * static_cast<double>(m_parameters.num_steps))) {
                break;
            }
        }

        prependInitialHistory(voltage_history, electron_history, hole_history, electric_field_history, avalanche_current_history);
        writeCsv(output_file, voltage_history, electron_history, hole_history, electric_field_history, avalanche_current_history);
    }

 private:
    struct StepResult {
        std::size_t electrons_collected = 0;
        double      avalanche_current   = 0.0;
    };

    SimulationParameters                   m_parameters;
    double                                 m_bias_voltage   = 0.0;
    double                                 m_voltage        = 0.0;
    double                                 m_electric_field = 0.0;
    std::vector<double>                    m_electrons;
    std::vector<double>                    m_holes;
    std::mt19937                           m_rng;
    std::uniform_real_distribution<double> m_uniform{0.0, 1.0};
    SimulationResult                       m_result;

    void validateParameters() const {
        if (m_parameters.q <= 0.0) {
            throw std::invalid_argument("Electron charge must be positive.");
        }
        if (m_parameters.electron_velocity <= 0.0) {
            throw std::invalid_argument("Electron velocity must be positive.");
        }
        if (m_parameters.hole_velocity <= 0.0) {
            throw std::invalid_argument("Hole velocity must be positive.");
        }
        if (m_parameters.width <= 0.0) {
            throw std::invalid_argument("Multiplication-region width must be positive.");
        }
        if (m_parameters.breakdown_voltage <= 0.0) {
            throw std::invalid_argument("Breakdown voltage must be positive.");
        }
        if (m_parameters.excess_voltage < 0.0) {
            throw std::invalid_argument("Excess voltage must be non-negative.");
        }
        if (m_parameters.capacitance <= 0.0) {
            throw std::invalid_argument("Capacitance must be positive.");
        }
        if (m_parameters.resistance <= 0.0) {
            throw std::invalid_argument("Resistance must be positive.");
        }
        if (m_parameters.dt <= 0.0) {
            throw std::invalid_argument("Time step must be positive.");
        }
        if (m_parameters.num_steps == 0) {
            throw std::invalid_argument("Number of time steps must be positive.");
        }
        if (m_parameters.max_carriers == 0) {
            throw std::invalid_argument("Maximum carrier count must be positive.");
        }
        if (m_parameters.recharge_ratio <= 0.0 || m_parameters.recharge_ratio > 1.0) {
            throw std::invalid_argument("Recharge ratio must be in the interval (0, 1].");
        }
    }

    double sampleInitialPosition() {
        const double u   = std::clamp(m_uniform(m_rng), 1e-12, 1.0 - 1e-12);
        const double eta = m_parameters.eta;

        if (std::abs(eta) < 1e-12) {
            return m_parameters.width * u;
        }

        const double denominator = std::expm1(eta);
        return m_parameters.width * std::log1p(u * denominator) / eta;
    }

    std::pair<double, double> impactIonizationCoefficients(double electric_field) const {
        if (electric_field <= 0.0) {
            return {0.0, 0.0};
        }

        const double alpha = m_parameters.alpha_0 * std::exp(-m_parameters.alpha_p / electric_field);
        const double beta  = m_parameters.beta_0 * std::exp(-m_parameters.beta_p / electric_field);
        return {alpha, beta};
    }

    std::vector<double> generateImpactPairs(const std::vector<double>& carriers, double ionization_coefficient, double velocity) {
        std::vector<double> generated_positions;

        if (carriers.empty() || ionization_coefficient <= 0.0 || velocity <= 0.0) {
            return generated_positions;
        }

        const double probability         = 1.0 - std::exp(-ionization_coefficient * velocity * m_parameters.dt);
        const double bounded_probability = std::clamp(probability, 0.0, 1.0);

        generated_positions.reserve(static_cast<std::size_t>(bounded_probability * static_cast<double>(carriers.size())) + 1);

        for (const double position : carriers) {
            if (m_uniform(m_rng) < bounded_probability) {
                generated_positions.push_back(position);
            }
        }

        return generated_positions;
    }

    void appendGeneratedPairs(const std::vector<double>& positions) {
        m_electrons.insert(m_electrons.end(), positions.begin(), positions.end());
        m_holes.insert(m_holes.end(), positions.begin(), positions.end());
    }

    void driftCarriers() {
        const double electron_displacement = m_parameters.electron_velocity * m_parameters.dt;
        const double hole_displacement     = m_parameters.hole_velocity * m_parameters.dt;

        for (double& position : m_electrons) {
            position += electron_displacement;
        }

        for (double& position : m_holes) {
            position -= hole_displacement;
        }
    }

    std::size_t removeCollectedElectrons() {
        const auto first_collected =
            std::partition(m_electrons.begin(), m_electrons.end(), [this](double position) { return position < m_parameters.width; });

        const std::size_t collected = static_cast<std::size_t>(std::distance(first_collected, m_electrons.end()));
        m_electrons.erase(first_collected, m_electrons.end());
        return collected;
    }

    void removeCollectedHoles() {
        const auto first_collected = std::partition(m_holes.begin(), m_holes.end(), [](double position) { return position >= 0.0; });

        m_holes.erase(first_collected, m_holes.end());
    }

    double effectiveVoltage() const {
        const double total_carriers      = static_cast<double>(m_electrons.size() + m_holes.size());
        const double line_charge_density = m_parameters.q * total_carriers / m_parameters.width;
        const double space_charge_drop   = m_parameters.space_charge_coefficient * line_charge_density;
        return std::max(m_voltage - space_charge_drop, 0.0);
    }

    void updateElectricField() { m_electric_field = effectiveVoltage() / m_parameters.width; }

    void updateCircuit(double avalanche_current) {
        const double resistor_current = (m_bias_voltage - m_voltage) / m_parameters.resistance;
        const double d_voltage        = (resistor_current - avalanche_current) * m_parameters.dt / m_parameters.capacitance;
        m_voltage                     = std::clamp(m_voltage + d_voltage, 0.0, m_bias_voltage);
    }

    StepResult simulateStep() {
        updateElectricField();

        const auto [alpha, beta] = impactIonizationCoefficients(m_electric_field);

        const std::vector<double> electrons_before_ionization = m_electrons;
        const std::vector<double> holes_before_ionization     = m_holes;

        const std::vector<double> pairs_from_electrons =
            generateImpactPairs(electrons_before_ionization, alpha, m_parameters.electron_velocity);
        const std::vector<double> pairs_from_holes = generateImpactPairs(holes_before_ionization, beta, m_parameters.hole_velocity);

        appendGeneratedPairs(pairs_from_electrons);
        appendGeneratedPairs(pairs_from_holes);

        driftCarriers();

        const std::size_t collected_electrons = removeCollectedElectrons();
        removeCollectedHoles();

        const double avalanche_current = m_parameters.q * static_cast<double>(collected_electrons) / m_parameters.dt;
        updateCircuit(avalanche_current);
        updateElectricField();

        return StepResult{collected_electrons, avalanche_current};
    }

    void updateEventState(std::size_t step, double time) {
        const std::size_t total_carriers = m_electrons.size() + m_holes.size();

        if (!m_result.avalanche && total_carriers >= m_parameters.avalanche_carrier_threshold) {
            m_result.avalanche      = true;
            m_result.avalanche_time = time;
        }

        if (m_result.avalanche && !m_result.quenched && 
            total_carriers <= m_parameters.extinction_carrier_threshold) {
            m_result.quenched    = true;
            m_result.quench_time = time;
        }

    }

    void prependInitialHistory(std::vector<double>& voltage_history,
                               std::vector<double>& electron_history,
                               std::vector<double>& hole_history,
                               std::vector<double>& electric_field_history,
                               std::vector<double>& avalanche_current_history) const {
        constexpr std::size_t initial_steps = 1000;

        const double initial_voltage = voltage_history.empty() ? m_bias_voltage : voltage_history.front();
        const double initial_field   = initial_voltage / m_parameters.width;

        voltage_history.insert(voltage_history.begin(), initial_steps, initial_voltage);
        electron_history.insert(electron_history.begin(), initial_steps, 0.0);
        hole_history.insert(hole_history.begin(), initial_steps, 0.0);
        electric_field_history.insert(electric_field_history.begin(), initial_steps, initial_field);
        avalanche_current_history.insert(avalanche_current_history.begin(), initial_steps, 0.0);
    }

    void writeCsv(const std::string&         output_file,
                  const std::vector<double>& voltage_history,
                  const std::vector<double>& electron_history,
                  const std::vector<double>& hole_history,
                  const std::vector<double>& electric_field_history,
                  const std::vector<double>& avalanche_current_history) const {
        std::ofstream file(output_file);
        if (!file) {
            throw std::runtime_error("Failed to open output file: " + output_file);
        }

        file << "time,voltage,electron_count,hole_count,electric_field,avalanche_current\n";

        for (std::size_t i = 0; i < voltage_history.size(); ++i) {
            const double time = static_cast<double>(i) * m_parameters.dt;
            file << time << ',' << voltage_history[i] << ',' << electron_history[i] << ',' << hole_history[i] << ','
                 << electric_field_history[i] << ',' << avalanche_current_history[i] << '\n';
        }
    }
};

void createDirectory(const std::string& directory) { fs::create_directories(directory); }

SimulationParameters parseSimulationParameters(argparse::ArgumentParser& parser) {
    SimulationParameters parameters;

    parameters.q                        = parser.get<double>("--q");
    parameters.electron_velocity        = parser.get<double>("--v_e");
    parameters.hole_velocity            = parser.get<double>("--v_h");
    parameters.width                    = parser.get<double>("--W");
    parameters.breakdown_voltage        = parser.get<double>("--V_BD");
    parameters.excess_voltage           = parser.get<double>("--V_ex");
    parameters.capacitance              = parser.get<double>("--C");
    parameters.resistance               = parser.get<double>("--R");
    parameters.alpha_0                  = parser.get<double>("--alpha_0");
    parameters.beta_0                   = parser.get<double>("--beta_0");
    parameters.alpha_p                  = parser.get<double>("--alpha_p");
    parameters.beta_p                   = parser.get<double>("--beta_p");
    parameters.eta                      = parser.get<double>("--eta");
    parameters.dt                       = parser.get<double>("--dt");
    parameters.space_charge_coefficient = parser.get<double>("--k_sc");

    const int num_steps           = parser.get<int>("--num_steps");
    const int avalanche_threshold = parser.get<int>("--avalanche_threshold");
    const int max_carriers        = parser.get<int>("--max_carriers");

    if (num_steps <= 0) {
        throw std::invalid_argument("--num_steps must be positive.");
    }
    if (avalanche_threshold <= 0) {
        throw std::invalid_argument("--avalanche_threshold must be positive.");
    }
    if (max_carriers <= 0) {
        throw std::invalid_argument("--max_carriers must be positive.");
    }

    parameters.num_steps                   = static_cast<std::size_t>(num_steps);
    parameters.avalanche_carrier_threshold = static_cast<std::size_t>(avalanche_threshold);
    parameters.max_carriers                = static_cast<std::size_t>(max_carriers);

    return parameters;
}

void addArguments(argparse::ArgumentParser& parser) {
    parser.add_argument("--q").default_value(1.6e-19).help("Electron charge (C)").scan<'g', double>();
    parser.add_argument("--v_e").default_value(1.02e7).help("Electron saturation velocity (cm/s)").scan<'g', double>();
    parser.add_argument("--v_h").default_value(8.31e6).help("Hole saturation velocity (cm/s)").scan<'g', double>();
    parser.add_argument("--W").default_value(0.8e-4).help("Multiplication-region width (cm)").scan<'g', double>();
    parser.add_argument("--V_BD").default_value(29.55).help("Breakdown voltage (V)").scan<'g', double>();
    parser.add_argument("--V_ex").default_value(0.45).help("Excess voltage (V)").scan<'g', double>();
    parser.add_argument("--C").default_value(30e-15).help("Capacitance (F)").scan<'g', double>();
    parser.add_argument("--R").default_value(12e3).help("Quenching resistance (Ohms)").scan<'g', double>();
    parser.add_argument("--alpha_0").default_value(3.80e6).help("Electron ionization prefactor (cm^-1)").scan<'g', double>();
    parser.add_argument("--beta_0").default_value(2.25e7).help("Hole ionization prefactor (cm^-1)").scan<'g', double>();
    parser.add_argument("--alpha_p").default_value(1.75e6).help("Electron ionization critical field parameter (V/cm)").scan<'g', double>();
    parser.add_argument("--beta_p").default_value(3.26e6).help("Hole ionization critical field parameter (V/cm)").scan<'g', double>();
    parser.add_argument("--eta").default_value(3.0).help("Initial generation-position bias parameter").scan<'g', double>();
    parser.add_argument("--dt").default_value(0.1e-12).help("Time step (s)").scan<'g', double>();
    parser.add_argument("--num_steps").default_value(150000).help("Number of time steps").scan<'i', int>();
    parser.add_argument("--j").default_value(1).help("Number of OpenMP threads").scan<'i', int>();
    parser.add_argument("--output_dir").default_value(std::string("simulation_results")).help("Output directory base name");
    parser.add_argument("--num_simulations").default_value(1).help("Number of Monte Carlo simulations").scan<'i', int>();
    parser.add_argument("--k_sc").default_value(5e11).help("Empirical space-charge coefficient (V cm/C)").scan<'g', double>();
    parser.add_argument("--avalanche_threshold")
        .default_value(1000)
        .help("Carrier-count threshold used to detect avalanche")
        .scan<'i', int>();
    parser.add_argument("--max_carriers")
        .default_value(10000000)
        .help("Carrier-count limit used to stop runaway simulations")
        .scan<'i', int>();
    parser.add_argument("-p", "--plot").default_value(false).implicit_value(true).help("Plot results with the Python script");
}

void writeGlobalResults(const std::string&          output_file,
                        const SimulationParameters& parameters,
                        int                         num_simulations,
                        int                         avalanche_count,
                        int                         quench_count,
                        int                         recharge_count,
                        int                         carrier_limit_count,
                        const std::vector<double>&  avalanche_times,
                        const std::vector<double>&  quench_times,
                        const std::vector<double>&  recharge_times) {
    std::ofstream file(output_file);
    if (!file) {
        throw std::runtime_error("Failed to open output file: " + output_file);
    }

    const double avalanche_probability =
        num_simulations > 0 ? static_cast<double>(avalanche_count) / static_cast<double>(num_simulations) : 0.0;
    const double quench_probability = avalanche_count > 0 ? static_cast<double>(quench_count) / static_cast<double>(avalanche_count) : 0.0;
    const double recharge_probability = quench_count > 0 ? static_cast<double>(recharge_count) / static_cast<double>(quench_count) : 0.0;

    file << fmt::format("C (F) = {:.5e}\n", parameters.capacitance);
    file << fmt::format("R (Ohms) = {:.5e}\n", parameters.resistance);
    file << fmt::format("RC (ns) = {:.5f}\n", parameters.resistance * parameters.capacitance * 1e9);
    file << fmt::format("V_bias (V) = {:.8g}\n", parameters.breakdown_voltage + parameters.excess_voltage);
    file << fmt::format("V_BD (V) = {:.8g}\n", parameters.breakdown_voltage);
    file << fmt::format("V_ex (V) = {:.8g}\n", parameters.excess_voltage);
    file << fmt::format("W (cm) = {:.5e}\n", parameters.width);
    file << fmt::format("dt (s) = {:.5e}\n", parameters.dt);
    file << fmt::format("num_simulations = {}\n", num_simulations);
    file << fmt::format("avalanche_count = {}\n", avalanche_count);
    file << fmt::format("quench_count = {}\n", quench_count);
    file << fmt::format("recharge_count = {}\n", recharge_count);
    file << fmt::format("carrier_limit_count = {}\n", carrier_limit_count);
    file << fmt::format("probability_avalanche = {:.8f}\n", avalanche_probability);
    file << fmt::format("probability_quench_given_avalanche = {:.8f}\n", quench_probability);
    file << fmt::format("probability_recharge_given_quench = {:.8f}\n", recharge_probability);

    file << "\navalanche_times_s\n";
    for (const double time : avalanche_times) {
        fmt::print(file, "{:.6e}\n", time);
    }

    file << "\nquench_times_s\n";
    for (const double time : quench_times) {
        fmt::print(file, "{:.6e}\n", time);
    }

    file << "\nrecharge_times_s\n";
    for (const double time : recharge_times) {
        fmt::print(file, "{:.6e}\n", time);
    }
}

int main(int argc, char* argv[]) {
    argparse::ArgumentParser parser("SPAD Simulation");
    addArguments(parser);

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& exception) {
        std::cerr << "Error parsing arguments: " << exception.what() << '\n';
        return 1;
    }

    try {
        const SimulationParameters parameters      = parseSimulationParameters(parser);
        const std::string          base_output_dir = parser.get<std::string>("--output_dir");
        const int                  num_simulations = parser.get<int>("--num_simulations");
        const int                  num_threads     = parser.get<int>("--j");

        if (num_simulations <= 0) {
            throw std::invalid_argument("--num_simulations must be positive.");
        }
        if (num_threads <= 0) {
            throw std::invalid_argument("--j must be positive.");
        }

        const std::string output_dir = fmt::format("{}_C_{:.2e}_R_{:.2e}_Vex_{:.2f}_VB_{:.2f}_W_{:.2e}",
                                                   base_output_dir,
                                                   parameters.capacitance,
                                                   parameters.resistance,
                                                   parameters.excess_voltage,
                                                   parameters.breakdown_voltage,
                                                   parameters.width);

        createDirectory(output_dir);

        int avalanche_count     = 0;
        int quench_count        = 0;
        int recharge_count      = 0;
        int carrier_limit_count = 0;

        std::vector<double> avalanche_times;
        std::vector<double> quench_times;
        std::vector<double> recharge_times;

        std::atomic<int> progress_counter{0};

#pragma omp parallel for num_threads(num_threads) reduction(+ : avalanche_count) reduction(+ : quench_count) reduction(+ : recharge_count) \
    reduction(+ : carrier_limit_count) reduction(merge_double_vector : avalanche_times) reduction(merge_double_vector : quench_times)      \
    reduction(merge_double_vector : recharge_times)
        for (int i = 0; i < num_simulations; ++i) {
            std::random_device random_device;
            std::seed_seq      seed_sequence{random_device(),
                                             random_device(),
                                             static_cast<std::uint32_t>(i),
                                             static_cast<std::uint32_t>(omp_get_thread_num())};

            std::vector<std::uint32_t> seed_data(1);
            seed_sequence.generate(seed_data.begin(), seed_data.end());

            SPAD              spad(parameters, seed_data.front());
            const std::string output_file = output_dir + "/simulation_" + std::to_string(i + 1) + ".csv";
            spad.run(output_file);

            const SimulationResult& result = spad.result();

            avalanche_count += result.avalanche ? 1 : 0;
            quench_count += result.quenched ? 1 : 0;
            recharge_count += result.recharged ? 1 : 0;
            carrier_limit_count += result.stopped_by_carrier_limit ? 1 : 0;

            if (result.avalanche) {
                avalanche_times.push_back(result.avalanche_time);
            }
            if (result.quenched) {
                quench_times.push_back(result.quench_time);
            }
            if (result.recharged) {
                recharge_times.push_back(result.recharge_time);
            }

            const int current_progress = progress_counter.fetch_add(1) + 1;
            if (current_progress % 10 == 0 || current_progress == num_simulations) {
#pragma omp critical
                {
                    fmt::print("\rProgress: {}/{} simulations completed.", current_progress, num_simulations);
                    std::cout << std::flush;
                }
            }
        }

        const double avalanche_probability = static_cast<double>(avalanche_count) / static_cast<double>(num_simulations);
        const double quench_probability =
            avalanche_count > 0 ? static_cast<double>(quench_count) / static_cast<double>(avalanche_count) : 0.0;
        const double recharge_probability =
            quench_count > 0 ? static_cast<double>(recharge_count) / static_cast<double>(quench_count) : 0.0;

        const double rc_ns = parameters.resistance * parameters.capacitance * 1e9;

        std::cout << "\n\n";
        fmt::print("C = {:.2e}, R = {:.2e}, RC = {:.2f} ns, V_bias = {:.8g} V, W = {:.2e} cm\n",
                   parameters.capacitance,
                   parameters.resistance,
                   rc_ns,
                   parameters.breakdown_voltage + parameters.excess_voltage,
                   parameters.width);
        fmt::print("Results saved in directory: {}\n", output_dir);
        fmt::print("Probability of avalanche: {:.6f} ({}/{})\n", avalanche_probability, avalanche_count, num_simulations);
        fmt::print("Probability of quenching after avalanche: {:.6f} ({}/{})\n", quench_probability, quench_count, avalanche_count);
        fmt::print("Probability of recharge after quench: {:.6f} ({}/{})\n", recharge_probability, recharge_count, quench_count);
        fmt::print("Stopped by carrier limit: {}\n", carrier_limit_count);

        const std::string global_results_file = output_dir + "/global_results.txt";
        writeGlobalResults(global_results_file,
                           parameters,
                           num_simulations,
                           avalanche_count,
                           quench_count,
                           recharge_count,
                           carrier_limit_count,
                           avalanche_times,
                           quench_times,
                           recharge_times);

        std::cout << "\n -------------------------------- \n";

        if (parser["--plot"] == true) {
            const std::string python_script_file = std::string(CMAKE_SOURCE_DIR) + "/python/plot_quencher.py";
            const std::string command            = fmt::format("python {} {}", python_script_file, output_dir);
            fmt::print("Running command: {}\n", command);

            const int status = std::system(command.c_str());
            if (status != 0) {
                fmt::print("Failed to run the Python script.\n");
            }
        }
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << '\n';
        return 1;
    }

    return 0;
}
