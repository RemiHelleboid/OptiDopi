/**
 * Create and export dataset
 */

#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/xchar.h>
#include <omp.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <random>

#include "AdvectionDiffusionMC.hpp"
#include "Device1D.hpp"
#include "McIntyre.hpp"
#include "PoissonSolver1D.hpp"

std::vector<double> x_acceptors(double length_donor, double total_length, std::size_t nb_points_acceptor) {
    // The x positions are first on a fine grid then on a coarse grid
    double dx_fine        = 0.3;
    double size_fine_area = 2.0;

    std::vector<double> x_acceptor(nb_points_acceptor);
    x_acceptor[0] = length_donor;

    // First we fill the fine area
    std::size_t i = 1;
    while (x_acceptor[i - 1] < length_donor + size_fine_area) {
        x_acceptor[i] = x_acceptor[i - 1] + dx_fine;
        ++i;
    }
    double dx_coarse = (total_length - length_donor - size_fine_area) / (nb_points_acceptor - i);
    while (i < nb_points_acceptor && x_acceptor[i - 1] + dx_coarse <= total_length) {
        x_acceptor[i] = x_acceptor[i - 1] + dx_coarse;
        ++i;
    }
    x_acceptor[x_acceptor.size() - 1] = total_length;
    return x_acceptor;
}

cost_function_result main_function_simple_spad(double donor_length, double donor_level, double doping_acceptor, double total_length) {
    std::size_t number_points    = 400;
    double      length_donor     = donor_length;
    double      doping_donor     = donor_level;
    double      doping_intrinsic = 1.0e13;
    double      length_intrinsic = 0.0;

    Device1D my_device;
    my_device.setup_pin_diode(total_length, number_points, length_donor, length_intrinsic, doping_donor, doping_acceptor, doping_intrinsic);
    my_device.smooth_doping_profile(5);

    // Solve the Poisson and McIntyre equations
    double       target_anode_voltage  = 1000.0;
    double       tol                   = 1.0e-8;
    const int    max_iter              = 1000;
    double       mcintyre_voltage_step = 0.25;
    const double stop_above_bv         = 5.0;
    double       BiasAboveBV           = 3.0;

    my_device.solve_poisson_and_mcintyre(target_anode_voltage, tol, max_iter, mcintyre_voltage_step, stop_above_bv);
    bool poisson_success = my_device.get_poisson_success();
    if (!poisson_success) {
        // fmt::print(std::cerr, "Poisson failed\n");
    }
    cost_function_result cost_result = my_device.compute_cost_function(BiasAboveBV, 1.0);

    return cost_result;
}

cost_function_result main_function_spad_complex(double                     donor_length,
                                                double                     donor_level,
                                                double                     total_length,
                                                const std::vector<double>& x_acceptor,
                                                const std::vector<double>& doping_acceptor) {
    std::size_t number_points    = 200;
    double      intrinsic_length = 0.0;
    double      intrinsic_level  = 1.0e13;
    int         DopSmooth        = 11;

    Device1D my_device;
    my_device.set_up_complex_diode(total_length,
                                   number_points,
                                   donor_length,
                                   intrinsic_length,
                                   donor_level,
                                   intrinsic_level,
                                   x_acceptor,
                                   doping_acceptor);
    my_device.smooth_doping_profile(DopSmooth);

    // Solve the Poisson and McIntyre equations
    double       target_anode_voltage  = 1000.0;
    double       tol                   = 1.0e-8;
    const int    max_iter              = 1000;
    double       mcintyre_voltage_step = 0.25;
    const double stop_above_bv         = 5.0;
    double       BiasAboveBV           = 3.0;

    my_device.solve_poisson_and_mcintyre(target_anode_voltage, tol, max_iter, mcintyre_voltage_step, stop_above_bv);
    bool poisson_success = my_device.get_poisson_success();
    if (!poisson_success) {
        // fmt::print(std::cerr, "Poisson failed\n");
    }
    cost_function_result cost_result = my_device.compute_cost_function(BiasAboveBV, 1.0);

    return cost_result;
}

void generate_dataset_simple_spad(std::size_t NbSamples) {
    std::size_t number_points          = 200;
    double      min_total_length       = 5.0;
    double      max_total_length       = 10.0;
    double      min_donor_length       = 0.1;
    double      max_donor_length       = 4.0;
    double      log_min_donor_level    = std::log10(1.0e14);
    double      log_max_donor_level    = std::log10(1.0e20);
    double      log_min_acceptor_level = std::log10(1.0e14);
    double      log_max_acceptor_level = std::log10(1.0e20);

    // Create a random number generator
    std::random_device               rd;
    std::mt19937                     gen(rd());
    std::uniform_real_distribution<> dis_total_length(min_total_length, max_total_length);
    std::uniform_real_distribution<> dis_donor_length(min_donor_length, max_donor_length);
    std::uniform_real_distribution<> dis_donor_level(log_min_donor_level, log_max_donor_level);
    std::uniform_real_distribution<> dis_acceptor_level(log_min_acceptor_level, log_max_acceptor_level);

    // Create the output directory
    std::filesystem::path output_dir = "DATASET";
    if (!std::filesystem::exists(output_dir)) {
        std::filesystem::create_directory(output_dir);
    }

    std::size_t number_samples = 2500;

    std::vector<double> total_lengths(number_samples);
    std::vector<double> donor_lengths(number_samples);
    std::vector<double> donor_levels(number_samples);
    std::vector<double> acceptor_levels(number_samples);
    std::vector<double> BreakdownVoltages(number_samples);
    std::vector<double> BreakdownProbabilities(number_samples);
    std::vector<double> DepletionWidths(number_samples);
    std::vector<int>    failed_samples(number_samples);

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < number_samples; ++i) {
        double total_length   = dis_total_length(gen);
        double donor_length   = dis_donor_length(gen);
        double donor_level    = std::pow(10.0, dis_donor_level(gen));
        double acceptor_level = std::pow(10.0, dis_acceptor_level(gen));
        if (donor_length > total_length) {
            continue;
        }
        try {
            cost_function_result cost_result = main_function_simple_spad(donor_length, donor_level, acceptor_level, total_length);
            total_lengths[i]                 = total_length;
            donor_lengths[i]                 = donor_length;
            donor_levels[i]                  = donor_level;
            acceptor_levels[i]               = acceptor_level;
            BreakdownVoltages[i]             = cost_result.result.BV;
            BreakdownProbabilities[i]        = cost_result.result.BrP;
            DepletionWidths[i]               = cost_result.result.DW * 1.0e6;
        } catch (const std::exception& e) {
            failed_samples[i] = 1;
            continue;
        }
        if (i % 100 == 0) {
            fmt::print("Completed {} samples\n", i);
        }
    }
    // Save the data
    const std::string timestamp = fmt::format("{:%Y-%m-%d_%H-%M-%S}", fmt::localtime(std::time(nullptr)));
    const std::string filename  = fmt::format("{}/dataset_{}.csv", output_dir.string(), timestamp);
    fmt::print("Saving data to {}\n", filename);
    std::ofstream file(filename);
    file << "TotalLength,DonorLength,DonorLevel,AcceptorLevel,BreakdownVoltage,BreakdownProbability,DepletionWidth\n";
    for (std::size_t i = 0; i < number_samples; ++i) {
        if (failed_samples[i] == 1) {
            continue;
        }
        file << fmt::format("{:.3f},{:.3f},{:.3e},{:.3e},{:.3f},{:.3f},{:.3f}\n",
                            total_lengths[i],
                            donor_lengths[i],
                            donor_levels[i],
                            acceptor_levels[i],
                            BreakdownVoltages[i],
                            BreakdownProbabilities[i],
                            DepletionWidths[i]);
    }
}

void create_dataset_complex_spad(std::size_t nb_samples) {
    std::size_t number_points    = 100;
    double      intrinsic_length = 0.0;
    double      intrinsic_level  = 1e12;
    int         DopSmooth        = 11;

    double min_total_length       = 8.0;
    double max_total_length       = 8.0;
    double min_donor_length       = 0.1;
    double max_donor_length       = 4.0;
    double log_min_donor_level    = std::log10(1.0e16);
    double log_max_donor_level    = std::log10(1.0e20);
    double log_min_acceptor_level = std::log10(1.0e16);
    double log_max_acceptor_level = std::log10(1.0e20);
    int    NbAcceptors            = 10;

    std::size_t NbPointSamplingDoping = 20;

    // Create a random number generator
    std::uniform_real_distribution<> dis_total_length(min_total_length, max_total_length);
    std::uniform_real_distribution<> dis_donor_length(min_donor_length, max_donor_length);
    std::uniform_real_distribution<> dis_donor_level(log_min_donor_level, log_max_donor_level);
    std::uniform_real_distribution<> dis_acceptor_level(log_min_acceptor_level, log_max_acceptor_level);

    std::vector<double>              total_lengths(nb_samples);
    std::vector<double>              donor_lengths(nb_samples);
    std::vector<std::vector<double>> doping_levels(nb_samples, std::vector<double>(number_points));
    std::vector<std::vector<double>> donor_levels(nb_samples, std::vector<double>(number_points));
    std::vector<std::vector<double>> acceptor_levels(nb_samples, std::vector<double>(number_points));

    std::vector<double> BreakdownVoltages(nb_samples);
    std::vector<double> BreakdownProbabilities(nb_samples);
    std::vector<double> DepletionWidths(nb_samples);
    std::vector<int>    failed_samples(nb_samples);

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < nb_samples; ++i) {
        std::random_device               rd;
        std::mt19937                     gen(rd());
        double                           total_length = dis_total_length(gen);
        std::uniform_real_distribution<> dis_donor_length(min_donor_length, total_length / 2.0);
        double                           donor_length = dis_donor_length(gen);
        if (donor_length > total_length) {
            continue;
        }
        double              donor_level = std::pow(10.0, dis_donor_level(gen));
        std::vector<double> acceptor_x  = x_acceptors(donor_length, total_length, NbAcceptors);
        std::vector<double> acceptor_lev(NbAcceptors);
        for (int j = 0; j < NbAcceptors; ++j) {
            acceptor_lev[j] = std::pow(10.0, dis_acceptor_level(gen));
        }
        Device1D my_device;
        my_device.set_up_complex_diode(total_length,
                                       number_points,
                                       donor_length,
                                       intrinsic_length,
                                       donor_level,
                                       intrinsic_level,
                                       acceptor_x,
                                       acceptor_lev);
        // my_device.smooth_doping_profile(DopSmooth);
        donor_levels[i]    = my_device.get_doping_profile().get_donor_concentration();
        acceptor_levels[i] = my_device.get_doping_profile().get_acceptor_concentration();
        doping_levels[i]   = my_device.get_doping_profile().get_doping_concentration();
        total_lengths[i]   = total_length;
        donor_lengths[i]   = donor_length;
        try {
            cost_function_result cost_result =
                main_function_spad_complex(donor_length, donor_level, total_length, acceptor_x, acceptor_lev);
            BreakdownVoltages[i]      = cost_result.result.BV;
            BreakdownProbabilities[i] = cost_result.result.BrP;
            DepletionWidths[i]        = cost_result.result.DW * 1.0e6;
        } catch (const std::exception& e) {
            failed_samples[i] = 1;
            continue;
        }
        if (i % 10 == 0) {
            fmt::print("Sample {} / {}\n", i, nb_samples);
        }
    }

    std::cout << "Number of failed samples: " << std::accumulate(failed_samples.begin(), failed_samples.end(), 0) << std::endl;

    // Create the output directory
    std::filesystem::path output_dir = "DATASET_COMPLEX_SPAD";
    if (!std::filesystem::exists(output_dir)) {
        std::filesystem::create_directory(output_dir);
    }
    // Save the dataset
    const std::string timestamp = fmt::format("{:%Y-%m-%d_%H-%M-%S}", fmt::localtime(std::time(nullptr)));
    const std::string filename  = fmt::format("{}/dataset_complex_{}.csv", output_dir.string(), timestamp);
    fmt::print("Saving data to {}\n", filename);

    std::ofstream file(filename);
    file << "TotalLength,DonorLength";
    for (std::size_t i = 0; i < number_points; ++i) {
        file << fmt::format(",Donor_{}", i);
    }
    for (std::size_t i = 0; i < number_points; ++i) {
        file << fmt::format(",Acceptors_{}", i);
    }
    file << ",BreakdownVoltage,BreakdownProbability,DepletionWidth\n";
    for (std::size_t i = 0; i < nb_samples; ++i) {
        if (failed_samples[i] == 1) {
            continue;
        }
        file << fmt::format("{:.3f}", total_lengths[i]) << fmt::format(",{:.3f}", donor_lengths[i]);
        for (std::size_t j = 0; j < number_points; ++j) {
            file << fmt::format(",{:.3e}", donor_levels[i][j]);
        }
        for (std::size_t j = 0; j < number_points; ++j) {
            file << fmt::format(",{:.3e}", acceptor_levels[i][j]);
        }
        file << fmt::format(",{:.3f},{:.3f},{:.3f}\n", BreakdownVoltages[i], BreakdownProbabilities[i], DepletionWidths[i]);
    }
    file.close();
}

int main(int argc, char** argv) {
    int nb_threads = 1;

#pragma omp parallel
    { nb_threads = omp_get_num_threads(); }
    std::cout << "Number threads: " << nb_threads << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    int number_samples = 200;
    if (argc > 1) {
        number_samples = std::stoi(argv[1]);
    }
    std::cout << "Nb samples: " << number_samples << std::endl;
    fmt::print("Number of samples : {}\n", number_samples);

    // generate_dataset_simple_spad(number_samples);
    std::cout << "Create dataset of complex 1D SPADs..." << std::endl;
    create_dataset_complex_spad(number_samples);

    auto                          end             = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    fmt::print("Total time : {:.3f} s \n\n", elapsed_seconds.count());

    return 0;
}
