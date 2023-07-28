/**
 * @file McIntyreSolo.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-07-04
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <fmt/core.h>
// #include <fmt/stream.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <random>

#include "AdvectionDiffusionMC.hpp"
#include "Device1D.hpp"
#include "McIntyre.hpp"
#include "PoissonSolver1D.hpp"

std::vector<std::vector<double>> read_csv_fieldline(const std::string filename, int idx_col_x, int idx_col_field) {
    std::ifstream       file(filename);
    std::vector<double> x_line;
    std::vector<double> electric_field_line;
    std::string         line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double             x, electric_field;
        if (!(iss >> x >> electric_field)) {
            throw std::runtime_error("Error reading file");
        }

        
        x_line.push_back(x);
        electric_field_line.push_back(electric_field);
    }
    std::cout << "Size of the line: " << x_line.size() << std::endl;
    std::cout << "Size of the line: " << electric_field_line.size() << std::endl;
    return {x_line, electric_field_line};
}

void solve_mcintyre_over_field_line(std::vector<double> x_line, std::vector<double> electric_field_line) {
    if (x_line.size() != electric_field_line.size()) {
        throw std::runtime_error("x_line and electric_field_line must have the same size");
    }
    for (auto &x: x_line){
        x *= 1.0e6;
    }
    mcintyre::McIntyre mcintyre_solver;
    mcintyre_solver.set_xline(x_line);
    mcintyre_solver.set_electric_field(electric_field_line, true, 1.0e-4);

    double tol = 1.0e-8;
    mcintyre_solver.ComputeDampedNewtonSolution(tol);

    if (!mcintyre_solver.get_Solver_Has_Converged()) {
        std::cout << "McIntyre solver has not converged" << std::endl;
        exit(1);
    }

    mcintyre::McIntyreSolution mcintyre_solution = mcintyre_solver.get_solution();
    mcintyre_solution.export_to_file("McIntyreSolution.csv", x_line, electric_field_line);

    std::cout << "Total BrP: " << mcintyre_solution.m_mean_breakdown_probability << std::endl;
}

int main(int argc, const char** argv) {
    std::string filename_fieldline;
    int index_col_x = 0;
    int index_col_field = 1;

    if (argc < 2) {
        std::cout << "Missing arguments.\n";
        std::cout << "Usage: McIntyreSolo.x filename\n";
    }
    filename_fieldline = std::string(argv[1]);
    if (argc == 4) {
        index_col_x = std::atoi(argv[2]);
        index_col_field = std::atoi(argv[3]);
    }

    auto data = read_csv_fieldline(filename_fieldline, index_col_x, index_col_field);
    solve_mcintyre_over_field_line(data[0], data[1]);

    return 0;

}