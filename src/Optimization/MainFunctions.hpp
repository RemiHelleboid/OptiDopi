/**
 * @file MainFunctions.hpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-02-22
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/xchar.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>

#include "ImpactIonization.hpp"
#include "OptimStruct.hpp"
#include "ParticleSwarm.hpp"
#include "PoissonSolver1D.hpp"
#include "SimulatedAnneal.hpp"
#include "Device1D.hpp"
#include "DopingProfile1D.hpp"
#include "fill_vector.hpp"
#include "omp.h"


namespace Optimization {

std::vector<double> x_acceptors(double length_donor, double total_length, std::size_t nb_points_acceptor);

void export_best_path(std::vector<std::vector<double>> best_path, std::string dirname);

double intermediate_cost_function(double donor_length, double log_donor_level, std::vector<double> log_acceptor_levels);

double cost_function(std::vector<double> variables);

void MainParticleSwarmSPAD(std::size_t nb_particles, std::size_t max_iter, double w, double c1, double c2);

void MainSimulatedAnnealingSPAD();


} // namespace Optimization