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
#include "PoissonSolver.hpp"
#include "SimulatedAnneal.hpp"
#include "device.hpp"
#include "doping_profile.hpp"
#include "fill_vector.hpp"
#include "omp.h"


namespace Optimization {

std::vector<double> x_acceptors(double length_donor, double total_length, std::size_t nb_points_acceptor);

void export_best_path(std::vector<std::vector<double>> best_path, std::string dirname);

double intermediate_cost_function(double donor_length, double log_donor_level, std::vector<double> log_acceptor_levels);

double cost_function(std::vector<double> variables);

void MainParticleSwarmSPAD();

void MainSimulatedAnnealingSPAD();


} // namespace Optimization