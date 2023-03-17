#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/xchar.h>

#include <filesystem>
#include <numeric>

#include "Device1D.hpp"
#include "McIntyre.hpp"
#include "PoissonSolver1D.hpp"
#include "fill_vector.hpp"
#include "omp.h"

// Set number of threads

#define NAN_DOUBLE std::numeric_limits<double>::quiet_NaN()
#define BIG_DOUBLE 1.0e10

static int idx = 0;

struct result_sim {
    double               length_intrinsic;
    double               doping_acceptor;
    double               BV;
    double               BrP;
    double               DW;
    cost_function_result cost_result;

    result_sim(double length_intrinsic, double doping_acceptor, double BV, double BrP, double DW, cost_function_result cost) {
        this->length_intrinsic = length_intrinsic;
        this->doping_acceptor  = doping_acceptor;
        this->BV               = BV;
        this->BrP              = BrP;
        this->DW               = DW;
        this->cost_result      = cost;
    }
};

double intermediate_cost_function(double length_intrinsic, double log_doping_donor, double log_doping_acceptor, int thread_id = 0) {
    std::size_t number_points    = 150;
    double      total_length     = 10.0;
    double      length_donor     = 1.0;
    double      doping_intrinsic = 1.0e13;

    double doping_donor    = pow(10, log_doping_donor);
    double doping_acceptor = pow(10, log_doping_acceptor);

    Device1D my_device;
    my_device.setup_pin_diode(total_length, number_points, length_donor, length_intrinsic, doping_donor, doping_acceptor, doping_intrinsic);
    my_device.smooth_doping_profile(5);
    double       target_anode_voltage  = 30000.0;
    double       tol                   = 1.0e-8;
    const int    max_iter              = 1000;
    double       mcintyre_voltage_step = 0.25;
    const double stop_above_bv         = 5.0;
    double       BiasAboveBV           = 3.0;

    my_device.solve_poisson_and_mcintyre(target_anode_voltage, tol, max_iter, mcintyre_voltage_step, stop_above_bv);
    bool poisson_success = my_device.get_poisson_success();
    if (!poisson_success) {
        // fmt::print("Poisson failed\n");
        return BIG_DOUBLE;
    }

    double               time_       = 0.0;
    cost_function_result cost_result = my_device.compute_cost_function(BiasAboveBV, time_);
    double               cost        = cost_result.total_cost;
    // fmt::print("BV: {:.2f}, BRP: {:.2f}, DW: {:.2e}, Cost: {:.2f}\n", BV, BRP, DW, cost);

    return cost;
}

double cost_function(std::vector<double> variables) {
    double length_intrinsic = variables[0];
    double doping_donor     = variables[1];
    double doping_acceptor  = variables[1];
    double cost             = intermediate_cost_function(length_intrinsic, doping_donor, doping_acceptor);
    // fmt::print("Doping: {:.5e}, Length: {:.5e}, Cost: {:.5e}\n", pow(10, doping_acceptor), length_intrinsic, cost);
    return cost;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Error in arguments\n";
        std::cout << "Usage is > spad_cost_function length_intrinsic log_donor_level log_acceptor_level file_result.csv";
    }
    double arg_length_intrinsic = atof(argv[1]);
    double log_donor_level      = atof(argv[2]);
    double log_doping_acceptor  = atof(argv[3]);

    const std::string& file_res = std::string(argv[4]);
    double         results  = intermediate_cost_function(arg_length_intrinsic, log_donor_level, log_doping_acceptor);

    // double               length_intrinsic = results.length_intrinsic;
    // double               doping_acceptor  = results.doping_acceptor;
    // double               BV               = results.BV;
    // double               BrP              = results.BrP;
    // double               DW               = results.DW;
    // cost_function_result cost_result      = results.cost_result;

    // double BV_cost    = cost_result.BV_cost;
    // double BP_cost    = cost_result.BP_cost;
    // double DW_cost    = cost_result.DW_cost;
    double total_cost = results;

    std::ofstream file_result(file_res);
    fmt::print(file_result,
               "{:.6e},{:.6e},{:.6e},{:.6e}\n",
               arg_length_intrinsic,
               log_donor_level,
               log_doping_acceptor,
               total_cost);
    file_result.close();
}