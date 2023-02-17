#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/xchar.h>

#include <filesystem>
#include <numeric>

#include "PoissonSolver.hpp"
#include "SimulatedAnneal.hpp"
#include "ImpactIonization.hpp"
#include "device.hpp"
#include "doping_profile.hpp"
#include "fill_vector.hpp"
#include "omp.h"

// Set number of threads

#define NAN_DOUBLE std::numeric_limits<double>::quiet_NaN()

int main(int argc, const char** argv) {
    // Create a complexe pin diode.
    double x_length = 10.0;
    std::size_t nb_points = 1000;
    double donor_length = 1.0;
    double intrisic_length = 0.0;

    double donor_level = 5.0e19;
    double intrisic_level = 1.0e13;

    std::vector<double> acceptor_x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0};
    std::vector<double> acceptor_level = {1.0e15, 1.0e16, 1.0e17, 1.0e16, 1.0e15, 1.0e14, 1.0e13, 1.0e12};

    device my_device;
    // device::set_up_complex_diode(double              xlength,
    //                               std::size_t         number_points,
    //                               double              length_donor,
    //                               double              length_intrinsic,
    //                               double              donor_level,
    //                               double              intrisic_level,
    //                               std::vector<double> list_acceptor_level,
    //                               std::vector<double> list_acceptor_width)
    my_device.set_up_complex_diode(x_length, nb_points, donor_length, intrisic_length, donor_level, intrisic_level, acceptor_x, acceptor_level);
    my_device.smooth_doping_profile(10);
    my_device.export_doping_profile("doping_profile_complex.csv");

    double    target_anode_voltage = 40.0;
    double    tol                  = 1.0e-6;
    const int max_iter             = 100;
    double    voltage_step         = 0.01;
    my_device.solve_poisson(target_anode_voltage, tol, max_iter);
    my_device.export_poisson_solution("poisson_solution", "poisson_solution_");
}
