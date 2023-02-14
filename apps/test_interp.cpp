/**
 * @file test_interp.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-02-14
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <fstream>
#include <vector>

#include "interpolation.hpp"
#include "fill_vector.hpp"


int main(int argc, const char** argv) {
    double x0 = -1.0;
    double x1 = 158.2;

    std::vector<double> x = utils::linspace(x0, x1, 100);
    std::vector<double> y = utils::linspace(x0, x1, 100);

    // Resample the vector y
    std::vector<double> x_resampled = utils::linspace(x0, x1, 1000);
    std::vector<double> y_resampled = Utils::interp1d(x, y, x_resampled);

    // Print the result in a file
    std::ofstream file("test_interp.csv");
    for (std::size_t i = 0; i < x_resampled.size(); ++i) {
        file << x_resampled[i] << "," << y_resampled[i] << std::endl;
    }
    file.close();


    return 0;
    
}