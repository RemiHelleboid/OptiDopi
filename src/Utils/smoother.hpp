#pragma once

#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

namespace Utils {

template <typename T>
std::vector<T> convol_square(const std::vector<T>& y, int window_size) {
    std::vector<T> convol;
    convol.resize(y.size());
    for (int index = 0; index < y.size(); ++index) {
        T sum = 0;
        int actual_window_size = 0;
        for (int i = -window_size; i < window_size; ++i) {
            if (index + i >= 0 && index + i < y.size()) {
                sum += y[index + i];
                ++actual_window_size;
            }
        }
        // std::cout << "actual_window_size = " << actual_window_size << std::endl;
        convol[index] = sum / static_cast<double>(actual_window_size);
        // std::cout << "Initial value = " << y[index] << ", convoluted value = " << convol[index] << std::endl;
    }
    return convol;
}
}  // namespace Utils