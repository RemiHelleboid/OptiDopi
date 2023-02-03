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
        for (int i = -window_size; i < window_size; ++i) {
            if (index + i >= 0 && index + i < y.size()) {
                sum += y[index + i];
            }
        }
        convol[index] = sum / (2 * window_size + 1);
    }
    return convol;
}
}  // namespace Utils