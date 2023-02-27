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
    for (std::size_t index = 0; index < y.size(); ++index) {
        int idx_i              = static_cast<int>(index);
        T   sum                = 0;
        int actual_window_size = 0;
        for (int i = -window_size; i < window_size; ++i) {
            if (idx_i + i >= 0 && idx_i + i < static_cast<int>(y.size())) {
                sum += y[idx_i + i];
                ++actual_window_size;
            }
        }
        convol[index] = sum / static_cast<double>(actual_window_size);
    }
    return convol;
}
}  // namespace Utils