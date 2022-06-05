/**
 * @file doping_profile.hpp
 * @author remzerrr (remi.helleboid@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-05
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <iostream>
#include <memory>
#include <random>
#include <vector>

class doping_profile {
 private:
    std::vector<double> m_x_line;
    std::vector<double> m_acceptor_concentration;
    std::vector<double> m_donor_concentration;
    std::vector<double> m_doping_concentration;

 public:
    doping_profile();
    doping_profile(double x_min, double x_max, std::size_t number_points);
    doping_profile(const doping_profile&) = default;

    void re_compute_total_doping();
    void set_up_pin_diode(double      x_min,
                          double      x_max,
                          std::size_t number_points,
                          double      length_donor,
                          double      length_intrinsic,
                          double      donor_level,
                          double      acceptor_level,
                          double      intrisic_level);
};