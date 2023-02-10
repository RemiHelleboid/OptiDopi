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

constexpr double m3_to_cm3 = 1.0e6;

class doping_profile {
 private:
    std::vector<double> m_x_line;
    std::vector<double> m_acceptor_concentration;
    std::vector<double> m_donor_concentration;
    std::vector<double> m_doping_concentration;

 public:
    doping_profile() = default;
    doping_profile(double x_min, double x_max, std::size_t number_points);
    doping_profile(const doping_profile&) = default;

    void smooth_doping_profile(int window_size);

    std::vector<double> get_x_line() const { return m_x_line; }
    std::vector<double> get_acceptor_concentration() const { return m_acceptor_concentration; }
    std::vector<double> get_donor_concentration() const { return m_donor_concentration; }
    std::vector<double> get_doping_concentration() const { return m_doping_concentration; }

    void re_compute_total_doping();
    void set_up_pin_diode(double      x_min,
                          double      x_max,
                          std::size_t number_points,
                          double      length_donor,
                          double      length_intrinsic,
                          double      donor_level,
                          double      acceptor_level,
                          double      intrisic_level);

    void set_up_advanced_spad(double              total_length,
                              std::size_t         nb_points,
                              double              length_donor,
                              double              length_intrinsic,
                              double              donor_level,
                              double              intrinsic_level,
                              std::vector<double> acceptor_segment_length,
                              std::vector<double> acceptor_concentrations);

    void export_doping_profile(const std::string& filename) const;
};
