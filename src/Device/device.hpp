/**
 * @file device.hpp
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

#include "doping_profile.hpp"

class device {
 private:
    doping_profile m_doping_profile;

 public:
    device() = default;
    ~device() = default;

    const doping_profile& get_doping_profile() const { return m_doping_profile; }

    void add_doping_profile(doping_profile& doping_profile);
    void setup_pin_diode(double      xlenght,
                         std::size_t number_points,
                         double      length_donor,
                         double      length_intrinsic,
                         double      donor_level,
                         double      acceptor_level,
                         double      intrisic_level);

   void export_doping_profile(const std::string& filename) const { m_doping_profile.export_doping_profile(filename); }
};