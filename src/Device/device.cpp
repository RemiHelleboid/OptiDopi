/**
 * @file device.cpp
 */

#include "device.hpp"



void device::add_doping_profile(doping_profile& doping_profile) { m_doping_profile = doping_profile; }

void device::setup_pin_diode(double      xlenght,
                             std::size_t number_points,
                             double      length_donor,
                             double      length_intrinsic,
                             double      donor_level,
                             double      acceptor_level,
                             double      intrisic_level) {
    m_doping_profile
        .set_up_pin_diode(0.0, xlenght, number_points, length_donor, length_intrinsic, donor_level, acceptor_level, intrisic_level);
}
