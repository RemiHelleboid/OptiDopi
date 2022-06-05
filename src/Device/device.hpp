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
    device();
};