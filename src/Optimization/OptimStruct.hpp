/**
 * @file Struct.hpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-02-17
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include <iostream>
#include <memory>
#include <random>

#include "device.hpp"



struct result_sim {
    double               m_length_intrinsic;
    double               m_doping_acceptor;
    double               m_BV;
    double               m_BrP;
    double               m_DW;
    cost_function_result m_cost_result;

    result_sim(double length_intrinsic, double doping_acceptor, double BV, double BrP, double DW, cost_function_result cost) {
        this->m_length_intrinsic = length_intrinsic;
        this->m_doping_acceptor  = doping_acceptor;
        this->m_BV               = BV;
        this->m_BrP              = BrP;
        this->m_DW               = DW;
        this->m_cost_result      = cost;
    }
};