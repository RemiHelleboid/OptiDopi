/**
 * @file Vector3.hpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-03-01
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include <cmath>
#include <iostream>
#include <vector>

class Vector3 {
    double m_x;
    double m_y;
    double m_z;

public:
    Vector3() = default;
    Vector3(double x, double y, double z) : m_x(x), m_y(y), m_z(z) {}
    Vector3(const Vector3& other) = default;
    Vector3(Vector3&& other)      = default;
    ~Vector3()                    = default;

    Vector3& operator=(const Vector3& other) = default;
    Vector3& operator=(Vector3&& other) = default;

    double x() const { return m_x; }
    double y() const { return m_y; }
    double z() const { return m_z; }

    void set_x(double x) { m_x = x; }
    void set_y(double y) { m_y = y; }
    void set_z(double z) { m_z = z; }

    double norm() const { return std::sqrt(m_x * m_x + m_y * m_y + m_z * m_z); }

    Vector3& operator+=(const Vector3& other) {
        m_x += other.m_x;
        m_y += other.m_y;
        m_z += other.m_z;
        return *this;
    }

    Vector3& operator-=(const Vector3& other) {
        m_x -= other.m_x;
        m_y -= other.m_y;
        m_z -= other.m_z;
        return *this;
    }

    Vector3& operator*=(double scalar) {
        m_x *= scalar;
        m_y *= scalar;
        m_z *= scalar;
        return *this;
    }

    Vector3& operator/=(double scalar) {
        m_x /= scalar;
        m_y /= scalar;
        m_z /= scalar;
        return *this;
    }

    Vector3 operator-() const { return Vector3(-m_x, -m_y, -m_z); }

    friend Vector3 operator+(const Vector3& lhs, const Vector3& rhs) {
        return Vector3(lhs.m_x + rhs.m_x, lhs.m_y + rhs.m_y, lhs.m_z + rhs.m_z);
    }

    friend Vector3 operator-(const Vector3& lhs, const Vector3& rhs) {
        return Vector3(lhs.m_x - rhs.m_x, lhs.m_y - rhs.m_y, lhs.m_z - rhs.m_z);
    }

    friend Vector3 operator*(const Vector3& lhs, double scalar) {
        return Vector3(lhs.m_x * scalar, lhs.m_y * scalar, lhs.m_z * scalar);
    }

    friend Vector3 operator*(double scalar, const Vector3& rhs) {
        return Vector3(rhs.m_x * scalar, rhs.m_y * scalar, rhs.m_z * scalar);
    }

    friend Vector3 operator/(const Vector3& lhs, double scalar) {
        return Vector3(lhs.m_x / scalar, lhs.m_y / scalar, lhs.m_z / scalar);
    }

    friend double dot(const Vector3& lhs, const Vector3& rhs) {
        return lhs.m_x * rhs.m_x + lhs.m_y * rhs.m_y + lhs.m_z * rhs.m_z;
    }

    friend Vector3 cross(const Vector3& lhs, const Vector3& rhs) {
        return Vector3(lhs.m_y * rhs.m_z - lhs.m_z * rhs.m_y,
                       lhs.m_z * rhs.m_x - lhs.m_x * rhs.m_z,
                       lhs.m_x * rhs.m_y - lhs.m_y * rhs.m_x);
    }

    friend std::ostream& operator<<(std::ostream& os, const Vector3& vec) {
        os << vec.m_x << " " << vec.m_y << " " << vec.m_z;
        return os;
    }
};