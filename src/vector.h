#pragma once

#include <immintrin.h>

/* Class for SIMD operations using AVX-256 floats (8 floats per vector) */

class Vector {
private:
    __m256 value; // Represents a 256-bit vector of eight float-precision floating-point numbers.

public:

    /**
     * Initializes the vector with all elements set to zero.
     */
    Vector();

    /**
     * Initializes the vector with a single float value repeated across all elements.
     * @param initialValue - The initial value for all elements of the vector.
     */
    Vector(float initialValue);

    /**
     * Initializes the vector with a given __m256 value.
     * @param initialValue - The initial __m256 value for the vector.
     */
    Vector(__m256 initialValue);

    /**
     * Initializes the vector with eight specified float values.
     * @param a, b, c, d, e, f, g, h - The eight float values to initialize the vector.
     */
    Vector(float a, float b, float c, float d, float e, float f, float g, float h);

    /**
     * Initializes the vector by loading values from an array of eight floats.
     * @param array - The array containing float values to load into the vector.
     */
    Vector(float* array);

    /**
     * Stores the vector values into a provided float array.
     * @param array - The array where the vector values will be stored (must hold at least 8 floats).
     */
    void store(float* array) const;

    /**
     * Retrieves the current vector value.
     * @return __m256 - The current value of the vector.
     */
    __m256 getValue() const;

    /**
     * Performs element-wise addition of two vectors.
     * @param other - The Vector to be added.
     * @return Vector - The result of the addition operation.
     */
    Vector operator+(const Vector& other) const;

    /**
     * Performs element-wise subtraction of two vectors.
     * @param other - The Vector to be subtracted.
     * @return Vector - The result of the subtraction operation.
     */
    Vector operator-(const Vector& other) const;

    /**
     * Performs element-wise multiplication of two vectors.
     * @param other - The Vector to be multiplied.
     * @return Vector - The result of the multiplication operation.
     */
    Vector operator*(const Vector& other) const;

    /**
     * Performs element-wise division of two vectors.
     * @param other - The Vector to be divided.
     * @return Vector - The result of the division operation.
     */
    Vector operator/(const Vector& other) const;

    Vector floor() const;
    // Min function for vectors
    static Vector min(const Vector& a, const Vector& b);

    // Max function for vectors
    static Vector max(const Vector& a, const Vector& b);

};
