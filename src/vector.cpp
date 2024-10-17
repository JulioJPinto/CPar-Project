#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "vector.h"

/* Vector class implementation. */

Vector::Vector() {
    // Initialize the vector to zero
    value = _mm256_setzero_ps();
}

Vector::Vector(float initialValue) {
    // Initialize all elements of the vector to the same float value
    value = _mm256_set1_ps(initialValue);
}

Vector::Vector(__m256 initialValue) {
    // Initialize the vector with the provided __m256 value
    value = initialValue;
}

Vector::Vector(float a, float b, float c, float d, float e, float f, float g, float h) {
    // Set each element of the vector with the provided values
    value = _mm256_set_ps(h, g, f, e, d, c, b, a);
}

Vector::Vector(float* array) {
    // Load the float values from the array into the vector
    value = _mm256_loadu_ps(array);
}

void Vector::store(float* array) const {
    // Store the vector values into the provided array
    _mm256_storeu_ps(array, value);
}

__m256 Vector::getValue() const {
    // Return the current __m256 value of the vector
    return value;
}

Vector Vector::operator+(const Vector& other) const {
    // Perform element-wise addition of two vectors
    return Vector(_mm256_add_ps(value, other.getValue()));
}

Vector Vector::operator-(const Vector& other) const {
    // Perform element-wise subtraction of two vectors
    return Vector(_mm256_sub_ps(value, other.getValue()));
}

Vector Vector::operator*(const Vector& other) const {
    // Perform element-wise multiplication of two vectors
    return Vector(_mm256_mul_ps(value, other.getValue()));
}

Vector Vector::operator/(const Vector& other) const {
    // Perform element-wise division of two vectors
    return Vector(_mm256_div_ps(value, other.getValue()));
}

Vector Vector::floor() const {
    // Perform element-wise floor operation on the vector
    return Vector(_mm256_floor_ps(value));
}

static Vector min(const Vector& a, const Vector& b) {
    // Perform element-wise min operation on two vectors
    return Vector(_mm256_min_ps(a.value, b.value));
}

static Vector max(const Vector& a, const Vector& b) {
    // Perform element-wise max operation on two vectors
    return Vector(_mm256_max_ps(a.value, b.value));
}
