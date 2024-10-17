#include "fluid_solver.h"
#include "vector.h"
#include <cmath>
#include <algorithm> // For std::max

#define IX(i, j, k) ((i) + (M + 2) * ((j) + (N + 2) * (k)))

#define SWAP(x0, x) { std::swap(x0, x); }
#define MAX3(a, b, c) (std::max({a, b, c}))
#define LINEARSOLVERTIMES 20

// Block size for tiling optimization
#define BLOCK_SIZE 4
#define BLOCK_VECT_SIZE 8

/**
 * @brief Adds the source term to the target array.
 * 
 * This function adds the source values `s` to the target array `x`, scaled by the time step `dt`.
 * 
 * @param M Size of the grid in the x-dimension.
 * @param N Size of the grid in the y-dimension.
 * @param O Size of the grid in the z-dimension.
 * @param x Target array to which the source will be added.
 * @param s Source array.
 * @param dt Time step for the simulation.
 */
void add_source(int M, int N, int O, float *x, float *s, float dt) {
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    x[i] += dt * s[i];
  }
}

/**
 * @brief Sets the boundary conditions for the simulation.
 * 
 * This function applies boundary conditions to the simulation data, ensuring that
 * the values at the boundaries behave correctly based on the type of data being handled.
 * 
 * @param M Size of the grid in the x-dimension.
 * @param N Size of the grid in the y-dimension.
 * @param O Size of the grid in the z-dimension.
 * @param b Boundary condition type (1, 2, or 3 for velocity; 0 for density).
 * @param x Array representing the field for which boundary conditions are set.
 */
void set_bnd(int M, int N, int O, int b, float *x) {
  int i, j;
  int loopMN = 1, loopNO = 1, loopMO = 1; 

  switch (b) {
    case 3:
      loopMN = -1;
      break;
    case 2:
      loopNO = -1;
      break;
    case 1:
      loopMO = -1;
      break;
  }

  // Set boundary on faces
  for (j = 1; j <= N; j++) {
    for (i = 1; i <= N; i++) {
      x[IX(i, j, 0)] = x[IX(i, j, 1)] * loopMN;
      x[IX(i, j, O + 1)] = x[IX(i, j, O)] * loopMN;
    }
  }

  for (j = 1; j <= O; j++) {
    for (i = 1; i <= N; i++) {
      x[IX(0, i, j)] = x[IX(1, i, j)] * loopNO;
      x[IX(M + 1, i, j)] = x[IX(M, i, j)] * loopNO;
    }
  }

  for (j = 1; j <= O; j++) {
    for (i = 1; i <= M; i++) {
      x[IX(i, 0, j)] = x[IX(i, 1, j)] * loopMO;
      x[IX(i, N + 1, j)] = x[IX(i, N, j)] * loopMO;
    }
  }

  // Set corners
  x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
  x[IX(M + 1, 0, 0)] =
      0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
  x[IX(0, N + 1, 0)] =
      0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
  x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] +
                                    x[IX(M + 1, N + 1, 1)]);
}

/**
 * @brief Performs the linear solver step using the Gauss-Seidel method.
 * 
 * This function solves the linear system using an iterative Gauss-Seidel method to handle the diffusion process.
 * 
 * @param M Size of the grid in the x-dimension.
 * @param N Size of the grid in the y-dimension.
 * @param O Size of the grid in the z-dimension.
 * @param b Boundary condition type.
 * @param x Output array for the solved values.
 * @param x0 Input array of the previous values.
 * @param a Coefficient used in the solver.
 * @param c Constant term in the linear system.
 */
void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
  float invA = a / c;
  float invC = 1 / c;

  for (int l = 0; l < LINEARSOLVERTIMES; l++) {
    for (int kk = 1; kk <= O; kk += BLOCK_SIZE) {
      for (int jj = 1; jj <= N; jj += BLOCK_SIZE) {
        for (int ii = 1; ii <= M; ii += BLOCK_SIZE) {
          // Process each block
          for (int k = kk; k < std::min(kk + BLOCK_SIZE, O + 1); k++) {
            for (int j = jj; j < std::min(jj + BLOCK_SIZE, N + 1); j++) {
              for (int i = ii; i < std::min(ii + BLOCK_SIZE, M + 1); i++) {
                int idx = IX(i, j, k);
                // Cache neighbor values
                float left = x[IX(i - 1, j, k)];
                float right = x[IX(i + 1, j, k)];
                float down = x[IX(i, j - 1, k)];
                float up = x[IX(i, j + 1, k)];
                float back = x[IX(i, j, k - 1)];
                float front = x[IX(i, j, k + 1)];
                
                x[idx] = (x0[idx] * invC +
                          invA * (left + right + down + up + back + front));
              }
            }
          }
        }
      }
    }
    set_bnd(M, N, O, b, x);
  }
}

/**
 * @brief Diffusion step for the fluid simulation.
 * 
 * This function simulates the diffusion of quantities (e.g., velocity or density)
 * using an implicit method.
 * 
 * @param M Size of the grid in the x-dimension.
 * @param N Size of the grid in the y-dimension.
 * @param O Size of the grid in the z-dimension.
 * @param b Boundary condition type.
 * @param x Output array for the diffused values.
 * @param x0 Input array for the previous values.
 * @param diff Diffusion coefficient.
 * @param dt Time step for the simulation.
 */
void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff, float dt) {
  int max = MAX3(M, N, O);
  float a = dt * diff * max * max;
  lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

/**
 * @brief Advection step for the fluid simulation.
 * 
 * This function moves quantities (e.g., velocity or density) through the velocity field.
 * 
 * @param M Size of the grid in the x-dimension.
 * @param N Size of the grid in the y-dimension.
 * @param O Size of the grid in the z-dimension.
 * @param b Boundary condition type.
 * @param d Output array for the advected values.
 * @param d0 Input array of the previous values.
 * @param u x-component of the velocity field.
 * @param v y-component of the velocity field.
 * @param w z-component of the velocity field.
 * @param dt Time step for the simulation.
 */
void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
    float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

    Vector dtX_vec(dtX), dtY_vec(dtY), dtZ_vec(dtZ);
    Vector half_vec(0.5f);
    Vector M_vec((float)M + 0.5f), N_vec((float)N + 0.5f), O_vec((float)O + 0.5f);

    // Outer block loops (blocking technique to improve cache usage)
    for (int kk = 1; kk <= O; kk += BLOCK_VECT_SIZE) {
        for (int jj = 1; jj <= N; jj += BLOCK_VECT_SIZE) {
            for (int ii = 1; ii <= M; ii += BLOCK_VECT_SIZE) {

                // Process each block
                int k_end = std::min(kk + BLOCK_VECT_SIZE, O + 1);
                int j_end = std::min(jj + BLOCK_VECT_SIZE, N + 1);
                int i_end = std::min(ii + BLOCK_VECT_SIZE, M + 1);

                for (int k = kk; k < k_end; ++k) {
                  for (int j = jj; j < j_end; ++j) {
                      for (int i = ii; i < i_end; i += BLOCK_VECT_SIZE) {
                          // Cache the base index for vectorized operations
                          int idx_base = IX(i, j, k);

                          // Load velocities (u, v, w) for BLOCK_VECT_SIZE elements at once
                          Vector u_vec(&u[idx_base]);
                          Vector v_vec(&v[idx_base]);
                          Vector w_vec(&w[idx_base]);

                          // Backtrace particle positions (x, y, z)
                          Vector i_vec((float)i, (float)(i+1), (float)(i+2), (float)(i+3),
                                      (float)(i+4), (float)(i+5), (float)(i+6), (float)(i+7));
                          Vector j_vec((float)j); // j and k are the same for all elements
                          Vector k_vec((float)k);

                          Vector x = i_vec - dtX_vec * u_vec;
                          Vector y = j_vec - dtY_vec * v_vec;
                          Vector z = k_vec - dtZ_vec * w_vec;

                          // Clamp positions to grid boundaries
                          x = Vector::max(half_vec, Vector::min(M_vec, x));
                          y = Vector::max(half_vec, Vector::min(N_vec, y));
                          z = Vector::max(half_vec, Vector::min(O_vec, z));

                          // Precompute integer and fractional components for interpolation
                          Vector i0_vec = x.floor();
                          Vector j0_vec = y.floor();
                          Vector k0_vec = z.floor();

                          Vector i1_vec = i0_vec + Vector(1.0f);
                          Vector j1_vec = j0_vec + Vector(1.0f);
                          Vector k1_vec = k0_vec + Vector(1.0f);

                          Vector s1 = x - i0_vec;
                          Vector s0 = Vector(1.0f) - s1;
                          Vector t1 = y - j0_vec;
                          Vector t0 = Vector(1.0f) - t1;
                          Vector u1 = z - k0_vec;
                          Vector u0 = Vector(1.0f) - u1;

                          // Prepare to load d0 values for interpolation
                          Vector d000, d001, d010, d011, d100, d101, d110, d111;

                          // Load values for d000 to d111 for the 8 elements in the current vector
                          for (int offset = 0; offset < BLOCK_VECT_SIZE; ++offset) {
                              int i0 = (int)i0_vec[offset], j0 = (int)j0_vec[offset], k0 = (int)k0_vec[offset];
                              int i1 = i0 + 1, j1 = j0 + 1, k1 = k0 + 1;

                              d000[offset] = d0[IX(i0, j0, k0)];
                              d001[offset] = d0[IX(i0, j0, k1)];
                              d010[offset] = d0[IX(i0, j1, k0)];
                              d011[offset] = d0[IX(i0, j1, k1)];
                              d100[offset] = d0[IX(i1, j0, k0)];
                              d101[offset] = d0[IX(i1, j0, k1)];
                              d110[offset] = d0[IX(i1, j1, k0)];
                              d111[offset] = d0[IX(i1, j1, k1)];
                          }

                          // Perform linear interpolation for the 8 elements in the vector
                          Vector d_interp = s0 * (t0 * (u0 * d000 + u1 * d001) +
                                                  t1 * (u0 * d010 + u1 * d011)) +
                                            s1 * (t0 * (u0 * d100 + u1 * d101) +
                                                  t1 * (u0 * d110 + u1 * d111));

                          // Store the result back into d array
                          d_interp.store(&d[idx_base]);
                      }
                  }
              }
            }
        }
    }

    // Set boundaries (this function should be optimized separately if needed)
    set_bnd(M, N, O, b, d);
}

/**
 * @brief Projection step to ensure incompressibility.
 * 
 * This function projects the velocity field to ensure that it is divergence-free.
 * 
 * @param M Size of the grid in the x-dimension.
 * @param N Size of the grid in the y-dimension.
 * @param O Size of the grid in the z-dimension.
 * @param u x-component of the velocity field.
 * @param v y-component of the velocity field.
 * @param w z-component of the velocity field.
 * @param p Pressure field.
 * @param div Divergence field.
 */
void project(int M, int N, int O, float *u, float *v, float *w, float *p,
             float *div) {

  float max = -0.5f / MAX3(M,N,O);

  for (int k = 1; k <= O; k++) {
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= M; i++) {
        div[IX(i, j, k)] = (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] + v[IX(i, j + 1, k)] -
             v[IX(i, j - 1, k)] + w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) * max;
        p[IX(i, j, k)] = 0;
      }
    }
  }

  set_bnd(M, N, O, 0, div);
  set_bnd(M, N, O, 0, p);
  lin_solve(M, N, O, 0, p, div, 1, 6);

  for (int k = 1; k <= O; k++) {
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= M; i++) {
        u[IX(i, j, k)] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
        v[IX(i, j, k)] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
        w[IX(i, j, k)] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
      }
    }
  }
  set_bnd(M, N, O, 1, u);
  set_bnd(M, N, O, 2, v);
  set_bnd(M, N, O, 3, w);
}

/***/
void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v,
               float *w, float diff, float dt) {
  add_source(M, N, O, x, x0, dt);
  SWAP(x0, x);
  diffuse(M, N, O, 0, x, x0, diff, dt);
  SWAP(x0, x);
  advect(M, N, O, 0, x, x0, u, v, w, dt);
}

// Step function for velocity
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0,
              float *v0, float *w0, float visc, float dt) {
  add_source(M, N, O, u, u0, dt);
  add_source(M, N, O, v, v0, dt);
  add_source(M, N, O, w, w0, dt);
  SWAP(u0, u);
  diffuse(M, N, O, 1, u, u0, visc, dt);
  SWAP(v0, v);
  diffuse(M, N, O, 2, v, v0, visc, dt);
  SWAP(w0, w);
  diffuse(M, N, O, 3, w, w0, visc, dt);
  project(M, N, O, u, v, w, u0, v0);
  SWAP(u0, u);
  SWAP(v0, v);
  SWAP(w0, w);
  advect(M, N, O, 1, u, u0, u0, v0, w0, dt);
  advect(M, N, O, 2, v, v0, u0, v0, w0, dt);
  advect(M, N, O, 3, w, w0, u0, v0, w0, dt);
  project(M, N, O, u, v, w, u0, v0);
}
