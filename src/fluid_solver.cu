#include "fluid_solver.h"
#include <cuda_runtime.h>
#include <omp.h>
#include <cmath>
#include <iostream>
#include <algorithm> // For std::max

#define IX(i, j, k) ((i) + (M + 2) * ((j) + (N + 2) * (k)))

#define SWAP(x0, x) \
  {                \
    float *tmp = x0; \
    x0 = x;        \
    x = tmp;       \
  }

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX3(a, b, c) MAX(MAX(a, b), c)
#define LINEARSOLVERTIMES 20


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
__global__ void add_source_kernel(int size, float *x, float *s, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    x[i] += dt * s[i];
  }
}

void add_source(int M, int N, int O, float *x, float *s, float dt) {
  int size = (M + 2) * (N + 2) * (O + 2);
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  add_source_kernel<<<blocksPerGrid, threadsPerBlock>>>(size, x, s, dt);
  cudaDeviceSynchronize();
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
// Set boundary conditions

__global__
void set_bnd_z_kernel(int M, int N, int O, float x_signal, float *x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= 1 && i <= M && j >= 1 && j <= N) {
    x[IX(i, j, 0)] = x_signal * x[IX(i, j, 1)];
    x[IX(i, j, O + 1)] = x_signal * x[IX(i, j, O)];
  }
}

__global__
void set_bnd_y_kernel(int M, int N, int O, float x_signal, float *x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= 1 && i <= M && k >= 1 && k <= O) {
    x[IX(i, 0, k)] = x_signal * x[IX(i, 1, k)];
    x[IX(i, N + 1, k)] = x_signal * x[IX(i, N, k)];
  }
}

__global__
void set_bnd_x_kernel(int M, int N, int O, float x_signal, float *x) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  if (j >= 1 && j <= N && k >= 1 && k <= O) {
    x[IX(0, j, k)] = x_signal * x[IX(1, j, k)];
    x[IX(M + 1, j, k)] = x_signal * x[IX(M, j, k)];
  }
}

__global__
void set_bnd_corners_kernel(int M, int N, int O, float *x) {
  x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
  x[IX(M + 1, 0, 0)] = 0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
  x[IX(0, N + 1, 0)] = 0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
  x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] + x[IX(M + 1, N + 1, 1)]);
}

void set_bnd(int M, int N, int O, int b, float *x) {
  float signal = (b == 3 || b == 1 || b == 2) ? -1.0f : 1.0f;

  dim3 block_size(8, 8);
  dim3 grid_size((M + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y);

  set_bnd_z_kernel<<<grid_size, block_size>>>(M, N, O, signal, x);
  set_bnd_y_kernel<<<grid_size, block_size>>>(M, N, O, signal, x);
  set_bnd_x_kernel<<<grid_size, block_size>>>(M, N, O, signal, x);
  set_bnd_corners_kernel<<<1, 1>>>(M, N, O, x);
  cudaDeviceSynchronize();
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
__global__
void white_lin_solve_kernel(int M, int N, int O, float *x, float *x0, float a, float c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= 1 && i <= M && j >= 1 && j <= N && k >= 1 && k <= O && (i + j + k) % 2 == 1) {
    x[IX(i, j, k)] = (x0[IX(i, j, k)] * c +
                      a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                           x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                           x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)]));
  }
}

__global__
void black_lin_solve_kernel(int M, int N, int O, float *x, float *x0, float a, float c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= 1 && i <= M && j >= 1 && j <= N && k >= 1 && k <= O && (i + j + k) % 2 == 0) {
    x[IX(i, j, k)] = (x0[IX(i, j, k)] * c +
                      a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                           x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                           x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)]));
  }
}

void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
  //float tol = 1e-7, max_c, old_x, change;
  int l = 0;

  float invC = 1.0f / c;
  float invA = a * invC;

  do {
    //max_c = 0.0f;
    // First half of the sweep (black cells)
    dim3 block_size(8, 8, 8);
    dim3 grid_size((M + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y, (O + block_size.z - 1) / block_size.z);
    black_lin_solve_kernel<<<grid_size, block_size>>>(M, N, O, x, x0, a, c);
    cudaDeviceSynchronize();

    // Second half of the sweep (white cells)
    white_lin_solve_kernel<<<grid_size, block_size>>>(M, N, O, x, x0, a, c);
    cudaDeviceSynchronize();

    // Synchronize boundary conditions
    set_bnd(M, N, O, b, x);

  } while (++l < LINEARSOLVERTIMES);
}

/**
 * @brief Diffusion step for the fluid simulation.
 * 
 * This function simulates the diffusion of quantities (e.g., velocity or density)
 * using an implicit method.
 * 
 * @param M Size of th´e grid in the x-dimension.
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
__global__ void advect_kernel(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= 1 && i <= M && j >= 1 && j <= N && k >= 1 && k <= O) {
    float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

    float x = i - dtX * u[IX(i, j, k)];
    float y = j - dtY * v[IX(i, j, k)];
    float z = k - dtZ * w[IX(i, j, k)];

    // Clamp to grid boundaries
    x = fmaxf(0.5f, fminf(M + 0.5f, x));
    y = fmaxf(0.5f, fminf(N + 0.5f, y));
    z = fmaxf(0.5f, fminf(O + 0.5f, z));

    int i0 = (int) x, i1 = i0 + 1;
    int j0 = (int) y, j1 = j0 + 1;
    int k0 = (int) z, k1 = k0 + 1;

    float s1 = x - i0, s0 = 1 - s1;
    float t1 = y - j0, t0 = 1 - t1;
    float u1 = z - k0, u0 = 1 - u1;

    d[IX(i, j, k)] =
        s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
            t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
        s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
            t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
  }
}

void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
  dim3 blockSize(8, 8, 8);
  dim3 gridSize((M + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y, (O + blockSize.z - 1) / blockSize.z);

  advect_kernel<<<gridSize, blockSize>>>(M, N, O, b, d, d0, u, v, w, dt);

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
 __global__
 void compute_div_and_reset_p(int M, int N, int O, float *u, float *v, float *w, float *p, float *div, float max) {
     int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
     int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
     int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
 
     if (i <= M && j <= N && k <= O) {
         div[IX(i, j, k)] = (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] +
                             v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)] +
                             w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) * max;
         p[IX(i, j, k)] = 0.0f;
     }
 }
 
 __global__
 void update_velocity(int M, int N, int O, float *u, float *v, float *w, float *p) {
     int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
     int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
     int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
 
     if (i <= M && j <= N && k <= O) {
         u[IX(i, j, k)] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
         v[IX(i, j, k)] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
         w[IX(i, j, k)] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
     }
 }
 
 void project(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {
     float max = -0.5f / MAX3(M, N, O);
 
     dim3 block_size(8, 8, 8);
     dim3 grid_size((M + block_size.x - 1) / block_size.x, 
                    (N + block_size.y - 1) / block_size.y, 
                    (O + block_size.z - 1) / block_size.z);
 
     // Compute divergence and reset pressure
     compute_div_and_reset_p<<<grid_size, block_size>>>(M, N, O, u, v, w, p, div, max);
 
     // Apply boundary conditions
     set_bnd(M, N, O, 0, div);
     set_bnd(M, N, O, 0, p);
 
     // Solve linear system
     lin_solve(M, N, O, 0, p, div, 1, 6);
 
     // Update velocities
     update_velocity<<<grid_size, block_size>>>(M, N, O, u, v, w, p);
 
     // Apply boundary conditions for velocities
     set_bnd(M, N, O, 1, u);
     set_bnd(M, N, O, 2, v);
     set_bnd(M, N, O, 3, w);
 }

/***/
void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v,
               float *w, float diff, float dt) {
                  
  add_source(M, N, O, x, x0, dt);
  //SWAP(x0, x);
  diffuse(M, N, O, 0, x0, x, diff, dt);
  //SWAP(x0, x);
  advect(M, N, O, 0, x, x0, u, v, w, dt);
}

// Step function for velocity
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0,
              float *v0, float *w0, float visc, float dt) {

  // Add the source term
  add_source(M, N, O, u, u0, dt);
  add_source(M, N, O, v, v0, dt);
  add_source(M, N, O, w, w0, dt);
  
  //SWAP(u0, u);
  diffuse(M, N, O, 1, u0, u, visc, dt);
  //SWAP(v0, v
  diffuse(M, N, O, 2, v0, v, visc, dt);
  //SWAP(w0, w
  diffuse(M, N, O, 3, w0, w, visc, dt);
  project(M, N, O, u0, v0, w0, u, v);

  //SWAP(u0, u);
  //SWAP(v0, v);
  //SWAP(w0, w);
  advect(M, N, O, 1, u, u0, u0, v0, w0, dt);
  advect(M, N, O, 2, v, v0, u0, v0, w0, dt);
  advect(M, N, O, 3, w, w0, u0, v0, w0, dt);
  project(M, N, O, u, v, w, u0, v0);

}
