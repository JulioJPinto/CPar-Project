#include "fluid_solver.h"
#include <stdbool.h>
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
#define BLOCK_SIZE 8


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
__global__
void add_source_kernel(int size, float *x, float *s, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    x[i] += dt * s[i];
  }
}

void add_source(int M, int N, int O, float *x, float *s, float dt) {
  int size = (M + 2) * (N + 2) * (O + 2);
  int block_size = 256;
  int grid_size = (size + block_size - 1) / block_size;
  add_source_kernel<<<grid_size, block_size>>>(size, x, s, dt);
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
void set_bnd_x_kernel(int M, int N, int O, float signal, float *x) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (j <= N && k <= O) {
    x[IX(0, j, k)] = signal * x[IX(1, j, k)];
    x[IX(M + 1, j, k)] = signal * x[IX(M, j, k)];
  }
}

__global__
void set_bnd_y_kernel(int M, int N, int O, float signal, float *x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i <= M && k <= O) {
    x[IX(i, 0, k)] = signal * x[IX(i, 1, k)];
    x[IX(i, N + 1, k)] = signal * x[IX(i, N, k)];
  }
}

__global__
void set_bnd_z_kernel(int M, int N, int O, float signal, float *x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i <= M && j <= N) {
    x[IX(i, j, 0)] = signal * x[IX(i, j, 1)];
    x[IX(i, j, O + 1)] = signal * x[IX(i, j, O)];
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

  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_size_z((M + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y);
  set_bnd_z_kernel<<<grid_size_z, block_size>>>(M, N, O, signal, x);
  

  dim3 grid_size_y((M + block_size.x - 1) / block_size.x, (O + block_size.y - 1) / block_size.y);
  set_bnd_y_kernel<<<grid_size_y, block_size>>>(M, N, O, signal, x);
  

  dim3 grid_size_x((N + block_size.x - 1) / block_size.x, (O + block_size.y - 1) / block_size.y);
  set_bnd_x_kernel<<<grid_size_x, block_size>>>(M, N, O, signal, x);
  

  set_bnd_corners_kernel<<<1, 1>>>(M, N, O, x);

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

#define LIN_SOLVE_TOL 1e-7

__global__ void lin_solve_black(int M, int N, int O, float a, float invC, float* __restrict__ x, const float* __restrict__ x0,bool* __restrict__ done) {
    int j = (blockIdx.y * blockDim.y + threadIdx.y) + 1;
    int k = (blockIdx.z * blockDim.z + threadIdx.z) + 1;
    int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + 1 + (j + k) % 2;

  __shared__ bool local_done;
  if(threadIdx.x==0){
    local_done = true;
    }
  __syncthreads();

  if (i <= M && j <= N && k <= O) {
      int idx = IX(i, j, k);
      float old = x[idx];
      x[idx] = (x0[idx] * invC +
                a * invC * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                            x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                            x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)]));
      if (fabs(x[idx]-old) >  1e-7){
        local_done = false;
      };
  }  
  __syncthreads();
  if (threadIdx.x == 0 && !local_done){
    *done = false;
  }
}

// White cell kernel
__global__ void lin_solve_white(int M, int N, int O, float a, float invC, float* __restrict__ x, const float* __restrict__ x0,bool* __restrict__ done) {

    int j = (blockIdx.y * blockDim.y + threadIdx.y) + 1;
    int k = (blockIdx.z * blockDim.z + threadIdx.z) + 1;
    int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + 1 + (j + k + 1  ) % 2;


__shared__ bool local_done;
  if(threadIdx.x==0){
    local_done = true;
  }
  __syncthreads();

    if (i <= M && j <= N && k <= O) {
      int idx = IX(i, j, k);
      float old = x[idx];
      x[idx] = (x0[idx] * invC +
                a * invC * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                            x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                            x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)]));
      if (fabs(x[idx]-old) > 1e-7){
        local_done = false;
      };
  }
  __syncthreads();
  if (threadIdx.x == 0 && !local_done){
    *done = false;
  }
}

void lin_solve(int M, int N, int O, int b, float* x, float* x0, float a, float c) {
  float invC = 1.0f / c;
  bool *dev_done;
  cudaMalloc(&dev_done,sizeof(bool));
  bool done = true;
  cudaMemcpy(dev_done, &done, sizeof(bool), cudaMemcpyHostToDevice);
  done = false;
  

  // Set boundary conditions
  // Tunable thread and block dimensions
    dim3 blockDim(8, 8, 8);  // Threads per block
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x,  // Blocks along X
                 (N + blockDim.y - 1) / blockDim.y,  // Blocks along Y
                 (O + blockDim.z - 1) / blockDim.z); // Blocks along Z
  for (int iter = 0; iter < LINEARSOLVERTIMES && !done; ++iter) {
      // Launch black cell kernel
      done = true;
      cudaMemcpy(dev_done, &done, sizeof(bool), cudaMemcpyHostToDevice);

      lin_solve_black<<<gridDim, blockDim>>>(M, N, O, a, invC, x, x0,dev_done);
      
      // Launch white cell kernel
      lin_solve_white<<<gridDim, blockDim>>>(M, N, O, a, invC, x, x0,dev_done);
      

      cudaMemcpy(&done, dev_done, sizeof(bool), cudaMemcpyDeviceToHost);

      // Synchronize boundary conditions
      set_bnd(M, N, O, b, x);
  }
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
__global__
void advect_kernel(int M, int N, int O, float dtX, float dtY, float dtZ, float *d, float *d0, float *u, float *v, float *w) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  if (i < M && j < N && k < O) {
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
    float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((M + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y, (O + block_size.z - 1) / block_size.z);

    advect_kernel<<<grid_size, block_size>>>(M, N, O, dtX, dtY, dtZ, d, d0, u, v, w);
    
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
void div_kernel(int M, int N, int O, float *u, float *v, float *w, float *div, float max) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  if (i < M && j < N && k < O) {
    div[IX(i, j, k)] = (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] + v[IX(i, j + 1, k)] -
                        v[IX(i, j - 1, k)] + w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) * max;
  }
}

  
__global__
void update_velocity_kernel(int M, int N, int O, float *u, float *v, float *w, float *p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  if (i <= M && j <= N && k <= O) {
    u[IX(i, j, k)] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
    v[IX(i, j, k)] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
    w[IX(i, j, k)] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
  }
}


void project(int M, int N, int O, float *u, float *v, float *w, float *p,
             float *div) {

  float max = -0.5f / MAX3(M,N,O);

  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_size((M + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y, (O + block_size.z - 1) / block_size.z);

  div_kernel<<<grid_size, block_size>>>(M, N, O, u, v, w, div, max);
  

  set_bnd(M, N, O, 0, div);
  set_bnd(M, N, O, 0, p);
  lin_solve(M, N, O, 0, p, div, 1, 6);
  
  update_velocity_kernel<<<grid_size, block_size>>>(M, N, O, u, v, w, p);
  

  set_bnd(M, N, O, 1, u);
  set_bnd(M, N, O, 2, v);
  set_bnd(M, N, O, 3, w);
}

/***/
void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v,
               float *w, float diff, float dt) {
  add_source(M, N, O, x, x0, dt);
  diffuse(M, N, O, 0, x0, x, diff, dt);
  advect(M, N, O, 0, x, x0, u, v, w, dt);
}

// Step function for velocity
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0,
              float *v0, float *w0, float visc, float dt) {
  add_source(M, N, O, u, u0, dt);
  add_source(M, N, O, v, v0, dt);
  add_source(M, N, O, w, w0, dt);

  diffuse(M, N, O, 1, u0, u, visc, dt);

  diffuse(M, N, O, 2, v0, v, visc, dt);
 
  diffuse(M, N, O, 3, w0, w, visc, dt);
  project(M, N, O, u0, v0, w0, u, v);

  advect(M, N, O, 1, u, u0, u0, v0, w0, dt);
  advect(M, N, O, 2, v, v0, u0, v0, w0, dt);
  advect(M, N, O, 3, w, w0, u0, v0, w0, dt);
  project(M, N, O, u, v, w, u0, v0);
}
