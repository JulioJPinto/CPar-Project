#include "fluid_solver.h"
#include <stdbool.h>
#include <cmath>
#include <iostream>
#include <algorithm> 

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



__global__
void set_bnd_x_kernel(int M, int N, int O, float signal, float *x) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (j <= N && k <= O) {
    x[IX(0, j, k)]     = signal * x[IX(1, j, k)];
    x[IX(M + 1, j, k)] = signal * x[IX(M, j, k)];
  }
}

__global__
void set_bnd_y_kernel(int M, int N, int O, float signal, float *x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i <= M && k <= O) {
    x[IX(i, 0, k)]     = signal * x[IX(i, 1, k)];
    x[IX(i, N + 1, k)] = signal * x[IX(i, N, k)];
  }
}

__global__
void set_bnd_z_kernel(int M, int N, int O, float signal, float *x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i <= M && j <= N) {
    x[IX(i, j, 0)]     = signal * x[IX(i, j, 1)];
    x[IX(i, j, O + 1)] = signal * x[IX(i, j, O)];
  }
}

__global__ 
void set_bnd_corners_kernel(int M, int N, int O, float *x) {
  x[IX(0, 0, 0)]             = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
  x[IX(M + 1, 0, 0)]         = 0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
  x[IX(0, N + 1, 0)]         = 0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
  x[IX(M + 1, N + 1, 0)]     = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] + x[IX(M + 1, N + 1, 1)]);
}

void set_bnd(int M, int N, int O, int b, float *x) {
  float signal = (b == 3 || b == 1 || b == 2) ? -1.0f : 1.0f;

  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_size_z((M + block_size.x - 1) / block_size.x,
                   (N + block_size.y - 1) / block_size.y);
  set_bnd_z_kernel<<<grid_size_z, block_size>>>(M, N, O, signal, x);

  dim3 grid_size_y((M + block_size.x - 1) / block_size.x,
                   (O + block_size.y - 1) / block_size.y);
  set_bnd_y_kernel<<<grid_size_y, block_size>>>(M, N, O, signal, x);

  dim3 grid_size_x((N + block_size.x - 1) / block_size.x,
                   (O + block_size.y - 1) / block_size.y);
  set_bnd_x_kernel<<<grid_size_x, block_size>>>(M, N, O, signal, x);

  set_bnd_corners_kernel<<<1, 1>>>(M, N, O, x);
}

#define LIN_SOLVE_TOL 1e-7

__global__ 
void lin_solve_black(int M, int N, int O, float a, float invC,
                     float* __restrict__ x, const float* __restrict__ x0,
                     bool* __restrict__ done)
{
  int j = (blockIdx.y * blockDim.y + threadIdx.y) + 1;
  int k = (blockIdx.z * blockDim.z + threadIdx.z) + 1;
  int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + 1 + ((j + k) % 2);

  if (i <= M && j <= N && k <= O) {
    int idx   = IX(i, j, k);
    float old = x[idx];
    x[idx]    = (x0[idx] * invC +
                 a * invC * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                             x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                             x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)]));
    if (fabsf(x[idx] - old) > LIN_SOLVE_TOL) {
      *done = false;
    }
  }
}

__global__ 
void lin_solve_white(int M, int N, int O, float a, float invC,
                     float* __restrict__ x, const float* __restrict__ x0,
                     bool* __restrict__ done)
{
  int j = (blockIdx.y * blockDim.y + threadIdx.y) + 1;
  int k = (blockIdx.z * blockDim.z + threadIdx.z) + 1;
  int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + 1 + ((j + k + 1) % 2);

  if (i <= M && j <= N && k <= O) {
    int idx   = IX(i, j, k);
    float old = x[idx];
    x[idx]    = (x0[idx] * invC +
                 a * invC * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                             x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                             x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)]));
    if (fabsf(x[idx] - old) > LIN_SOLVE_TOL) {
      *done = false;
    }
  }
}

void lin_solve(int M, int N, int O, int b,
               float* x, float* x0,
               float a, float c)
{
  float invC = 1.0f / c;
  bool done  = false;
  bool *dev_done;
  cudaMalloc(&dev_done, sizeof(bool));

  dim3 blockDim(16, 4, 4);
  dim3 gridDim((M/2 + blockDim.x - 1) / blockDim.x,
               (N + blockDim.y - 1) / blockDim.y,
               (O + blockDim.z - 1) / blockDim.z);

  for (int iter = 0; iter < LINEARSOLVERTIMES && !done; ++iter) {
    done = true; 
    cudaMemset(dev_done, 1, sizeof(bool));

    lin_solve_black<<<gridDim, blockDim>>>(M, N, O, a, invC, x, x0, dev_done);
    
    lin_solve_white<<<gridDim, blockDim>>>(M, N, O, a, invC, x, x0, dev_done);

    cudaMemcpy(&done, dev_done, sizeof(bool), cudaMemcpyDeviceToHost);
    set_bnd(M, N, O, b, x);
  }

  cudaFree(dev_done);
}


void diffuse(int M, int N, int O, int b,
             float *x, float *x0,
             float diff, float dt)
{
  int maxDim = MAX3(M, N, O);
  float a    = dt * diff * (maxDim * maxDim);

  lin_solve(M, N, O, b, x, x0, a, 1.0f + 6.0f * a);
}


__global__
void advect_kernel(int M, int N, int O,
                   float dtX, float dtY, float dtZ,
                   float *d, float *d0,
                   float *u, float *v, float *w)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  if (i <= M && j <= N && k <= O) {
    float x = i - dtX * u[IX(i, j, k)];
    float y = j - dtY * v[IX(i, j, k)];
    float z = k - dtZ * w[IX(i, j, k)];
 
    x = fmaxf(0.5f, fminf(M + 0.5f, x));
    y = fmaxf(0.5f, fminf(N + 0.5f, y));
    z = fmaxf(0.5f, fminf(O + 0.5f, z));

    int i0 = (int)x; 
    int i1 = i0 + 1;
    int j0 = (int)y; 
    int j1 = j0 + 1;
    int k0 = (int)z; 
    int k1 = k0 + 1;

    float s1 = x - i0; 
    float s0 = 1 - s1;
    float t1 = y - j0; 
    float t0 = 1 - t1;
    float u1 = z - k0; 
    float u0 = 1 - u1;

    d[IX(i, j, k)] =
      s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
            t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
      s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
            t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
  }
}

void advect(int M, int N, int O, int b,
            float *d, float *d0,
            float *u, float *v, float *w,
            float dt)
{
  float dtX = dt * M;
  float dtY = dt * N;
  float dtZ = dt * O;

  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_size((M + block_size.x - 1) / block_size.x,
                 (N + block_size.y - 1) / block_size.y,
                 (O + block_size.z - 1) / block_size.z);

  advect_kernel<<<grid_size, block_size>>>(M, N, O, dtX, dtY, dtZ, d, d0, u, v, w);
  set_bnd(M, N, O, b, d);
}

__global__
void div_kernel(int M, int N, int O,
                float *u, float *v, float *w,
                float *div, float scale)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  if (i <= M && j <= N && k <= O) {
    div[IX(i, j, k)] =
      (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] +
       v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)] +
       w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) * scale;
  }
}

__global__
void update_velocity_kernel(int M, int N, int O,
                           float *u, float *v, float *w,
                           float *p)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  if (i <= M && j <= N && k <= O) {
    u[IX(i, j, k)] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
    v[IX(i, j, k)] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
    w[IX(i, j, k)] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
  }
}

void project(int M, int N, int O,
             float *u, float *v, float *w,
             float *p, float *div)
{
  float scale = -0.5f / (float)(MAX3(M, N, O));

  {
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((M + block_size.x - 1) / block_size.x,
                   (N + block_size.y - 1) / block_size.y,
                   (O + block_size.z - 1) / block_size.z);

    div_kernel<<<grid_size, block_size>>>(M, N, O, u, v, w, div, scale);

    cudaMemset(p, 0, sizeof(float) * (M + 2) * (N + 2) * (O + 2));

    set_bnd(M, N, O, 0, div);
    set_bnd(M, N, O, 0, p);

    lin_solve(M, N, O, 0, p, div, 1.0f, 6.0f);
 
    update_velocity_kernel<<<grid_size, block_size>>>(M, N, O, u, v, w, p);
  }

  set_bnd(M, N, O, 1, u);
  set_bnd(M, N, O, 2, v);
  set_bnd(M, N, O, 3, w);
}


void dens_step(int M, int N, int O,
               float *x, float *x0,
               float *u, float *v, float *w,
               float diff, float dt)
{

  add_source(M, N, O, x, x0, dt);

  SWAP(x0, x);
  diffuse(M, N, O, 0, x, x0, diff, dt);

  SWAP(x0, x);
  advect(M, N, O, 0, x, x0, u, v, w, dt);
}

void vel_step(int M, int N, int O,
              float *u, float *v, float *w,
              float *u0, float *v0, float *w0,
              float visc, float dt)
{
  
  add_source(M, N, O, u,  u0, dt);
  add_source(M, N, O, v,  v0, dt);
  add_source(M, N, O, w,  w0, dt);

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
