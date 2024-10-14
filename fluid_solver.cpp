#include "fluid_solver.h"
#include <cmath>
#include <algorithm> // For std::max

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))

#define SWAP(x0, x) { std::swap(x0, x); }
#define MAX3(a, b, c) (std::max({a, b, c}))
#define LINEARSOLVERTIMES 20

// Block size for tiling optimization
#define BLOCK_SIZE 4

// Add sources (density or velocity)
void add_source(int M, int N, int O, float *x, float *s, float dt) {
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    x[i] += dt * s[i];
  }
}

// Set boundary conditions
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

// Linear solve for implicit methods (diffusion)
// Apply blocking (tiling) for cache optimization
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
                x[idx] = (x0[idx] * invC +
                          invA * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                               x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                               x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)]));
              }
            }
          }
        }
      }
    }
    set_bnd(M, N, O, b, x);
  }
}

// Diffusion step (uses implicit method)
void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff, float dt) {
  int max = MAX3(M, N, O);
  float a = dt * diff * max * max;
  lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

// Advection step (uses velocity field to move quantities)
void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v,
            float *w, float dt) {
  float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

  for (int k = 1; k <= O; k++) {
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= M; i++) {
        float x = i - dtX * u[IX(i, j, k)];
        float y = j - dtY * v[IX(i, j, k)];
        float z = k - dtZ * w[IX(i, j, k)];

        // Clamp to grid boundaries
        x = std::max(0.5f, std::min((float)M + 0.5f, x));
        y = std::max(0.5f, std::min((float)N + 0.5f, y));
        z = std::max(0.5f, std::min((float)O + 0.5f, z));

        int i0 = (int)x, i1 = i0 + 1;
        int j0 = (int)y, j1 = j0 + 1;
        int k0 = (int)z, k1 = k0 + 1;

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
  }
  set_bnd(M, N, O, b, d);
}


// Projection step to ensure incompressibility (make the velocity field
// divergence-free)
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

// Step function for density
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
