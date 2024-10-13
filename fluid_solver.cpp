#include "fluid_solver.h"
#include <cmath>
#include <algorithm> // For std::max

#define SWAP(x0, x)                                                            \
  {                                                                            \
    float ***tmp = x0;                                                           \
    x0 = x;                                                                    \
    x = tmp;                                                                   \
  }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define LINEARSOLVERTIMES 20

// Block size for tiling optimization
#define BLOCK_SIZE 4

// Add sources (density or velocity)
void add_source(int M, int N, int O, float ***x, float ***s, float dt) {
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < M + 2; i++) {
    for (int j = 0; j < N + 2; j++) {
      for (int k = 0; k < O + 2; k++) {
        x[i][j][k] += dt * s[i][j][k];
      }
    }
  }
}

// Set boundary conditions
void set_bnd(int M, int N, int O, int b, float ***x) {
  int i, j;

  int loopMN = b == 3 ? -1 : 1;
  // Set boundary on faces
  for (i = 1; i <= M; i++) {
    for (j = 1; j <= N; j++) {
      x[i][ j][ 0] = x[i][ j][ 1] * loopMN;
      x[i][ j][ O + 1] = x[i][ j][ O] * loopMN;
    }
  }

  int loopNO = b == 1 ? -1 : 1;

  for (i = 1; i <= N; i++) {
    for (j = 1; j <= O; j++) {
      x[0][ i][ j] = x[1][ i][ j] * loopNO;
      x[M + 1][ i][ j] = x[M][ i][ j] * loopNO;
    }
  }

  int loopMO = b == 2 ? -1 : 1;
  for (i = 1; i <= M; i++) {
    for (j = 1; j <= O; j++) {
      x[i][ 0][ j] = x[i][ 1][ j] * loopMO;
      x[i][ N + 1][ j] = x[i][ N][ j] * loopMO;
    }
  }

  // Set corners
  x[0][ 0][ 0] = 0.33f * (x[1][ 0][ 0] + x[0][ 1][ 0] + x[0][ 0][ 1]);
  x[M + 1][ 0][ 0] =
      0.33f * (x[M][ 0][ 0] + x[M + 1][ 1][ 0] + x[M + 1][ 0][ 1]);
  x[0][ N + 1][ 0] =
      0.33f * (x[1][ N + 1][ 0] + x[0][ N][ 0] + x[0][ N + 1][ 1]);
  x[M + 1][ N + 1][ 0] = 0.33f * (x[M][ N + 1][ 0] + x[M + 1][ N][ 0] +
                                    x[M + 1][ N + 1][ 1]);
}

// Linear solve for implicit methods (diffusion)
// Apply blocking (tiling) for cache optimization
void lin_solve(int M, int N, int O, int b, float ***x, float ***x0, float a, float c) {

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

                x[i][ j][ k] = (x0[i][ j][ k] * invC +
                          invA * (x[i - 1][ j][ k] + x[i + 1][ j][ k] +
                               x[i][ j - 1][ k] + x[i][ j + 1][ k] +
                               x[i][ j][ k - 1] + x[i][ j][ k + 1]));
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
void diffuse(int M, int N, int O, int b, float ***x, float ***x0, float diff, float dt) {
  int max = MAX(MAX(M, N), O);
  float a = dt * diff * max * max;
  lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

// Advection step (uses velocity field to move quantities)
void advect(int M, int N, int O, int b, float ***d, float ***d0, float ***u, float ***v,
            float ***w, float dt) {
  float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

  for (int i = 1; i <= M; i++) {
    for (int j = 1; j <= N; j++) {
      for (int k = 1; k <= O; k++) {
        float x = i - dtX * u[i][ j][ k];
        float y = j - dtY * v[i][ j][ k];
        float z = k - dtZ * w[i][ j][ k];

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

        d[i][ j][ k] =
            s0 * (t0 * (u0 * d0[i0][ j0][ k0] + u1 * d0[i0][ j0][ k1]) +
                  t1 * (u0 * d0[i0][ j1][ k0] + u1 * d0[i0][ j1][ k1])) +
            s1 * (t0 * (u0 * d0[i1][ j0][ k0] + u1 * d0[i1][ j0][ k1]) +
                  t1 * (u0 * d0[i1][ j1][ k0] + u1 * d0[i1][ j1][ k1]));
      }
    }
  }
  set_bnd(M, N, O, b, d);
}


// Projection step to ensure incompressibility (make the velocity field
// divergence-free)
void project(int M, int N, int O, float ***u, float ***v, float ***w, float ***p,
             float ***div) {

  float max = 1.0f / MAX(M, MAX(N, O));

  for (int i = 1; i <= M; i++) {
    for (int j = 1; j <= N; j++) {
      for (int k = 1; k <= O; k++) {
        div[i][ j][ k] =
            -0.5f *
            (u[i + 1][ j][ k] - u[i - 1][ j][ k] + v[i][ j + 1][ k] -
             v[i][ j - 1][ k] + w[i][ j][ k + 1] - w[i][ j][ k - 1]) * max;
        p[i][ j][ k] = 0;
      }
    }
  }

  set_bnd(M, N, O, 0, div);
  set_bnd(M, N, O, 0, p);
  lin_solve(M, N, O, 0, p, div, 1, 6);

  for (int i = 1; i <= M; i++) {
    for (int j = 1; j <= N; j++) {
      for (int k = 1; k <= O; k++) {
        u[i][ j][ k] -= 0.5f * (p[i + 1][ j][ k] - p[i - 1][ j][ k]);
        v[i][ j][ k] -= 0.5f * (p[i][ j + 1][ k] - p[i][ j - 1][ k]);
        w[i][ j][ k] -= 0.5f * (p[i][ j][ k + 1] - p[i][ j][ k - 1]);
      }
    }
  }
  set_bnd(M, N, O, 1, u);
  set_bnd(M, N, O, 2, v);
  set_bnd(M, N, O, 3, w);
}

// Step function for density
void dens_step(int M, int N, int O, float ***x, float ***x0, float ***u, float ***v,
               float ***w, float diff, float dt) {
  add_source(M, N, O, x, x0, dt);
  SWAP(x0, x);
  diffuse(M, N, O, 0, x, x0, diff, dt);
  SWAP(x0, x);
  advect(M, N, O, 0, x, x0, u, v, w, dt);
}

// Step function for velocity
void vel_step(int M, int N, int O, float ***u, float ***v, float ***w, float ***u0,
              float ***v0, float ***w0, float visc, float dt) {
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
