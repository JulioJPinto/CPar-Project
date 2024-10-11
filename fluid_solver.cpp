#include "fluid_solver.h"
#include <cmath>

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
#define SWAP(x0, x)                                                            \
  {                                                                            \
    float *tmp = x0;                                                           \
    x0 = x;                                                                    \
    x = tmp;                                                                   \
  }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define LINEARSOLVERTIMES 20

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

  int loopMN = b == 3 ? -1 : 1;
  // Set boundary on faces
  for (i = 1; i <= M; i++) {
    for (j = 1; j <= N; j++) {
      x[IX(i, j, 0)] = x[IX(i, j, 1)] * loopMN;
      x[IX(i, j, O + 1)] = x[IX(i, j, O)] * loopMN;
    }
  }

  int loopNO = b == 1 ? -1 : 1;

  for (i = 1; i <= N; i++) {
    for (j = 1; j <= O; j++) {
      x[IX(0, i, j)] = x[IX(1, i, j)] * loopNO;
      x[IX(M + 1, i, j)] = x[IX(M, i, j)] * loopNO;
    }
  }

  int loopMO = b == 2 ? -1 : 1;
  for (i = 1; i <= M; i++) {
    for (j = 1; j <= O; j++) {
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
void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {

  for (int l = 0; l < LINEARSOLVERTIMES; l++) {
    for (int k = 1; k <= O; k++) {
      for (int j = 1; j <= N; j++) {

        // Manually unroll the inner loop over `i`
        int i = 1;
        for (; i <= M - 3; i += 4) {
          int idx0 = IX(i, j, k);
          int idx1 = IX(i + 1, j, k);
          int idx2 = IX(i + 2, j, k);
          int idx3 = IX(i + 3, j, k);

          // Cache neighbor indices for idx0, idx1, idx2, and idx3
          int idx0_im1 = IX(i - 1, j, k);
          int idx0_ip1 = IX(i + 1, j, k);
          int idx0_jm1 = IX(i, j - 1, k);
          int idx0_jp1 = IX(i, j + 1, k);
          int idx0_km1 = IX(i, j, k - 1);
          int idx0_kp1 = IX(i, j, k + 1);

          int idx1_im1 = IX(i, j, k);  // For idx1: i-1 is idx0
          int idx1_ip1 = IX(i + 2, j, k);
          int idx1_jm1 = IX(i + 1, j - 1, k);
          int idx1_jp1 = IX(i + 1, j + 1, k);
          int idx1_km1 = IX(i + 1, j, k - 1);
          int idx1_kp1 = IX(i + 1, j, k + 1);

          int idx2_im1 = IX(i + 1, j, k);  // For idx2: i-1 is idx1
          int idx2_ip1 = IX(i + 3, j, k);
          int idx2_jm1 = IX(i + 2, j - 1, k);
          int idx2_jp1 = IX(i + 2, j + 1, k);
          int idx2_km1 = IX(i + 2, j, k - 1);
          int idx2_kp1 = IX(i + 2, j, k + 1);

          int idx3_im1 = IX(i + 2, j, k);  // For idx3: i-1 is idx2
          int idx3_ip1 = IX(i + 4, j, k);  // Only valid if i + 3 < M
          int idx3_jm1 = IX(i + 3, j - 1, k);
          int idx3_jp1 = IX(i + 3, j + 1, k);
          int idx3_km1 = IX(i + 3, j, k - 1);
          int idx3_kp1 = IX(i + 3, j, k + 1);

          // Cache values for x0 at the current indices
          float x0_0 = x0[idx0];  
          float x0_1 = x0[idx1];
          float x0_2 = x0[idx2];
          float x0_3 = x0[idx3];

          // Perform the calculations for unrolled iterations
          x[idx0] = (x0_0 + a * (x[idx0_im1] + x[idx0_ip1] +
                                 x[idx0_jm1] + x[idx0_jp1] +
                                 x[idx0_km1] + x[idx0_kp1])) / c;

          x[idx1] = (x0_1 + a * (x[idx1_im1] + x[idx1_ip1] +
                                 x[idx1_jm1] + x[idx1_jp1] +
                                 x[idx1_km1] + x[idx1_kp1])) / c;

          x[idx2] = (x0_2 + a * (x[idx2_im1] + x[idx2_ip1] +
                                 x[idx2_jm1] + x[idx2_jp1] +
                                 x[idx2_km1] + x[idx2_kp1])) / c;

          x[idx3] = (x0_3 + a * (x[idx3_im1] + x[idx3_ip1] +
                                 x[idx3_jm1] + x[idx3_jp1] +
                                 x[idx3_km1] + x[idx3_kp1])) / c;
        }

        // Handle any remaining iterations if M is not divisible by 4
        for (; i <= M; i++) {
          int idx = IX(i, j, k);
          x[idx] = (x0[idx] +
                    a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                         x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                         x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) / c;
        }
      }
    }
    set_bnd(M, N, O, b, x);
  }
}


// Diffusion step (uses implicit method)
void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff,
             float dt) {
  int max = MAX(MAX(M, N), O);
  float a = dt * diff * max * max;
  lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

// Advection step (uses velocity field to move quantities)
void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v,
            float *w, float dt) {
  float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

  for (int i = 1; i <= M; i++) {
    for (int j = 1; j <= N; j++) {
      for (int k = 1; k <= O; k++) {
        float x = i - dtX * u[IX(i, j, k)];
        float y = j - dtY * v[IX(i, j, k)];
        float z = k - dtZ * w[IX(i, j, k)];

        // Clamp to grid boundaries
        if (x < 0.5f)
          x = 0.5f;
        if (x > M + 0.5f)
          x = M + 0.5f;
        if (y < 0.5f)
          y = 0.5f;
        if (y > N + 0.5f)
          y = N + 0.5f;
        if (z < 0.5f)
          z = 0.5f;
        if (z > O + 0.5f)
          z = O + 0.5f;

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

  float max = MAX(M, MAX(N, O));

  for (int i = 1; i <= M; i++) {
    for (int j = 1; j <= N; j++) {
      for (int k = 1; k <= O; k++) {
        div[IX(i, j, k)] =
            -0.5f *
            (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] + v[IX(i, j + 1, k)] -
             v[IX(i, j - 1, k)] + w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) / max;
        p[IX(i, j, k)] = 0;
      }
    }
  }

  set_bnd(M, N, O, 0, div);
  set_bnd(M, N, O, 0, p);
  lin_solve(M, N, O, 0, p, div, 1, 6);

  for (int i = 1; i <= M; i++) {
    for (int j = 1; j <= N; j++) {
      for (int k = 1; k <= O; k++) {
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
