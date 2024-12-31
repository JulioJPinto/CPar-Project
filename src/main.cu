#include "EventManager.h"
#include "fluid_solver.h"

#include <iostream>
#include <vector>

#define SIZE 84

#define IX(i, j, k) ((i) + (M + 2) * ((j) + (N + 2) * (k)))

#define ALIGNED_ARRAY_FLOAT(s,a) static_cast<float*>(std::aligned_alloc(a, s * sizeof(float)))

// Globals for the grid size
static int M = SIZE;
static int N = SIZE;
static int O = SIZE;
static float dt = 0.1f;      // Time delta
static float diff = 0.0001f; // Diffusion constant
static float visc = 0.0001f; // Viscosity constant

// Fluid simulation arrays
static float *u, *v, *w, *u_prev, *v_prev, *w_prev;
static float *dens, *dens_prev;

// GPU Arrays
static float *d_u, *d_v, *d_w, *d_u_prev, *d_v_prev, *d_w_prev;
static float *d_dens, *d_dens_prev;

// Function to allocate simulation data
int allocate_data() {
  int size = (M + 2) * (N + 2) * (O + 2);
  u = new float[size];
  v = new float[size];
  w = new float[size];
  u_prev = new float[size];
  v_prev = new float[size];
  w_prev = new float[size];
  dens = new float[size];
  dens_prev = new float[size];
  if (!u || !v || !w || !u_prev || !v_prev || !w_prev || !dens || !dens_prev) {
    std::cerr << "Cannot allocate memory" << std::endl;
    return 0;
  }

  // Allocate GPU memory
  cudaMalloc(&d_u, size * sizeof(float));
  cudaMalloc(&d_v, size * sizeof(float));
  cudaMalloc(&d_w, size * sizeof(float));
  cudaMalloc(&d_u_prev, size * sizeof(float));
  cudaMalloc(&d_v_prev, size * sizeof(float));
  cudaMalloc(&d_w_prev, size * sizeof(float));
  cudaMalloc(&d_dens, size * sizeof(float));
  cudaMalloc(&d_dens_prev, size * sizeof(float));

  return 1;
}
// Function to clear the data (set all to zero)
void clear_data() {
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    u[i] = v[i] = w[i] = u_prev[i] = v_prev[i] = w_prev[i] = dens[i] =
        dens_prev[i] = 0.0f;
  }

  // Copy data to GPU
  cudaMemcpy(d_u, u, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, v, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, w, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_u_prev, u_prev, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v_prev, v_prev, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_w_prev, w_prev, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dens, dens, size * sizeof(float), cudaMemcpyHostToDevice); 
  cudaMemcpy(d_dens_prev, dens_prev, size * sizeof(float), cudaMemcpyHostToDevice);

}
// Free allocated memory
void free_data() {
  delete[] u;
  delete[] v;
  delete[] w;
  delete[] u_prev;
  delete[] v_prev;
  delete[] w_prev;
  delete[] dens;
  delete[] dens_prev;

  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_w);
  cudaFree(d_u_prev);
  cudaFree(d_v_prev);
  cudaFree(d_w_prev);
  cudaFree(d_dens);
  cudaFree(d_dens_prev);

}

__global__
void set_dens_kernel(float *d, float density, int index) {
  d[index] = density;
}

__global__
void set_forces_kernel(float *x, float *y, float *z, float fx, float fy, float fz, int index) {
  x[index] = fx;
  y[index] = fy;
  z[index] = fz;
}

// Apply events (source or force) for the current timestep
void apply_events(const std::vector<Event>& events) {
    int i = M / 2, j = N / 2, k = O / 2;
    int index = IX(i, j, k);
    bool density_set = false, force_set = false;
    float density = 0.0f, fx = 0.0f, fy = 0.0f, fz = 0.0f;

    for (const auto& event : events) {
        if (event.type == ADD_SOURCE) {
            density = event.density;
            density_set = true;
        } else if (event.type == APPLY_FORCE) {
            fx = event.force.x;
            fy = event.force.y;
            fz = event.force.z;
            force_set = true;
        }
    }

    if (density_set) {
        set_dens_kernel<<<1, 1>>>(d_dens, density, index);
    }

    if (force_set) {
        set_forces_kernel<<<1, 1>>>(d_u, d_v, d_w, fx, fy, fz, index);
    }
}


// Function to sum the total density
float sum_density() {
  float total_density = 0.0f;
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    total_density += dens[i];
  }
  return total_density;
}

// Simulation loop
void simulate(EventManager &eventManager, int timesteps) {
  for (int t = 0; t < timesteps; t++) {
    // Get the events for the current timestep
    std::vector<Event> events = eventManager.get_events_at_timestamp(t);

    // Apply events to the simulation
    apply_events(events);

    // Perform the simulation steps
    vel_step(M, N, O, d_u, d_v, d_w, d_u_prev, d_v_prev, d_w_prev, visc, dt);
    dens_step(M, N, O, d_dens, d_dens_prev, d_u, d_v, d_w, diff, dt);
  }

  // Copy dens back to CPU
  cudaMemcpy(dens, d_dens, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyDeviceToHost);
}

int main() {
  // Initialize EventManager
  EventManager eventManager;
  eventManager.read_events("events.txt");

  // Get the total number of timesteps from the event file
  int timesteps = eventManager.get_total_timesteps();

  // Allocate and clear data
  if (!allocate_data())
    return -1;
  clear_data();

  // Run simulation with events
  simulate(eventManager, timesteps);

  // Print total density at the end of simulation
  float total_density = sum_density();
  std::cout << "Total density after " << timesteps
            << " timesteps: " << total_density << std::endl;

  // Free memory
  free_data();

  return 0;
}