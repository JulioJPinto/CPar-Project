#include "EventManager.h"
#include "fluid_solver.h"

#include <cuda_runtime.h>
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

//Fluid GPU simulation arryas
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

  // Allocate memory on the GPU
  cudaMalloc(&d_u, size * sizeof(float));
  cudaMalloc(&d_v, size * sizeof(float));
  cudaMalloc(&d_w, size * sizeof(float));
  cudaMalloc(&d_u_prev, size * sizeof(float));
  cudaMalloc(&d_v_prev, size * sizeof(float));
  cudaMalloc(&d_w_prev, size * sizeof(float));
  cudaMalloc(&d_dens, size * sizeof(float));
  cudaMalloc(&d_dens_prev, size * sizeof(float));

  if(!d_u || !d_v || !d_w || !d_u_prev || !d_v_prev || !d_w_prev || !d_dens || !d_dens_prev) {
    std::cerr << "Cannot allocate memory on the GPU" << std::endl;
    return 0;
  }

  return 1;
}
// Function to clear the data (set all to zero)
void clear_data() {
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    u[i] = v[i] = w[i] = u_prev[i] = v_prev[i] = w_prev[i] = dens[i] =
        dens_prev[i] = 0.0f;
  }

  // Copy data to the GPU
  cudaMemcpy(d_u, 0, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, 0, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, 0, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_u_prev, 0, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v_prev, 0, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_w_prev, 0, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dens, 0, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dens_prev, 0, size * sizeof(float), cudaMemcpyHostToDevice);

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

// Apply events (source or force) for the current timestep
void apply_events(const std::vector<Event>& events) {
    int i = M / 2, j = N / 2, k = O / 2;
    int index = IX(i, j, k); 

    for (const auto& event : events) {
        if (event.type == ADD_SOURCE) {
            dens[index] = event.density;
        } else if (event.type == APPLY_FORCE) {
            u[index] = event.force.x;
            v[index] = event.force.y;
            w[index] = event.force.z;
        }
        std::cout << "Event at timestep " << event.type << " applied " << std::endl;
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

    //Copy data to GPU
    cudaMemcpy(d_u, u, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_prev, u_prev, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_prev, v_prev, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_prev, w_prev, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dens, dens, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dens_prev, dens_prev, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyHostToDevice);

    // Perform the simulation steps
    vel_step(M, N, O, d_u, d_v, d_w, d_u_prev, d_v_prev, d_w_prev, visc, dt);
    dens_step(M, N, O, d_dens, d_dens_prev, d_u, d_v, d_w, diff, dt);

    //Copy data back to CPU
    cudaMemcpy(u, d_u, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v, d_v, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(w, d_w, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(u_prev, d_u_prev, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_prev, d_v_prev, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(w_prev, d_w_prev, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dens, d_dens, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dens_prev, d_dens_prev, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyDeviceToHost);
  }
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