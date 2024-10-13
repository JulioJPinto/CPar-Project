#include "EventManager.h"
#include "fluid_solver.h"
#include <iostream>
#include <vector>

#define SIZE 42

// Globals for the grid size
static int M = SIZE;
static int N = SIZE;
static int O = SIZE;
static float dt = 0.1f;      // Time delta
static float diff = 0.0001f; // Diffusion constant
static float visc = 0.0001f; // Viscosity constant

// Fluid simulation arrays
static float ***u, ***v, ***w, ***u_prev, ***v_prev, ***w_prev;
static float ***dens, ***dens_prev;

float*** malloc3d(int x, int y, int z) {
  float ***array = new float**[x];
  for (int i = 0; i < x; i++) {
    array[i] = new float*[y];
    for (int j = 0; j < y; j++) {
      array[i][j] = new float[z];
    }
  }
  return array;
}

void dealloc3d(float ***array, int x, int y) {
  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      delete[] array[i][j];
    }
    delete[] array[i];
  }
  delete[] array;
}

// Function to allocate simulation data
int allocate_data() {

  u = malloc3d(M + 2, N + 2, O + 2);
  v = malloc3d(M + 2, N + 2, O + 2);
  w = malloc3d(M + 2, N + 2, O + 2);
  u_prev = malloc3d(M + 2, N + 2, O + 2);
  v_prev = malloc3d(M + 2, N + 2, O + 2);
  w_prev = malloc3d(M + 2, N + 2, O + 2);
  dens = malloc3d(M + 2, N + 2, O + 2);
  dens_prev = malloc3d(M + 2, N + 2, O + 2);

  if (!u || !v || !w || !u_prev || !v_prev || !w_prev || !dens || !dens_prev) {
    std::cerr << "Cannot allocate memory" << std::endl;
    return 0;
  }
  return 1;
}

// Function to clear the data (set all to zero)
void clear_data() {
  for (int i = 0; i < M + 2; i++) {
    for (int j = 0; j < N + 2; j++) {
      for (int k = 0; k < O + 2; k++) {
        u[i][j][k] = v[i][j][k] = w[i][j][k] = u_prev[i][j][k] =
            v_prev[i][j][k] = w_prev[i][j][k] = dens[i][j][k] =
                dens_prev[i][j][k] = 0.0f;
      }
    }
  }
}

// Free allocated memory
void free_data() {
  dealloc3d(u, M + 2, N + 2);
  dealloc3d(v, M + 2, N + 2);
  dealloc3d(w, M + 2, N + 2);
  dealloc3d(u_prev, M + 2, N + 2);
  dealloc3d(v_prev, M + 2, N + 2);
  dealloc3d(w_prev, M + 2, N + 2);
  dealloc3d(dens, M + 2, N + 2);
  dealloc3d(dens_prev, M + 2, N + 2);
}

// Apply events (source or force) for the current timestep
void apply_events(const std::vector<Event> &events) {
  for (const auto &event : events) {
    if (event.type == ADD_SOURCE) {
      // Apply density source at the center of the grid
      int i = M / 2, j = N / 2, k = O / 2;
      dens[i][ j][ k] = event.density;
    } else if (event.type == APPLY_FORCE) {
      // Apply forces based on the event's vector (fx, fy, fz)
      int i = M / 2, j = N / 2, k = O / 2;
      u[i][ j][ k] = event.force.x;
      v[i][ j][ k] = event.force.y;
      w[i][ j][ k] = event.force.z;
    }
  }
}

// Function to sum the total density
float sum_density() {
  float total_density = 0.0f;
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < M + 2; i++) {
    for (int j = 0; j < N + 2; j++) {
      for (int k = 0; k < O + 2; k++) {
        total_density += dens[i][j][k];
  }}}
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
    vel_step(M, N, O, u, v, w, u_prev, v_prev, w_prev, visc, dt);
    dens_step(M, N, O, dens, dens_prev, u, v, w, diff, dt);
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