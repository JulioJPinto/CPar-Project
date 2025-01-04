CPP = g++ -Wall
SRC_SEQ = src/fluid_solver_linear.cpp src/EventManager.cpp src/main_linear.cpp
SRC_PAR = src/fluid_solver_omp.cpp src/EventManager.cpp src/main_omp.cpp
CFLAGS = -Ofast -march=native -ftree-vectorize  
OPENMP = -fopenmp

NPP = nvcc
NSRCS = src/*.cu src/EventManager.cpp
NCFLAGS = -O3 -arch=sm_35 -std=c++11

all: cpu cuda

cpu:
	$(CPP) $(CFLAGS) $(OPENMP) $(SRC_PAR) -o fluid_sim_par
	$(CPP) $(CFLAGS) $(SRC_SEQ)  -o fluid_sim_seq

cuda:
	$(NPP) $(NCFLAGS) $(NSRCS) -o fluid_sim

runseq:
	./fluid_sim_seq

runpar:
	OMP_NUM_THREADS=20 ./fluid_sim

run-cuda:
	./fluid_sim

clean:
	@echo Cleaning up...
	@rm -f fluid_sim gmon.out fluid_sim_*
	@echo Done.

# Compiling for performance testing.

PROF_FLAGS = -pg

prof:
	$(CPP) $(CFLAGS) $(PROF_FLAGS) $(SRCS) -o fluid_sim -lm -o prof_md

run-prof: prof
	./prof_md 

graph-prof: run-prof
	gprof prof_md > main.gprof
	gprof2dot -o output.dot main.gprof
	rm gmon.out
	dot -Tpng -o output.png output.dot

clean-prof:
	@echo Cleaning up...
	@rm -f prof_md gmon.out main.gprof output.dot output.png
	@echo Done.

# Compiling for debugging.

DEBUG_FLAGS = -g

debug:
	$(CPP) $(DEBUG_FLAGS) $(SRCS) -o fluid_sim

run-debug: debug

clean-debug:
	@echo Cleaning up...
	@rm -f fluid_sim
	@echo Done.
