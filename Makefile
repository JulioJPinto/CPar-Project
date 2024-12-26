CPP = g++ -Wall
SRCS = src/fluid_solver.cpp src/EventManager.cpp src/main.cpp
CFLAGS = -Ofast -march=native -ftree-vectorize  
OPENMP = -fopenmp

NPP = nvcc
NSRCS = src/*.cu src/EventManager.cpp
NCFLAGS = -O3 -arch=sm_35

all:
	$(CPP) $(CFLAGS) $(OPENMP) $(SRCS) -o fluid_sim_par
	$(CPP) $(CFLAGS) $(SRCS)  -o fluid_sim_seq

cuda:
	$(NPP) $(NCFLAGS) $(NSRCS) -o fluid_sim

runseq:
	./fluid_sim_seq

runpar:
	OMP_NUM_THREADS=20 ./fluid_sim_par

runcuda:
	./fluid_sim

clean:
	@echo Cleaning up...
	@rm -f fluid_sim gmon.out
	@echo Done.





# Profiling
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
