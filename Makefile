CPP = nvcc 
SRCS = src/*.cpp
CFLAGS = --std=c++17 -O3 

all:
	$(CPP) $(CFLAGS) $(SRCS) -o fluid_sim 
	$(CPP) $(CFLAGS) $(SRCS)  -o fluid_sim_seq

run:
	./fluid_sim

runseq:
	./fluid_sim_seq

runpar:
	OMP_NUM_THREADS=20 ./fluid_sim

clean:
	@echo Cleaning up...
	@rm -f fluid_sim gmon.out fluid_sim_seq
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
