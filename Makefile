CPP = g++ -Wall
SRCS = src/*.cpp
CFLAGS = -Ofast -march=native -ftree-vectorize  
OPENMP = -fopenmp

all:
	$(CPP) $(CFLAGS) $(OPENMP) $(SRCS) -o fluid_sim 

seq:
	$(CPP) $(CFLAGS) -Wno-pragmas $(SRCS)  -o fluid_sim

run: 
	./fluid_sim

clean:
	@echo Cleaning up...
	@rm -f fluid_sim gmon.out
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
