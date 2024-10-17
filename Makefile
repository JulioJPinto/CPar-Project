CPP = g++ -Wall -pg
SRCS = main.cpp fluid_solver.cpp EventManager.cpp
CFLAGS = -O3 -Ofast -march=native -ftree-vectorize -mavx -Wall
CFLAGS-O = -O3 -march=native -ftree-vectorize -mavx -Wall
C-FLAGS-NO = -O0 -g -Wall

no-opt:
	$(CPP) $(C-FLAGS-NO) $(SRCS) -o fluid_sim -lm

simple:
	$(CPP) $(CFLAGS-O) $(SRCS) -o fluid_sim -lm

all:
	$(CPP) $(CFLAGS) $(SRCS) -o fluid_sim

clean:
	@echo Cleaning up...
	@rm -f fluid_sim gmon.out
	@echo Done.

run: 
	./fluid_sim

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
