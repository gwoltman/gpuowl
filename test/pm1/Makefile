# Checks of src/Pm1.cpp and of the GPU GCD (src/Gcd.cpp) against GMP.  GMP is used here only as an independent oracle: it is not a
# dependency of prpll (nothing in src/ includes or links it).
#
#   make        build and run check_pm1
#   make MPRIME_SRC=~/mprime   also compare the mprime save file names with Prime95's own tempFileName (commonc.c)
#   make check_gcd   build the GCD checker; it links prpll's objects, so build prpll first (make in the top directory).
#                    Run it from a scratch directory (it creates prpll's default folders): check_gcd -device N [-skew K] [-time p,...]

SRC = ../../src
OBJS = $(SRC)/Pm1.cpp $(SRC)/CycleFile.cpp $(SRC)/fs.cpp $(SRC)/File.cpp $(SRC)/log.cpp $(SRC)/common.cpp $(SRC)/timeutil.cpp
BIN = ../../build-release
PRPLL_OBJS = $(filter-out $(BIN)/main.o,$(wildcard $(BIN)/*.o))

check: check_pm1
	./check_pm1

stage1_oracle: stage1_oracle.cpp
	g++ -std=c++20 -O2 -Wall -Wextra -o $@ stage1_oracle.cpp -lgmp

check_pm1: check_pm1.cpp $(OBJS) $(SRC)/Pm1.h
	g++ -std=c++20 -O2 -Wall -Wextra -I$(SRC) -DMPRIME_SRC='"$(MPRIME_SRC)"' -o $@ check_pm1.cpp $(OBJS) -lgmp

p1flip: p1flip.cpp $(OBJS)
	g++ -std=c++20 -O2 -Wall -Wextra -I$(SRC) -o $@ p1flip.cpp $(OBJS)

check_gcd: check_gcd.cpp $(PRPLL_OBJS) $(SRC)/Gcd.h
	g++ -std=c++20 -O2 -flto -Wall -Wextra -I$(SRC) -o $@ check_gcd.cpp $(PRPLL_OBJS) -lgmp -lOpenCL -lpthread

# Stage 2 oracle (see check_stage2.cpp and run_stage2.sh)
check_stage2: check_stage2.cpp $(OBJS) $(SRC)/Pm1.h
	g++ -std=c++20 -O2 -Wall -Wextra -I$(SRC) -o $@ check_stage2.cpp $(OBJS) -lgmp -pthread

# Tune file round trips (no GPU; links prpll's objects).  check_tune_base:  the same against another build's objects (BASE=<its
# build-release directory>, PRP part only), to compare tune.txt round trips between the two builds.
check_tune: check_tune.cpp $(PRPLL_OBJS) $(SRC)/TuneEntry.h
	g++ -std=c++20 -O1 -Wall -Wextra -I$(SRC) -I$(BIN) -o $@ check_tune.cpp $(PRPLL_OBJS) -lOpenCL -lpthread

check_tune_base: check_tune.cpp
	g++ -std=c++20 -O1 -Wall -Wextra -DOLD_API -I$(BASE)/../src -I$(BASE) -o $@ check_tune.cpp $(filter-out $(BASE)/main.o,$(wildcard $(BASE)/*.o)) -lOpenCL -lpthread

# The Pfactor bounds model (no GPU).  With MPRIME_SRC, mprime's pm1prob.c (an independent copy of the probability model) is the oracle:
# it is compiled in pm1prob-build/ with a stub for mprime's common.h; nothing of it is part of prpll.
ifneq ($(MPRIME_SRC),)
PM1PROB = pm1prob-build/pm1prob.o
PM1PROB_FLAGS = -DHAVE_PM1PROB
endif

pm1prob-build/pm1prob.o: $(MPRIME_SRC)/pm1prob.c
	mkdir -p pm1prob-build
	cp $(MPRIME_SRC)/pm1prob.c $(MPRIME_SRC)/pm1prob.h pm1prob-build/
	printf '#include <stdint.h>\n#include <math.h>\n#include <assert.h>\n#define ASSERTG assert\n' > pm1prob-build/common.h
	gcc -O2 -c -o $@ pm1prob-build/pm1prob.c

check_bounds: check_bounds.cpp $(OBJS) $(SRC)/Pm1Bounds.cpp $(SRC)/Pm1Bounds.h $(SRC)/Pm1.h $(PM1PROB)
	g++ -std=c++20 -O2 -Wall -Wextra -I$(SRC) $(PM1PROB_FLAGS) -o $@ check_bounds.cpp $(SRC)/Pm1Bounds.cpp $(OBJS) $(PM1PROB)

# AllocTrac's per-device accounting (no GPU)
check_alloc: check_alloc.cpp $(SRC)/AllocTrac.cpp $(SRC)/AllocTrac.h
	g++ -std=c++20 -O1 -Wall -Wextra -I$(SRC) -I$(BIN) -o $@ check_alloc.cpp $(SRC)/AllocTrac.cpp $(SRC)/log.cpp $(SRC)/common.cpp $(SRC)/timeutil.cpp $(SRC)/File.cpp $(SRC)/fs.cpp

clean:
	rm -rf check_bounds pm1prob-build check_pm1 check_gcd check_stage2 check_alloc check_tune check_tune_base stage1_oracle p1flip savefile-test exp-cache-test
