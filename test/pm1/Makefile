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

check_gcd: check_gcd.cpp $(PRPLL_OBJS) $(SRC)/Gcd.h
	g++ -std=c++20 -O2 -flto -Wall -Wextra -I$(SRC) -o $@ check_gcd.cpp $(PRPLL_OBJS) -lgmp -lOpenCL -lpthread

clean:
	rm -rf check_pm1 check_gcd stage1_oracle savefile-test exp-cache-test
