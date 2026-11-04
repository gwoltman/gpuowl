# Checks of src/Pm1.cpp against GMP.  GMP is used here only as an independent oracle: it is not a dependency of prpll
# (nothing in src/ includes or links it).
#
#   make        build and run
#   make MPRIME_SRC=~/mprime   also compare the mprime save file names with Prime95's own tempFileName (commonc.c)

SRC = ../../src
OBJS = $(SRC)/Pm1.cpp $(SRC)/File.cpp $(SRC)/log.cpp $(SRC)/common.cpp $(SRC)/timeutil.cpp

check: check_pm1
	./check_pm1

check_pm1: check_pm1.cpp $(OBJS) $(SRC)/Pm1.h
	g++ -std=c++20 -O2 -Wall -Wextra -I$(SRC) -DMPRIME_SRC='"$(MPRIME_SRC)"' -o $@ check_pm1.cpp $(OBJS) -lgmp

clean:
	rm -rf check_pm1 savefile-test
