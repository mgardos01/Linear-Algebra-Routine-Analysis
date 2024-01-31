GPU_CPP = hipcc
CPU_CPP = g++
CFLAGS = -std=c++11 -lcublas

all: test

test: test.cpp
	$(GPU_CPP) $(CFLAGS) -o $@ $<

.PHONY: clean

clean: 
	rm -rf test *.out *.err *.png *.dat
