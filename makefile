# makefile for NN_mnist
#
SRC					= Delay.h File.h File.cpp NN_mnist.cpp
CFLAGS			= -Wall -Weffc++ -s -O3 -fopt-info-vec


NN_mnistNeon: $(SRC)
	g++ $(CFLAGS) -D _NEON -o NN_mnistNeon File.cpp NN_mnist.cpp

NN_mnistA64: $(SRC)
	g++ $(CFLAGS) -ffast-math -o NN_mnistA64 File.cpp NN_mnist.cpp

NN_mnist: $(SRC)
	g++ $(CFLAGS) -o NN_mnist File.cpp NN_mnist.cpp
