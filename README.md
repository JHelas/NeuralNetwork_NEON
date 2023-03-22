# NeuralNetwork_NEON
Neural Network for aarch64 using NEON

Inspired by the book "Make Your Own Neural Network" by Tariq Rashid, i wrote
a program in c++ to test the provided algorithms.

The neural network consists of 3 fully connected layers using the well-known
mnist data (the fashion data set is working too).

Download the mnist data from: http://yann.lecun.com/exdb/mnist/ and unzip the files.

System requirements:

- Linux OS
- installed g++ and make system

To use the NEON routines a ARM aarch64 cpu with 64bit OS is required.

The makefile offers to generate 3 variants of the program:

NN_mnistNeon  for aarch64 systems with NEON support
NN_mnistA64   for aarch64 systems without NEON support
NN_mnist      for ordinary linux systems

The program expect the data files in the current working directory.
For any epoch the network is trained with 60000 records and tested with 10000 records.

Commandline parameters:

-e:<epochs>   Epoch count, default 4
 -h:<size>     Size of the hidden layer, default 192, > 10, with NEON use multiple of 16
 -l:<lr>       Learningrate, default 0.107, between > 0 and <= 1
                                                               
Example:

NN_mnist -h:640 -e:13 -l:.1071


This is the output of the program with default params (running on a nvidia jetson nano):

me@nano:~/data/mnist$ NN_mnistNeon
START
Loading training data ... done
Loading test data ... done
Layers: 784, 192, 10 - Records: 60000, 10000, LearnRate: 0.10700, RAM usage 262 MByte
Training epoch  0 ... 17.28s - Testing ... Hits:  9590, Error rate:  4.10%
Training epoch  1 ... 17.28s - Testing ... Hits:  9698, Error rate:  3.02%
Training epoch  2 ... 17.28s - Testing ... Hits:  9740, Error rate:  2.60%
Training epoch  3 ... 17.28s - Testing ... Hits:  9759, Error rate:  2.41%
