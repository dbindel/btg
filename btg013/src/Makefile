##############################################################
# btg - Bayesian Trans-Gaussian prediction
##############################################################

# Makefile

##############################################################
# System dependent stuff
##############################################################

# Set these variables appropriately for your system.
# You will need to include the system math library (-lm)
# and whatever libraries are needed for the BLAS and LAPACK

# This configuration should work with any system with gcc/g++
CC = gcc
CXX = g++
CFLAGS =
CXXFLAGS =
LD = g++
LDFLAGS = 
LIBS = ../lapack/btglapack.a -lm

## SunOS with licensed libraries
#CC = cc
#CXX = CC
#CFLAGS = -dalign
#CXXFLAGS = -dalign
#LD = cc
#LDFLAGS = 
#LIBS = -lC -lm -lsunmath -xlic_lib=sunperf

## Digital Alpha with extended math libraries
#CC = cc
#CXX = cxx
#CFLAGS =
#CXXFLAGS =
#LD = cxx
#LDFLAGS = 
#LIBS = -lm -ldxml

## Other platforms (set variables appropriately)
#CC =
#CXX =
#CFLAGS =
#CXXFLAGS =
#LD = 
#LDFLAGS = 
#LIBS =

##############################################################

OBJ_FILES= btgdriver.o btgerror.o btgfuncs.o btg013.o dmat.o rkbesl.o hooke.o

btg: $(OBJ_FILES)
	$(LD) $(LDFLAGS) -o btg $(OBJ_FILES) $(LIBS)

btgdriver.o: btgdriver.cc
	$(CXX) -c $(CXXFLAGS) btgdriver.cc

btgerror.o: btgerror.cc
	$(CXX) -c $(CXXFLAGS) btgerror.cc

btgfuncs.o: btgfuncs.hh
	$(CXX) -c $(CXXFLAGS) btgfuncs.cc

btg013.o: btg013.cc btg013.hh btgfuncs.hh
	$(CXX) -c $(CXXFLAGS) btg013.cc

dmat.o: dmat.cc dmat.hh
	$(CXX) -c $(CXXFLAGS) dmat.cc

rkbesl.o: rkbesl.c
	$(CC) -c $(CFLAGS) rkbesl.c

hooke.o: hooke.c
	$(CC) -c $(CFLAGS) hooke.c

clean:
	rm -f *.o *~ config.* core
