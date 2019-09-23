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
CFLAGS =

## Other platforms (set variables appropriately)
#CC =
#CFLAGS =

##############################################################

all:
	$(CC) -c *.c
	ar rcv btglapack.a *.o
	ranlib btglapack.a
