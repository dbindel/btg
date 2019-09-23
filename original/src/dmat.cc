//==========================================================================
// btg - Bayesian Trans-Gaussian Prediction program
//==========================================================================

// dmat.cc

// Written by David Bindel

// The dmat and dmatchol classes are wrappers around the BLAS and
// LAPACK double precision routines.

#include <iostream.h>
#include <fstream.h>

#include <stdlib.h>
#include <math.h>

#include "dmat.hh"
#include "btgerror.hh"

//==========================================================================

// Resize matrix
void dmat::resize(int mm, int nn)
{
  if (mm == 0 || nn == 0) {
    m = 0;
    n = 0;
    if (x)
      delete []x;
    x = NULL;
  } else if (x == NULL) {
    x = new double[mm*nn];
    m = mm;
    n = nn;
  } else if (mm*nn == m*n) {
    m = mm;
    n = nn;
  } else {
    delete []x;
    x = new double[mm*nn];
    m = mm;
    n = nn;
  }
}

// Return matrix product
dmat dmat::mult(const dmat &b, char transa, char transb) const
{
  integer opan, opam;
  if (transa == 'n') {
    opan = n;
    opam = m;
  } else {
    opan = m;
    opam = n;
  }
  integer opbn, opbm;
  if (transb == 'n') {
    opbn = b.n;
    opbm = b.m;
  } else {
    opbn = b.m;
    opbm = b.n;
  }
  if (opan != opbm) {
    BtgError::Set("Error: bad matrix multiplication");
    return dmat();
  }
  dmat result(opam, opbn);
  dgemm_(transa, transb, opam, opbn, opan, 1, x, m, b.x, b.m, 0, result.x, opam);
  return result;
}

// Initialize empty Cholesky factor
dmatchol::dmatchol()
{
  n = 0;
  x = NULL;
}

// Initialize copy of a Cholesky factor
dmatchol::dmatchol(const dmatchol &b)
{
  n = b.n;
  len = b.len;
  uplo = b.uplo;
  x = new double[len];
  dcopy_(len, b.x, 1, x, 1);
}

// Destruct Cholesky factor
dmatchol::~dmatchol()
{
  if (x)
    delete []x;
}

// Initialize Cholesky factor from a positive definite matrix
void dmatchol::init(const dmat &b, char uplo_triang)
{
  if (b.m != b.n) {
    BtgError::Set("Error: cannot factor non-square matrix");
    return;
  }
  if (x)
    delete []x;
  uplo = uplo_triang;
  n = b.n;
  len = n*(n+1)/2;
  x = new double[len];
  double *xij = x;
  double *bxij = b.x;
  if (uplo == 'U' || uplo == 'u') {
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i, ++bxij)
	if (j >= i)
	  *xij++ = *bxij;
  } else {
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i, ++bxij)
	if (j <= i)
	  *xij++ = *bxij;
  }
  integer info;
  dpptrf_(uplo, n, x, info);
  if (info != 0) {
    BtgError::Set("Error: could not factor matrix");
    return;
  }
}

// Use Cholesky factor to solve a system
dmat dmatchol::solve(const dmat &b)
{
  dmat ret(b);
  ipsolve(ret);
  return ret;
}

// Use Cholesky factor to solve a system (in place)
dmat &dmatchol::ipsolve(dmat &b)
{
  integer info;
  dpptrs_(uplo, n, b.n, x, b.x, b.m, info);
  if (info != 0)
    BtgError::Set("Error: could not solve system");
  return b;
}

// Compute the determinant of a Cholesky factor
double dmatchol::det(int &xp)
{
  double m = 1;
  xp = 0;
  double *vec = x;
  if (uplo == 'U' || uplo == 'u') {
    int incx = 2;
    for (int i = 0; i < n; ++i) {
      int xptmp;
      m *= frexp(*vec, &xptmp);
      xp += xptmp;
      m = frexp(m, &xptmp);
      xp += xptmp;
      vec += incx;
      incx++;
    }
  } else {
    int incx = n;
    for (int i = 0; i < n; ++i) {
      int xptmp;
      m *= frexp(*vec, &xptmp);
      xp += xptmp;
      m = frexp(m, &xptmp);
      xp += xptmp;
      vec += incx;
      incx--;
    }
  }
  return m;
}


