//==========================================================================
// btg - Bayesian Trans-Gaussian Prediction program
//==========================================================================

// dmat.hh

// Written by David Bindel

// Wrapper classes for the BLAS and LAPACK

#ifndef _DMAT
#define _DMAT

#include "blas.hh"
#include <assert.h>

//==========================================================================

class dmatchol;

// Double-precision matrix
class dmat {
  friend class dmatchol;
private:
  integer m, n;
  double *x;
public:

  inline dmat()
    {
      x = NULL;
      resize(0,0);
    }

  inline dmat(const dmat &b)
    {
      x = NULL;
      resize(b.m, b.n);
      dcopy_(m*n, b.x, 1, x, 1);
    }

  inline dmat(int mm, int nn = 1)
    {
      x = NULL;
      resize(mm, nn);
    }

  inline ~dmat()
    {
      resize(0,0);
    }

  dmat &operator=(const dmat &b)
    {
      resize(b.m, b.n);
      dcopy_(m*n, b.x, 1, x, 1);
      return *this;
    }


  dmat &operator=(double alpha)
    {
      dcopy_(m*n, &alpha, 0, x, 1);
      return *this;
    }

  double dot(const dmat &b) const
    {
      return ddot_(n*m, x, 1, b.x, 1);
    }

  inline void operator+=(const dmat &b)
    {
      daxpy_(n*m, 1, b.x, 1, x, 1);
    }

  inline void operator-=(const dmat &b)
    {
      daxpy_(n*m, -1, b.x, 1, x, 1);
    }

  inline void operator+=(double alpha)
    {
      daxpy_(n*m, 1, &alpha, 0, x, 1);
    }

  inline void operator-=(double alpha)
    {
      daxpy_(n*m, -1, &alpha, 0, x, 1);
    }

  inline void operator*=(double alpha)
    {
      dscal_(n*m, alpha, x, 1);
    }

  inline void operator/=(double alpha)
    {
      dscal_(n*m, 1/alpha, x, 1);
    }

  inline double &operator()(int i)
    {
      return x[i];
    }

  inline double &operator()(int i, int j)
    {
      return x[j*m+i];
    }

  inline const double &operator()(int i) const
    {
      return x[i];
    }

  inline const double &operator()(int i, int j) const
    {
      return x[j*m+i];
    }

  inline double sum() const
    {
      return dasum_(m*n, x, 1);
    }

  inline double *mem()
    {
      return x;
    }

  inline double *mem(int i)
    {
      return x+i;
    }

  inline double *mem(int i, int j)
    {
      return x+j*m+i;
    }

  void resize(int mm, int nn = 1);
  dmat mult(const dmat &b, char transa = 'n', char transb = 'n') const;
};

// Double precision Cholesky decomposition
class dmatchol {
private:

  int n, len;
  char uplo;
  double *x;

public:

  dmatchol();
  dmatchol(const dmatchol &b);
  ~dmatchol();
  void init(const dmat &b, char uplo_triang = 'U');
  dmat solve(const dmat &b);
  dmat &ipsolve(dmat &b);
  double det(int &xp);

};

#endif
