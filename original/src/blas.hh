//==========================================================================
// btg - Bayesian Trans-Gaussian Prediction program
//==========================================================================

// blas.hh

// Written by David Bindel

// This header file is copied from man pages and the BLAS and LAPACK
// documentation on netlib.  A little bit is also taken from the f2c.h
// header file.

#ifndef _BLAS_H

#define SIZEOF_INT 4

#if SIZEOF_INT == 4
typedef int integer;
#else
typedef long integer;
#endif

//==========================================================================

extern "C" {

  // Some level 1 BLAS routines

  void dswap_(const integer &n,
	      double *dx, const integer &incx,
	      double *dy, const integer &incy);

  void dscal_(const integer &n,
	      const double &alpha,
	      double *dx, const integer &incx);

  void dcopy_(const integer &n,
	      double *dx, const integer &incx,
	      double *dy, const integer &incy);

  void daxpy_(const integer &n,
	      const double &alpha,
	      double *dx, const integer &incx,
	      double *dy, const integer &incy);

  double ddot_(const integer &n,
		   double *dx, const integer &incx,
		   double *dy, const integer &incy);

  double dnrm2_(const integer &n,
		    double *dx, const integer &incx);

  double dasum_(const integer &n,
		    double *dx, const integer &incx);

  integer idamax_(const integer &n,
		  double *dx, const integer &incx);


  // Some level 2 BLAS routines

  void dgemv_(const char &trans,
	      const integer &m, const integer &n,
	      const double &alpha,
	      double *da, const integer &lda,
	      double *x, const integer &incx,
	      const double &beta,
	      double *y, const integer &incy);

  void dsymv_(const char &uplo,
	      const integer &n,
	      const double &alpha,
	      double *da, const integer &lda,
	      double *dx, const integer &incx,
	      const double &beta,
	      double *dy, const integer &incy);

  void dspmv_(const char &uplo,
	      const integer &n,
	      const double &alpha,
	      double *dap,
	      double *dx, const integer &incx,
	      const double &beta,
	      double *dy, const integer &incy);

  void dtrmv_(const char &uplo, const char &trans, const char &diag,
	      const integer &n,
	      const double *dap, const integer &lda,
	      const double *dx, const integer &incx);

  void dtpmv_(const char &uplo, const char &trans, const char &diag,
	      const integer &n,
	      const double *dap,
	      const double *dx, const integer &incx);

  void dtrsv_(const char &uplo, const char &trans, const char &diag,
	      const integer &n,
	      const double *dap, const integer &lda,
	      const double *dx, const integer &incx);

  void dtpsv_(const char &uplo, const char &trans, const char &diag,
	      const integer &n,
	      const double *dap,
	      const double *dx, const integer &incx);

  void dger_(const integer &m, const integer &n,
	     const double &alpha,
	     double *dx, const integer &incx,
	     double *dy, const integer &incy,
	     double *da, const integer &lda);

  void dsyr_(const char &uplo,
	     const integer &n,
	     const double &alpha,
	     double *dx, const integer &incx,
	     double *da, const integer &lda);

  void dspr_(const char &uplo,
	     const integer &n,
	     const double &alpha,
	     double *dx, const integer &incx,
	     double *dap);

  void dsyr2_(const char &uplo,
	      const integer &n,
	      const double &alpha,
	      double *dx, const integer &incx,
	      double *dy, const integer &incy,
	      double *da, const integer &lda);

  void dspr2_(const char &uplo,
	      const integer &n,
	      const double &alpha,
	      double *dx, const integer &incx,
	      double *dy, const integer &incy,
	      double *da);


  // Some level 3 BLAS routines

  void dgemm_(const char &transa, const char &transb,
	      const integer &m, const integer &n, const integer &k,
	      const double &alpha,
	      double *da, const integer &lda,
	      double *db, const integer &ldb,
	      const double &beta,
	      double *dc, const integer &ldc);

  void dsymm_(const char &side, const char &uplo,
	      const integer &m, const integer &n,
	      const double &alpha,
	      double *da, const integer &lda,
	      double *db, const integer &ldb,
	      const double &beta,
	      double *dc, const integer &ldc);

  void dsyrk_(const char &uplo, const char &trans,
	      const integer &n, const integer &k,
	      const double &alpha,
	      double *da, const integer &lda,
	      const double &beta,
	      double *dc, const integer &ldc);

  void dsyr2k_(const char &uplo, const char &trans,
	       const integer &n, const integer &k,
	       const double &alpha,
	       double *da, const integer &lda,
	       double *db, const integer &ldb,
	       const double &beta,
	       double *dc, const integer &ldc);

  void dtrmm_(const char &side, const char &uplo,
	      const char &transa, const char &diag,
	      const integer &m, const integer &n,
	      const double &alpha,
	      double *da, const integer &lda,
	      double *db, const integer &ldb);

  void dtrsm_(const char &side, const char &uplo,
	      const char &transa, const char &diag,
	      const integer &m, const integer &n,
	      const double &alpha,
	      double *da, const integer &lda,
	      double *db, const integer &ldb);

}

extern "C" {

  // Some LAPACK routines

  void dpptrf_(const char &uplo, const integer &n,
	       double *ap,
	       integer &info);

  void dpptrs_(const char &uplo, const integer &n, const integer &nrhs,
	       double *ap,
	       double *b, const integer &ldb,
	       integer &info);

}

#endif

