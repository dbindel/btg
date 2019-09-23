//==========================================================================
// btg - Bayesian Trans-Gaussian Prediction program
//==========================================================================

// btgfuncs.cc

// Written by David Bindel

// Transformation, covariate, and correlation functions used by the btg program

#include <math.h>

#include <assert.h>
#include <iostream.h>

#include "btgfuncs.hh"
#include "btgerror.hh"

//==========================================================================

// Declaration of the rkbesl routine
extern "C" {
  int rkbesl_(const double &x, const double &alpha, const int &nb, const int &ize,
	       double *bk, int &ncalc);
}

// Box-Cox transformation function
double BoxCox(double lambda, double x)
{
  if (lambda == 0)
    return log(x);
  else
    return expm1(log(x)*lambda) / lambda;
}

// Derivative of Box-Cox transformation
double DBoxCox(double lambda, double x)
{
  return pow(x, lambda-1.0);
}

// Isotropic correlation function K (l is the Euclidean distance)
double ExpCorr(double theta[], double x1, double y1, double x2, double y2)
{
  double l = hypot(x2-x1, y2-y1);
  return pow(theta[0], pow(l, theta[1]));
}

// Matern class corrlelation functions
double MaternCorr(double theta[], double x1, double y1, double x2, double y2)
{
  double x = hypot(x2-x1, y2-y1);
  if (x == 0) {
    return 1;
  }
  double theta1 = -log(theta[0]);
  double theta2 = -log(theta[1]);
  int nb = (int) theta2;
  double kalpha = theta2 - nb;
  double *bk = new double[++nb];
  int ncalc;
  x /= theta1;
  rkbesl_(x, kalpha, nb, 2, bk, ncalc);
  if (ncalc != nb) {
    BtgError::Set("Error: could not compute modified bessel function K");
    delete []bk;
    return 0;
  }
  double ret = 2*exp(theta2*log(x/2)+log(bk[nb-1])-x-lgamma(theta2));
  if (ret < 0 || ret > 1)
    BtgError::Set("Error: correlation is outside [0,1]");
  delete []bk;
  return ret;
}

// Rational quadratic correlation function
double RationalCorr(double theta[], double x1, double y1, double x2, double y2)
{
  double l = hypot(x2-x1, y2-y1);
  if (l == 0)
    return 1;
  double theta1 = -log(theta[0]);
  double theta2 = -log(theta[1]);
  double scaled_l = l / theta1;
  return pow(1 + scaled_l*scaled_l, -theta2);
}

// Spherical correlation function
double SphericalCorr(double theta[], double x1, double y1, double x2, double y2)
{
  double l = hypot(x2-x1, y2-y1);
  if (l == 0)
    return 1;
  double theta1 = -log(theta[0]);
  if (l <= theta1) {
    double scaled_l = l/theta1;
    return (1 - scaled_l*(3 - scaled_l*scaled_l)/2);
  }
  return 0;
}

double UnitFn(double, double)
{
  return 1.0;
}

double XFn(double x, double)
{
  return x;
}

double YFn(double, double y)
{
  return y;
}

double XYFn(double x, double y)
{
  return x*y;
}

double XXFn(double x, double)
{
  return x*x;
}

double YYFn(double, double y)
{
  return y*y;
}
