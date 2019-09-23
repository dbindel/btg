//==========================================================================
// btg - Bayesian Trans-Gaussian Prediction program
//==========================================================================

// btgfuncs.hh

// Written by David Bindel

// Auxiliary functions (transformations, correlation functions, covariate functions)

//==========================================================================

#ifndef _BTGFUNCS
#define _BTGFUNCS

// Maximum number of correlation parameters and max number of covariate functions

const int MAX_NTHETA = 3;
const int NUM_COVARIATES = 6;

// Pointers to transformation, correlation, and covariate functions

typedef double (*PFTransform)(double lambda, double z);
typedef double (*PFCorrelation)(double *theta, double x1, double y1, double x2, double y2);
typedef double (*PFCovariate)(double x, double y);

// Transformation (Box-Cox)

double BoxCox(double lambda, double x);
double DBoxCox(double lambda, double x);

// Correlation functions

double ExpCorr(double theta[], double x1, double y1, double x2, double y2);
double MaternCorr(double theta[], double x1, double y1, double x2, double y2);
double RationalCorr(double theta[], double x1, double y1, double x2, double y2);
double SphericalCorr(double theta[], double x1, double y1, double x2, double y2);

// Covariate functions

double UnitFn(double, double);
double XFn(double, double);
double YFn(double, double);
double XYFn(double, double);
double XXFn(double, double);
double YYFn(double, double);

#endif


