//==========================================================================
// btg - Bayesian Trans-Gaussian Prediction program
//==========================================================================

// btg013.hh

// Written by David Bindel

#ifndef _BTG
#define _BTG

#include <iostream.h>
#include <math.h>

#include "dmat.hh"
#include "btgfuncs.hh"

//==========================================================================

// Parameter-dependent data for one draw from a Monte Carlo sample

struct ParamData {
  double lambda;            // Transformation parameter
  double theta[MAX_NTHETA]; // Correlation parameters
  dmatchol cholS;           // Decomposed sigma correlation matrix
  dmatchol cholY;           // Decomposition of Y = X'inv(S)X
  dmat gz;                  // Transformed data
  dmat Bh;                  // Estimated regression coefficients
  double q;                 // (? - q in paper)
  double pz;                // kp(z | theta, lambda)
  double M;                 // Location parameter for transformed t
  double qC;                // scale matrix (1x1) qC
  double sigma;             // sigma = sqrt(qC/(n-p)) is scale for transformed t
};

// Class that actually implements the BTG prediction algorithm

class Predictor {
private:

  // Introduced for minimization

  static double Opt_pz(double *x, int);
  static Predictor *predictor_link;

  // Saved parameter-dependent data

  ParamData *pdata;

  // User supplied data

  PFTransform g, dg;             // Pointer to transformation function
  PFCorrelation K;               // Pointer to correlation function
  PFCovariate f[NUM_COVARIATES]; // Pointers to covariate functions
  int p;		         // Number of covariate functions
  int m;		         // Number of samples to take
  int mesh;		         // Number of rectangles to use in integration
  int ntheta;		         // Number of correlation parameters
  double rangemin, rangemax;     // Interval on which predictive density is considered
  double lambdamin, lambdamax;   // Interval on which lambda is considered
  double thetamin[MAX_NTHETA], 
         thetamax[MAX_NTHETA];   // Region on which theta is considered

  int n;		 // Number of data points
  double *sx;            // Vector of observation point x-coordinates
  double *sy;            // Vector of observation point y-coordinates
  double *z;		 // Vector of observed data

  // Precomputed values

  dmat X;
  double gamma_factor;	// Factor used in the computation of p(z0 | theta, lambda)

  // These functions initialize the predictor programs data structures

  void Precompute();
  void InitParams();
  void InitParam(ParamData &d);

  // Do the computations associated with a single parameter draw

  void Compute_Parameters(double &lambda, double *theta);
  void Compute_S(dmatchol &cholS, double *theta);
  void Compute_Y(dmatchol &cholY, dmatchol &cholS);
  void Compute_gz(dmat &gz, double lambda);
  void Compute_Bh(dmat &Bh, dmatchol &cholS, dmatchol &cholY, dmat &gz);
  void Compute_q(double &q, dmatchol &cholS, dmat &gz, dmat &Bh);
  void Compute_pz(double &pz, dmatchol &cholS, dmatchol &cholY, double q, double lambda);

  // Do computations for p(z0 | theta[i], lambda[i], z) and p(z0 | z)
  // These are only really used for plotting the density

  double Compute_pz0(double z0, int i);
  double Compute_pz0(double z0);

public:

  // Initialize / shutdown the predictor

  Predictor();
  ~Predictor();

  // Set different program options
  // Syntax for variable argument list commands:
  //   Predictor::SetCorrelation(PFCorrelation func, int n,
  //                             double min1, double max1, ...,
  //                             double minn, double maxn)
  //   Predictor::SetCovariates(int n,
  //                            PFCovariate f1, ...
  //                            PFCovariate fn);

  void SetTransform(PFTransform func, PFTransform dfunc);
  void SetCorrelation(PFCorrelation func, int num_parameters, ...);
  void SetCovariates(int num_covariates, ...);
  void SetLambdaRange(double min, double max);
  void SetRange(double min, double max);
  void SetSampleSize(int sample_size);
  void SetMeshSize(int mesh_size);

  // Initialize the engine for different setting or for a new point (x,y)

  void InitComputations();
  void Precompute_pz0(double x, double y);

  // Compute the median and 95% prediction interval
  // Also compute the estimated CDF, or other quantiles and confidence intervals

  void ComputeStats(double &med, double &lb, double &rb);
  double ComputeCDF(double z0);
  double ComputeQuantile(double alpha);
  double ComputeSymmetricCI(double alpha, double med);

  // Diagnostics: Estimate Monte Carlo error for p(z0 | z) or cross-validate at the kth data point (zero indexed)

  double ComputeErrorEstimate(double x, double y);
  void CrossValidate(int k, double &x, double &y, double &actual, double &median, double &lb, double &rb);

  // Estimate model parameters

  void EstimateModel(double &lambda, double *theta, int &ntheta_arg);

  // Load a data set

  int Load(char *fname);

  // Plotting routines: plot a distribution or a map of the error
  // If xflag = 0, don't include coordinates (density plot is a vector of reals, map is a matrix)
  //    else, include coordinates (density plot is a vector of x/y pairs, map is a vector of x/y/z triples)

  void Plot(ostream &os, int xFlag = 0);
  void Map(double xmin, double xmax, double ymin, double ymax,
		  double step, ostream &osmap, ostream &oserr, int xFlag = 0);

  // Get the size of the current data set and the ranges of the x, y, and z values

  int GetDataSize();
  void GetRanges(double &xmin, double &xmax,
			double &ymin, double &ymax,
			double &zmin, double &zmax);

};

//==========================================================================

inline int Predictor::GetDataSize()
{
  return n;
}

//==========================================================================

#endif




