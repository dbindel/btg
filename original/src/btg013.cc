//==========================================================================
// btg - Bayesian Trans-Gaussian Prediction program
//==========================================================================

// btg013.cc

// Written by David Bindel

#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>

#include <iostream.h>
#include <fstream.h>

#include "btg013.hh"
#include "btgerror.hh"

// RAND_MAX should be defined for any ANSI compliant C compiler.
// Unfortunately, some platforms are still using outdated compilers.
#ifndef RAND_MAX
#define RAND_MAX 0x7FFFFFFF
#endif

// Include the Hooke-Jeeves optimization routine
extern "C" {
  int hooke(double (*)(double *, int), int, double *, double *, double, double, int);
}

//==========================================================================

// Define the maximum number of bisections to allow
#define BISECT_ITMAX 100

// Definitions of static storage
// (See the header file for declarations and more description)

Predictor *Predictor::predictor_link;

//==========================================================================
// These routines are just helpers

// Use a continued fraction approximation to compute the incomplete beta function
// This code is from Numerical Recipes in C.  The continued fraction expansion of
// the incomplete beta function can be found in Abramowitz and Stegun.  A similar
// approach is used in Computation of Special Functions by Shanjie Zhang and
// Jianming Jin.
float betacf(float a, float b, float x)
{
  const int ITMAX = 100;
  const float EPS = 3.0e-7;

  float qap, qam, qab, em, tem, d;
  float bz, bm=1.0, bp, bpp;
  float az=1.0, am=1.0, ap, app, aold;
  int m;

  qab = a+b;
  qap = a+1.0;
  qam = a-1.0;
  bz = 1.0-qab*x/qap;
  for (m=1; m<=ITMAX; m++) {
    em = (float) m;
    tem = em+em;
    d = em*(b-em)*x/((qam+tem)*(a+tem));
    ap = az+d*am;
    bp = bz+d*bm;
    d = -(a+em)*(qab+em)*x/((qam+tem)*(a+tem));
    app = ap+d*az;
    bpp = bp+d*bz;
    aold = az;
    am = ap/bpp;
    bm = bp/bpp;
    az = app/bpp;
    bz = 1.0;
    if (fabs(az-aold) < (EPS*fabs(az))) return az;
  }
  BtgError::Set("Error: Could not compute Student t cdf");
}

// Compute an integer power
double powi(double x, int i)
{
  double y = 1;
  while (i != 0) {
    if (i & 1)
      y *= x;
    x *= x;
    i /= 2;
  }
  return y;
}

// Compute the value of the student t cdf with nu degrees of freedom
// This code is (loosely) adapted from the routine in Computation of Special Functions
// for the computation of the incomplete beta function.
double Student(double t, int nu)
{
  double x = (double) nu/(nu+t*t);
  double a = nu/2.0, b = 0.5;
  double bt, bix;

  if (x == 0 || x == 1.0)
    bt = 0;
  else
    bt = exp(lgamma((nu+1)/2.0)-lgamma(nu/2.0))*sqrt(powi(x,nu)*(1-x)/M_PI);
  if (x < (nu+2.0)/(nu+5.0))
    bix = bt*betacf(a,b,x)/a;
  else
    bix = 1.0-bt*betacf(b,a,1.0-x)/b;
  if (t >= 0)
    return 1-bix/2;
  else
    return bix/2;
}

// Generate a uniform number in the range (rmin,rmax)
double Unif(double rmin, double rmax)
{
  return (rmax-rmin)*((double) rand() / RAND_MAX) + rmin;
}

// Compare two ParamData structures for sorting
int ParamCompare(const void *a, const void *b)
{
  double da = ((const ParamData *) a)->pz;
  double db = ((const ParamData *) b)->pz;
  return (da < db) ? -1 :
    (da > db) ? 1 :
    0;  
}

// Evaluate -kp(z | theta, lambda) where x = [theta; lambda]
// is the parameter vector.  Outside of the valid ranges,
// return a penalty value of zero.
// Used to find the maximum likelihood estimates for model parameters.
double Predictor::Opt_pz(double *x, int)
{
  ParamData d;

  d.lambda = x[0];
  if (d.lambda <= predictor_link->lambdamin || d.lambda >= predictor_link->lambdamax)
    return 0;
  for (int i = 0; i < predictor_link->ntheta; ++i) {
    d.theta[i] = x[i+1];
    if (d.theta[i] <= predictor_link->thetamin[i] || d.theta[i] >= predictor_link->thetamax[i])
      return 0;
  }
  predictor_link->InitParam(d);
  return -d.pz;
}

//==========================================================================

// Do some precomputations, specifically
// - resize X to be an n-by-p matrix, and fill it with the value 1.0
//   (UPDATE NOTE: Undo the hard-wiring of the covariate functions)
// - Compute gamma_factor, which probably should better be called "beta factor"
//   Note that this factor is actually equal to Beta((n-p)/2, 1/2), and is used
//   in the computation of the t-distribution function p(z0 | lambda, theta, z)
void Predictor::Precompute()
{
  X.resize(n,p);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < p; ++j)
      X(i,j) = f[j](sx[i], sy[i]);
  gamma_factor = exp(lgamma((n-p+1)/2.0)-lgamma((n-p)/2.0))/sqrt(M_PI);
}

// Initialize the parameter information
// - For each parameter sample, generate the actual parameter values
//   then initialize all the parameter-dependent variables
// - Rescale all of the k*p(z) fields so that they sum to 1.0 (equal w)
void Predictor::InitParams()
{
  int i;
  for (i = 0; i < m; ++i) {
    Compute_Parameters(pdata[i].lambda, pdata[i].theta);
    InitParam(pdata[i]);
  }
  qsort(pdata, m, sizeof(ParamData), ParamCompare);
  double pz_total = 0;
  for (i = 0; i < m; ++i)
    pz_total += pdata[i].pz;
  for (i = 0; i < m; ++i)
    pdata[i].pz /= pz_total;
}

// Initialize the information associated with a single ParamData
// structure
void Predictor::InitParam(ParamData &d)
{
  Compute_S(d.cholS, d.theta);
  if (BtgError::Status())
    return;
  Compute_Y(d.cholY, d.cholS);
  if (BtgError::Status())
    return;
  Compute_gz(d.gz, d.lambda);
  Compute_Bh(d.Bh, d.cholS, d.cholY, d.gz);
  Compute_q(d.q, d.cholS, d.gz, d.Bh);
  Compute_pz(d.pz, d.cholS, d.cholY, d.q, d.lambda);
}

// Generate random parameters
void Predictor::Compute_Parameters(double &lambda, double *theta)
{
  lambda = Unif(lambdamin, lambdamax);
  for (int j = 0; j < ntheta; ++j)
    theta[j] = Unif(thetamin[j], thetamax[j]);
}

// Compute the Cholesky factor of the correlation matrix S
// S (called Sigma in the paper) is defined by
// S_i,j = K_theta(z_i, z_j)
void Predictor::Compute_S(dmatchol &cholS, double *theta)
{
  dmat S(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = i; j < n; ++j)
      S(i,j) = S(j,i) = K(theta, sx[i], sy[i], sx[j], sy[j]);
  cholS.init(S);
}

// Compute the Cholesky factor of Y = X'*inv(S)*X
// This quantity is not explicitly named in the paper, but it occurs frequently enough
// that it is saved here.
void Predictor::Compute_Y(dmatchol &cholY, dmatchol &cholS)
{
  cholY.init(X.mult(cholS.solve(X), 't'));
}

// Compute the transformed data vector by using the transformation g_lambda to map each element
void Predictor::Compute_gz(dmat &gz, double lambda)
{
  gz.resize(n);
  for (int i = 0; i < n; ++i)
    gz(i) = g(lambda, z[i]);
}

// Compute the estimated regression parameter vector beta hat
// beta hat = inv(Y)*X'*inv(S)*gz
void Predictor::Compute_Bh(dmat &Bh, dmatchol &cholS, dmatchol &cholY, dmat &gz)
{
  Bh = cholY.solve(X.mult(cholS.solve(gz), 't'));
}

// Compute q tilde (see the paper)
// q tilde = (gz-X*Bh)'*inv(S)*(gz-X*Bh)
void Predictor::Compute_q(double &q, dmatchol &cholS, dmat &gz, dmat &Bh)
{
  dmat v = gz;
  v -= X.mult(Bh);
  q = v.mult(cholS.solve(v), 't')(0);
}

// Compute p(z | theta, lambda) = k * det(S)**-0.5 * det(Y)**-0.5 * q**(-(n-p)/2) * J**(1-p/n)
// where J = prod i = 1 to n of g_lambda(z_i)
// Note that this formula has problems with overflows and underflows, and so the actual computations
// don't look very much like the formula.  What is actually done is
// *) the determinants of the Cholesky factors of S and Y are computed as a normalized mantissa and
//    an integer binary exponent
// *) the same is done for the Jacobian J
// *) logarithms are used to prevent underflows/overflows when the formula is finally evaluated
// Here k is an unknown scaling constant independent of lambda, theta
void Predictor::Compute_pz(double &pz, dmatchol &cholS, dmatchol &cholY, double q, double lambda)
{
  int detS2, detY2;

  double detS1 = cholS.det(detS2);
  double detY1 = cholY.det(detY2);

  double J1 = 1;
  int J2 = 0;
  for (int i = 0; i < n; ++i) {
    int xptmp;
    J1 *= frexp(fabs(dg(lambda, z[i])), &xptmp);
    J2 += xptmp;
    J1 = frexp(J1, &xptmp);
    J2 += xptmp;
  }

  pz = exp((n-p)*(log(J1)/n - log(q)/2)+((n-p)*J2 - n*(detS2+detY2))*M_LN2/n) / (detS1*detY1);

  if (isnan(pz) || pz < 0 || pz >= HUGE_VAL)
    BtgError::Set("Error: Could not compute p(z | theta, lambda, i)");
}

// Compute p(z0 | theta_i, lambda_i, z)
double Predictor::Compute_pz0(double z0, int i)
{
  double M = pdata[i].M;
  double qC = pdata[i].qC;
  double lambda = pdata[i].lambda;
  double gmm = g(lambda, z0) - M;
  return gamma_factor * fabs(dg(lambda, z0)) * pow(1 + gmm*gmm/qC, -(n-p+1)/2.0) / sqrt(qC);
}

// Compute p(z0 | z)
double Predictor::Compute_pz0(double z0)
{
  double result = 0;
  for (int i = 0; i < m; ++i)
    result += Compute_pz0(z0, i) * pdata[i].pz;
  return result;
}

//==========================================================================

// Initialize the random number generator with the system time,
// and initialize the pointers for all the dynamically allocated structures.
Predictor::Predictor()
{
  srand(time(NULL));
  pdata = NULL;
  sx = NULL;
  sy = NULL;
  z = NULL;
  SetTransform(BoxCox, DBoxCox);
  SetCorrelation(ExpCorr, 2, 0.0, 1.0, 0.0, 2.0);
  SetCovariates(1, UnitFn);
  SetLambdaRange(-3, 3);
  SetSampleSize(500);
  SetMeshSize(1000);
}

// Deallocate everything, including the theta range vectors
Predictor::~Predictor()
{
  if (pdata)
    delete []pdata;
  if (z) {
    delete []sx;
    delete []sy;
    delete []z;
  }
}

// Set the transformation function to be used
void Predictor::SetTransform(PFTransform func, PFTransform dfunc)
{
  g = func;
  dg = dfunc;
}

// Set the correlation function and parameter ranges to be used
// Syntax: Predictor::SetCorrelation(K, n, min1, max1, min2, max2, ..., minn, maxn)
void Predictor::SetCorrelation(PFCorrelation func, int num_parameters, ...)
{
  va_list arg;
  K = func;
  ntheta = num_parameters;
  va_start(arg, num_parameters);
  for (int i = 0; i < ntheta; ++i) {
    thetamin[i] = va_arg(arg, double);
    thetamax[i] = va_arg(arg, double);
  }
  va_end(arg);
}

// Set the covariate functions to be used for the underlying Gaussian field
// Syntax: Predictor::SetCovariates(n, f1, f2, ..., fn)
void Predictor::SetCovariates(int num_covariates, ...)
{
  va_list arg;
  p = num_covariates;
  va_start(arg, num_covariates);
  for (int i = 0; i < p; ++i)
    f[i] = va_arg(arg, PFCovariate);
}

// Set the allowed range for the lambda parameter
void Predictor::SetLambdaRange(double min, double max)
{
  lambdamin = min;
  lambdamax = max;
}

// Set the allowed range for z0
void Predictor::SetRange(double min, double max)
{
  rangemin = min;
  rangemax = max;
}

// Set the sample size to be used for Monte Carlo integration
void Predictor::SetSampleSize(int sample_size)
{
  m = sample_size;
}

// Set the mesh size to be used for density plot and for determining
// precision to use in root-finding
void Predictor::SetMeshSize(int mesh_size)
{
  mesh = mesh_size;
}

// Initialize the computations by reallocating the data structures to be sized
// according to the current settings, then initializing them.
void Predictor::InitComputations()
{
  if (pdata)
    delete []pdata;
  pdata = new ParamData[m];
  Precompute();
  InitParams();
}

// Do some location-dependent computations used in computing p(z0 | z), namely
// *) compute the correlation vector B where B_i = K_theta(s0, s_i) with s0=(x,y)
//    note that this B is actually the transpose of the one in the paper
//    (this is a column vector, as opposed to a row vector)
// *) compute the factor invSB = inv(S)*B = (B'*inv(S))'
// *) compute H = X0 - B'*inv(S)*X where X0 is the location-dependent covariate vector
// *) compute M = B*inv(S)*gz + H*Bh
// *) compute q*C = q*(H'*inv(Y)*H + K_theta(s0,s0) - B'*inv(S)*B)
// *) compute pz0_factor (just a factor in the formula for p(z0 | z) that can be precomputed)
void Predictor::Precompute_pz0(double x, double y)
{
  int i, j;
  dmat X0(p), B(n), invSB, H;
  for (j = 0; j < p; ++j)
    X0(j) = f[j](x, y);
  for (i = 0; i < m; ++i) {
    ParamData &d = pdata[i];
    for (j = 0; j < n; ++j)
      B(j) = K(d.theta, sx[j], sy[j], x, y);
    invSB = d.cholS.solve(B);
    if (BtgError::Status())
      return;
    H = X.mult(invSB, 't');
    for (j = 0; j < p; ++j)
      H(j) = X0(j) - H(j);
    d.M = invSB.dot(d.gz) + H.dot(d.Bh);
    d.qC = d.q*(H.dot(d.cholY.solve(H)) + K(d.theta, x, y, x, y) - invSB.dot(B));
    d.sigma = sqrt(d.qC/(n-p));
  }
}

// Compute the median and a symmetric 95% prediction interval
void Predictor::ComputeStats(double &med, double &lb, double &rb)
{
  med = ComputeQuantile(0.5);
  double wid = ComputeSymmetricCI(0.95, med);
  lb = med-wid;
  rb = med+wid;
}

// Evaluate the cdf F(z0 | z)
double Predictor::ComputeCDF(double z0)
{
  double cdf = 0;
  for (int i = 0; i < m; ++i) {
    double t = (g(pdata[i].lambda, z0) - pdata[i].M) / pdata[i].sigma;
    cdf += Student(t, n-p) * pdata[i].pz;
  }
  return cdf;
}

// Use a bisection search to find the alpha quantile of
// the predictive cdf F(z0 | z)
double Predictor::ComputeQuantile(double alpha)
{
  double eps = (rangemax - rangemin) / mesh;
  double min = rangemin;
  double max = rangemax;
  double minval = ComputeCDF(min);
  double maxval = ComputeCDF(max);
  if (minval > alpha)
    return min;
  else if (maxval < alpha)
    return max;
  int i = 0;
  for (i = 0; max-min >= eps && !BtgError::Status() && i < BISECT_ITMAX; ++i) {
    double mid = min+(max-min)/2;
    double midval = ComputeCDF(mid);
    if (midval < alpha) {
      min = mid;
      minval = midval;
    } else if (midval > alpha) {
      max = mid;
      maxval = midval;
    } else
      return mid;
  }
  if (i == BISECT_ITMAX)
    BtgError::Set("Error: could not compute quantile to desired accuracy");
  return min+(max-min)*(alpha-minval)/(maxval-minval);
}

// Compute the symmetric prediction interval about med that covers
// probability of alpha
double Predictor::ComputeSymmetricCI(double alpha, double med)
{
  double eps = (rangemax-rangemin) / mesh;
  double min = 0;
  double max = (rangemax-med > med-rangemin) ? med-rangemin : rangemax-med;
  double minval = 0;
  double maxval = ComputeCDF(med+max)-ComputeCDF(med-max);
  if (maxval < alpha)
    return max;
  int i = 0;
  for (i = 0; max-min >= eps && !BtgError::Status() && i < BISECT_ITMAX; ++i) {
    double mid = min+(max-min)/2;
    double midval = ComputeCDF(med+mid)-ComputeCDF(med-mid);
    if (midval < alpha) {
      min = mid;
      minval = midval;
    } else if (midval > alpha) {
      max = mid;
      maxval = midval;
    } else
      return mid;
  }
  if (i == BISECT_ITMAX)
    BtgError::Set("Error: could not compute quantile to desired accuracy");
  return min+(max-min)*(alpha-minval)/(maxval-minval);
}

// Compute the estimate phi of the maximum estimated standard error in the Monte Carlo
// estimation of p(z0 | z)
double Predictor::ComputeErrorEstimate(double x, double y)
{
  Precompute_pz0(x, y);
  double max_phi2 = 0;
  for (int j = 0; j < mesh; ++j) {
    double phi = 0;
    double z0 = j*(rangemax-rangemin)/mesh+rangemin;
    double pz0 = Compute_pz0(z0);
    for (int i = 0; i < m; ++i) {
      double x = (Compute_pz0(z0, i) - pz0) * pdata[i].pz;
      phi += x*x;
    }
    if (phi > max_phi2)
      max_phi2 = phi;
  }
  return sqrt(max_phi2 / m);
}

// Cross validate by deleting the kth point (zero indexed) and then computing the
// predictive distribution at that point with the remaining data
void Predictor::CrossValidate(int k, double &x, double &y, double &actual, double &median, double &lb, double &rb)
{
  actual = z[k];
  x = sx[k];
  y = sy[k];
  z[k] = z[n-1];
  sx[k] = sx[n-1];
  sy[k] = sy[n-1];
  n--;
  Precompute();
  InitParams();
  Precompute_pz0(x, y);
  ComputeStats(median, lb, rb);
  n++;
  sx[k] = x;
  sy[k] = y;
  z[k] = actual;
}

// Compute the constrained maximum a posteriori estimates for the parameter vector
void Predictor::EstimateModel(double &lambda, double *theta, int &ntheta_arg)
{
  int i,j;
  double x[1+MAX_NTHETA];
  double x_out[1+MAX_NTHETA];
  int approx_mle = 0;
  for (i = 1; i < m; ++i)
    if (pdata[i].pz > pdata[approx_mle].pz)
      approx_mle = i;
  x[0] = pdata[approx_mle].lambda;
  for (j = 0; j < ntheta; ++j)
    x[j+1] = pdata[approx_mle].theta[j];

  predictor_link = this;
  hooke(Opt_pz, 1+ntheta, x, x_out, 0.5, 1e-6, 5000);
  hooke(Opt_pz, 1+ntheta, x, x_out, 0.5, 1e-6, 5000);

  lambda = x_out[0];
  for (j = 0; j < ntheta; ++j)
    theta[j] = x_out[j+1];
  ntheta_arg = ntheta;
}

// Load size data points from the stream is
int Predictor::Load(char *fname)
{
  if (z) {
    delete []z;
    delete []sx;
    delete []sy;
    z = NULL;
    sx = NULL;
    sy = NULL;
  }

  ifstream is(fname);
  if (!is)
    return 0;

  double tmp_x, tmp_y, tmp_z;
  n = 0;
  while (is >> tmp_x >> tmp_y >> tmp_z)
    n++;
  is.close();
  if (n == 0)
    return 0;

  is.open(fname);

  z = new double[n];
  sx = new double[n];
  sy = new double[n];

  for (int i = 0; i < n; ++i)
    is >> sx[i] >> sy[i] >> z[i];

  return 1;
}

// Output the predictive density to the stream os
// If xflag, write the coordinate z0 and p(z0) for each z0
//    else,  write just p(z0) for each z0
void Predictor::Plot(ostream &os, int xFlag)
{
  double step = (rangemax - rangemin) / mesh;
  for (int i = 0; i < mesh; ++i) {
    double z0 = rangemin + i*step;
    if (xFlag)
      os << z0 << ' ';
    os << Compute_pz0(z0) << endl;
  }
}

// Output maps of the median predictions and the uncertainty (quarter 95% pi width) over a grid on
// a rectangular region (x1,y1)-(x2,y2) using step size step.
// If xFlag, write the maps in x y z triplets
//    else,  write the maps in matrix form, where the matrix indices (i,j) correspond to the (x,y) grid points
void Predictor::Map(double xmin, double xmax, double ymin, double ymax, double step,
		    ostream &osmap, ostream &oserr, int xFlag)
{
  for (double y = ymin; y < ymax; y += step) {
    for (double x = xmin; x < xmax; x += step) {
      if (BtgError::Status())
	return;
      double median;
      double errwidth = 0;
      int nearFlag = 0;
      for (int i = 0; i < n; ++i) {
	if (fabs(sy[i]-y) + fabs(sx[i]-x) < step) {
	  median = z[i];
	  nearFlag = 1;
	}
      }
      if (!nearFlag) {
	double lerr, rerr;
	Precompute_pz0(x, y);
	ComputeStats(median, lerr, rerr);
	errwidth = rerr-lerr;
      }
      if (xFlag) {
	osmap << x << ' ' << y << ' ' << median << endl;
	oserr << x << ' ' << y << ' ' << errwidth/4 << endl;
      } else {
	osmap << median << ' ';
	oserr << errwidth/4 << ' ';
      }
    }
    if (!xFlag) {
      osmap << endl;
      oserr << endl;
    }
  }
}

// Get the ranges of the x,y,z values of the current data set
void Predictor::GetRanges(double &xmin, double &xmax,
			  double &ymin, double &ymax,
			  double &zmin, double &zmax)
{
  if (!z) {
    xmin = xmax = 0;
    ymin = ymax = 0;
    zmin = zmax = 0;
    return;
  }
  xmin = sx[0];
  xmax = sx[0];
  ymin = sy[0];
  ymax = sy[0];
  zmin = z[0];
  zmax = z[0];
  for (int i = 1; i < n; ++i) {

    if (sx[i] < xmin)
      xmin = sx[i];
    else if (sx[i] > xmax)
      xmax = sx[i];

    if (sy[i] < ymin)
      ymin = sy[i];
    else if (sy[i] > ymax)
      ymax = sy[i];

    if (z[i] < zmin)
      zmin = z[i];
    else if (z[i] > zmax)
      zmax = z[i];

  }
}
