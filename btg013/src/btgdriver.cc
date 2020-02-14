//==========================================================================
// btg - Bayesian Trans-Gaussian Prediction program
//==========================================================================

// btgdriver.cc

// Written by David Bindel

// Note: The driver routines are written in a style closer to C than C++
//       to facilitate interactions with other languages.  In particular,
//       each of the BTG_CMD routines at one time implemented a Tcl command
//       stored in a dynamically loadable package.
//       Dynamically loaded packages are not supported before Tcl 7.6 and
//       compiling an integrated front end proved to be more of a problem
//       than it was worth (particularly since some of the machines I used
//       weren't properly installed).  However, this driver code should still
//       be flexible enough that it can be modified to integrate the program
//       functionality into other languages/packages

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream.h>
#include <fstream.h>

#include "btgfuncs.hh"
#include "btg013.hh"
#include "btgerror.hh"

//==========================================================================

// Macros used to define the interface to btg driver command routines

#define BTG_CMD(s) int s(int argc, char *argv[], char *&return_buf, Predictor &predictor, DriverSettings &settings)
#define BTG_ERROR 0
#define BTG_OK 1

#define CHECK_BTG_ERROR \
  if (BtgError::Status()) { \
    strcpy(return_buf, BtgError::Message()); \
    BtgError::Clear(); \
    return BTG_ERROR; \
  }

// Size to use for buffers

const int BUF_SIZ = 200;

//==========================================================================

// Global variables
// If anyone decides to modify this code, these should probably be put
// into a structure.  These are global from when all this code was in a
// package, and the settings variables were bound to Tcl variables.

struct DriverSettings {
  double x_min, x_max;
  double y_min, y_max;
  double z_min, z_max;

  double range_min;
  double range_max;
  int mesh_size;
  int sample_size;
  int trend_order;
  char correlation[BUF_SIZ];
  double lambda_min;
  double lambda_max;

  int data_loaded;           // Program data file is loaded (0/1)
  int needs_initialization;  // Prediction engine needs initialization (0/1)
};

// Trim Tcl quotes (curly braces) off a buffer

char *TrimQuote(char *buf);

// Change engine settings/initialize the engine.
// Return NULL on success, else error message.

char *SetEngine(Predictor &predictor, DriverSettings &settings);
char *InitEngine(Predictor &predictor, DriverSettings &settings);

// Command routines

BTG_CMD(LoadData);
BTG_CMD(Init);
BTG_CMD(Plot);
BTG_CMD(Pred);
BTG_CMD(ErrorEstimate);
BTG_CMD(Model);
BTG_CMD(Validate);
BTG_CMD(Map);
BTG_CMD(Update);
BTG_CMD(Refresh);

//==========================================================================

// Trim Tcl quotes
char *TrimQuote(char *buf)
{
  if (*buf == '{') {
    buf++;
    buf[strlen(buf)-1] = '\0';
  }
  return buf;
}

// Update engine settings
char *SetEngine(Predictor &predictor, DriverSettings &settings)
{
  settings.needs_initialization = 1;
  char *error_string = NULL;

  if (strcmp(settings.correlation, "exponential") == 0)
    predictor.SetCorrelation(ExpCorr, 2, 0.0,1.0, 0.0,2.0);
  else if (strcmp(settings.correlation, "matern") == 0)
    predictor.SetCorrelation(MaternCorr, 2, 0.0,1.0, 0.0,1.0);
  else if (strcmp(settings.correlation, "rational") == 0)
    predictor.SetCorrelation(RationalCorr, 2, 0.0,1.0, 0.0,1.0);
  else if (strcmp(settings.correlation, "spherical") == 0)
    predictor.SetCorrelation(SphericalCorr, 1, 0.0,1.0);
  else {
    error_string = "Error: invalid correlation function";
    return error_string;
  }

  if (settings.trend_order == 0)
    predictor.SetCovariates(1, UnitFn);
  else if (settings.trend_order == 1)
    predictor.SetCovariates(3, UnitFn, XFn, YFn);
  else if (settings.trend_order == 2)
    predictor.SetCovariates(6, UnitFn, XFn, YFn, XYFn, XXFn, YYFn);
  else {
    error_string = "Error: trend order must be 0, 1, or 2";
    return error_string;
  }

  predictor.SetLambdaRange(settings.lambda_min, settings.lambda_max);
  predictor.SetSampleSize(settings.sample_size);
  predictor.SetRange(settings.range_min, settings.range_max);
  predictor.SetSampleSize(settings.sample_size);
  predictor.SetMeshSize(settings.mesh_size);

  return error_string;
}

// Reinitialize the engine
char *InitEngine(Predictor &predictor, DriverSettings &settings)
{
  char *error_string = NULL;
  if (!settings.data_loaded)
    error_string = "Error: no data loaded";
  else if (settings.needs_initialization) {
    predictor.InitComputations();
    if (BtgError::Status()) {
      error_string = BtgError::Message();
      BtgError::Clear();
    }
    settings.needs_initialization = 0;
  }
  return error_string;
}

//==========================================================================

// Load a data file
// Argument list: file
// Output list: x min, y min, x max, y max, z min, z max, filename
BTG_CMD(LoadData)
{
  if (argc != 2) {
    return_buf = "Usage: load [file]";
    return BTG_ERROR;
  }
  if (predictor.Load(argv[1])) {
    settings.data_loaded = 1;
    predictor.GetRanges(settings.x_min, settings.x_max, settings.y_min, settings.y_max, settings.z_min, settings.z_max);
    sprintf(return_buf, "%g %g %g %g %g %g \"%s\"",
	    settings.x_min, settings.x_max, settings.y_min, settings.y_max, settings.z_min, settings.z_max, argv[1]);
  } else {
    settings.data_loaded = 0;
    return_buf = "Error: could not load file.";
    return BTG_ERROR;
  }
  return BTG_OK;
}

// Explicitly reinitialize the prediction engine
// Argument list: none
// Output list: none
BTG_CMD(Init)
{
  if (argc != 1) {
    return_buf = "Usage: initialize";
    return BTG_ERROR;
  }
  settings.needs_initialization = 1;
  char *error_string = InitEngine(predictor, settings);
  if (error_string != NULL) {
    return_buf = error_string;
    return BTG_ERROR;
  }
  return BTG_OK;
}

// Plot the predictive density at a point
// Argument list: x, y, filename, xflag
// Output list: none
BTG_CMD(Plot)
{
  double x, y;
  char *error_string = InitEngine(predictor, settings);
  if (error_string != NULL) {
    return_buf = error_string;
    return BTG_ERROR;
  }
  if (argc != 5) {
    return_buf = "Usage: plot [x] [y] [filename] [xflag]";
    return BTG_ERROR;
  }
  x = atof(argv[1]);
  y = atof(argv[2]);

  ofstream os(argv[3]);
  if (!os) {
    sprintf(return_buf, "Error: could not open %s", argv[3]);
    return BTG_ERROR;
  }

  predictor.Precompute_pz0(x, y);
  CHECK_BTG_ERROR
  predictor.Plot(os, atoi(argv[4]));
  CHECK_BTG_ERROR
  return BTG_OK;
}

// Predict at a point
// Argument list: x, y
// Output list: median, lower PI bound, upper PI bound
BTG_CMD(Pred)
{
  double x, y;
  double med, lb, rb;
  char tmp_buf[40];
  char *error_string = InitEngine(predictor, settings);
  if (error_string != NULL) {
    return_buf = error_string;
    return BTG_ERROR;
  }
  if (argc != 3) {
    return_buf = "Usage: predict [x] [y]";
    return BTG_ERROR;
  }
  x = atof(argv[1]);
  y = atof(argv[2]);

  predictor.Precompute_pz0(x, y);
  CHECK_BTG_ERROR
  predictor.ComputeStats(med, lb, rb);
  CHECK_BTG_ERROR
  sprintf(return_buf, "%g %g %g", med, lb, rb);
  return BTG_OK;
}

// Estimate Monte Carlo error in p(z0 | z)
// Argument list: x, y
// Output list: phi
BTG_CMD(ErrorEstimate)
{
  double x, y;
  char *error_string = InitEngine(predictor, settings);
  if (error_string != NULL) {
    return_buf = error_string;
    return BTG_ERROR;
  }
  if (argc != 3) {
    return_buf = "Usage: phi [x] [y]";
    return BTG_ERROR;
  }
  x = atof(argv[1]);
  y = atof(argv[2]);

  predictor.Precompute_pz0(x, y);
  CHECK_BTG_ERROR
  sprintf(return_buf, "%g", predictor.ComputeErrorEstimate(x, y));
  CHECK_BTG_ERROR
  return BTG_OK;
}

// Estimate model parameters
// Argument list: none
// Output list: lambda, theta (list)
BTG_CMD(Model)
{
  int ntheta;
  char tmp_buf[40];
  double lambda, theta[MAX_NTHETA];
  char *error_string = InitEngine(predictor, settings);
  if (error_string != NULL) {
    return_buf = error_string;
    return BTG_ERROR;
  }
  if (argc != 1) {
    return BTG_ERROR;
  }

  predictor.EstimateModel(lambda, theta, ntheta);
  CHECK_BTG_ERROR

  sprintf(tmp_buf, "%g ", lambda);
  strcat(return_buf, tmp_buf);
  for (int j = 0; j < ntheta; ++j) {
    sprintf(tmp_buf, "%g ", theta[j]);
    strcat(return_buf, tmp_buf);
  }
  return BTG_OK;
}

// Cross-validate
// Argument list: none
// Output list: none
BTG_CMD(Validate)
{
  char *error_string = InitEngine(predictor, settings);
  if (error_string != NULL) {
    return_buf = error_string;
    return BTG_ERROR;
  }
  if (argc != 2) {
    return_buf = "Usage: validate [filename]";
    return BTG_ERROR;
  }

  int hit = 0;
  double x, y;
  double actual, median, lb, rb;
  double diff, residual, sqres;
  int n = predictor.GetDataSize();
  sqres = 0;
  ofstream os(argv[1]);
  if (!os) {
    sprintf(return_buf, "Error: could not open %s", argv[1]);
    return BTG_ERROR;
  }
  for (int k = 0; k < n; ++k) {
    predictor.CrossValidate(k, x, y, actual, median, lb, rb);
    CHECK_BTG_ERROR
    diff = actual-median;
    residual = 4*diff / (lb-rb);
    sqres += diff*diff;
    os << "Cross validation at: " << x << ' ' << y << endl;
    os << "Actual: " << actual << endl;
    os << "Prediction: " << median << endl;
    os << "95% PI: (" << lb << ", " << rb << ")" << endl;
    os << "Residual: " << diff << endl;
    os << "Scaled residual: " << residual << endl;
    os << endl;
    if (lb <= actual && rb >= actual)
      hit++;
  }
  os << "Mean squared difference in measurement: " << sqres/n << endl;
  os << "Prediction interval coverage: " << (float) hit/n << endl;
  predictor.InitComputations();
  CHECK_BTG_ERROR
  return BTG_OK;
}

// Map a region:
// Argument list: x range min, x range max, yrange min, y range max,
//                grid box size, prediction file name, uncertainty file name, 
//                format flag
// Output list: none
BTG_CMD(Map)
{
  double xmin, xmax, ymin, ymax, step;
  char *error_string = InitEngine(predictor, settings);
  if (error_string != NULL) {
    return_buf = error_string;
    return BTG_ERROR;
  }
  if (argc != 9) {
    return_buf = "Usage: map [xmin] [xmax] [ymin] [ymax] [box_size] [prediction_file] [uncertainty_file] [xflag]";
    return BTG_ERROR;
  }
  xmin = atof(argv[1]);
  xmax = atof(argv[2]);
  ymin = atof(argv[3]);
  ymax = atof(argv[4]);
  step = atof(argv[5]);

  ofstream medianfile(argv[6]);
  if (!medianfile) {
    sprintf(return_buf, "Error: could not open %s", argv[3]);
    return BTG_ERROR;
  }
  ofstream errfile(argv[7]);
  if (!errfile) {
    sprintf(return_buf, "Error: could not open %s", argv[3]);
    return BTG_ERROR;
  }
  predictor.Map(xmin, xmax, ymin, ymax, step, medianfile, errfile, atoi(argv[8]));
  CHECK_BTG_ERROR
  return BTG_OK;
}

// Update settings
// Argument list: range min, range max, mesh size, sample size
//                trend order, lambda min, lambda max, correlation name
// Output list: none
BTG_CMD(Update)
{
  if (argc != 9) {
    return_buf = "Usage: update ...";
    return BTG_ERROR;
  }

  settings.needs_initialization = 1;
  settings.range_min = atof(argv[1]);
  settings.range_max = atof(argv[2]);
  settings.mesh_size = atoi(argv[3]);
  settings.sample_size = atoi(argv[4]);
  settings.trend_order = atoi(argv[5]);
  settings.lambda_min = atof(argv[6]);
  settings.lambda_max = atof(argv[7]);
  strcpy(settings.correlation, TrimQuote(argv[8]));

  char *error_string = SetEngine(predictor, settings);
  if (error_string != NULL) {
    return_buf = error_string;
    return BTG_ERROR;
  }
  return BTG_OK;
}

// Return current settings
// Argument list: none
// Output list: range min, range max, mesh size, sample size
//              trend order, lambda min, lambda max, correlation name
BTG_CMD(Refresh)
{
  static char local_buf[800];
  if (argc != 1) {
    return_buf = "Usage: refresh";
    return BTG_ERROR;
  }
  sprintf(local_buf, "%g %g %d %d %d %g %g \"%s\"",
	  settings.range_min, settings.range_max,
	  settings.mesh_size, settings.sample_size, settings.trend_order,
	  settings.lambda_min, settings.lambda_max,
	  settings.correlation);
  return_buf = local_buf;
  return BTG_OK;
}

//==========================================================================

// Tokenize an input string
void ParseArgs(char *s, int &argc, char **argv)
{
  argc = 0;
  for (char *cur_tok = strtok(s, " "); cur_tok != NULL; cur_tok = strtok(NULL, " "))
    argv[argc++] = cur_tok;
}

// Main routine
int main()
{
  DriverSettings settings;
  Predictor predictor;
  settings.data_loaded = 0;
  settings.needs_initialization = 1;
  //  predictor.Init();

  char tmp_buf[BUF_SIZ];
  char general_return_buf[BUF_SIZ];
  int argc;
  char *argv[20];
  while (cin.getline(tmp_buf, BUF_SIZ)) {
    ParseArgs(tmp_buf, argc, argv);
    if (argc != 0) {
      strcpy(general_return_buf, "");
      char *return_buf = general_return_buf;
      int return_status;
      if (strcmp(argv[0], "load") == 0)
	return_status = LoadData(argc, argv, return_buf, predictor, settings);
      else if (strcmp(argv[0], "initialize") == 0)
	return_status = Init(argc, argv, return_buf, predictor, settings);
      else if (strcmp(argv[0], "plot") == 0)
	return_status = Plot(argc, argv, return_buf, predictor, settings);
      else if (strcmp(argv[0], "predict") == 0)
	return_status = Pred(argc, argv, return_buf, predictor, settings);
      else if (strcmp(argv[0], "errorest") == 0)
	return_status = ErrorEstimate(argc, argv, return_buf, predictor, settings);
      else if (strcmp(argv[0], "model") == 0)
	return_status = Model(argc, argv, return_buf, predictor, settings);
      else if (strcmp(argv[0], "validate") == 0)
	return_status = Validate(argc, argv, return_buf, predictor, settings);
      else if (strcmp(argv[0], "map") == 0)
	return_status = Map(argc, argv, return_buf, predictor, settings);
      else if (strcmp(argv[0], "update") == 0)
	return_status = Update(argc, argv, return_buf, predictor, settings);
      else if (strcmp(argv[0], "refresh") == 0)
	return_status = Refresh(argc, argv, return_buf, predictor, settings);
      else {
	return_status = BTG_ERROR;
	return_buf = "Error: invalid command";
      }
      cout << return_status << endl;
      cout << return_buf << endl;
    }
  }

  //  predictor.Shutdown();
  return 0;
}

// This is a hack to fool old versions of F77 and V77,
// for systems where the Fortran libraries must be used.

extern "C" { void MAIN_(); }
void MAIN_()
{
}
