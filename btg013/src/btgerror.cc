//==========================================================================
// btg - Bayesian Trans-Gaussian Prediction program
//==========================================================================

// btgerror.cc

// Written by David Bindel

// btg program error handling code

#include <string.h>
#include <iostream.h>
#include "btgerror.hh"

//==========================================================================

// Error message buffer and status flag

char BtgError::error_string[256];
int BtgError::error_status = 0;

// Set an error
void BtgError::Set(char *s)
{
  strcpy(error_string, s);
  error_status = 1;
}

// Clear error
void BtgError::Clear()
{
  error_status = 0;
}

// Return error status flag
int BtgError::Status()
{
  return error_status;
}

// Return error message
char *BtgError::Message()
{
  return error_string;
}
