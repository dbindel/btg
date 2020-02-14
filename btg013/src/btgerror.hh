//==========================================================================
// btg - Bayesian Trans-Gaussian Prediction program
//==========================================================================

// btgerror.hh

// Written by David Bindel

// Error handling mechanism for the btg program

//==========================================================================

#ifndef _BTGERROR_H
#define _BTGERROR_H

// NOTE: This particular error handling mechanism is kludgy, and would
//       be better implemented using C++ exception handling.  However,
//       that part of the C++ standard keeps changing, and many compilers
//       still have inadequate support for the exception handling mechanism.

// Class to register errors
class BtgError {
private:

  static char error_string[256];  // Saved error message
  static int error_status;        // 1 if error, 0 if no error

public:

  static void Set(char *s);  // Set an error
  static void Clear();       // Clear error status
  static int Status();       // Return error status
  static char *Message();    // Return error message

};

#endif
