#include "decG_utils.h"
//#include <gsl/gsl_rng.h>
//#include <gsl/gsl_randist.h>
#include "omp.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <cstdlib>

using namespace std;

PLK::PLK()
{
	// Nothing to be done
}

PLK::~PLK()
{
    // Nothing to be done
}


/****************************************************************************/
// Compute matrix transposition with nr rows and nc columns
/****************************************************************************/

void PLK::transpose(double* A, double* tA, int nr, int nc)
{
    for (int i=0;i<nr;i++)
    {
        for (int j=0;j<nc;j++)
        {
            tA[j*nr + i] = A[i*nc + j];
        }
    }
}

/****************************************************************************/
// Compute matrix - matrix multiplication with nr rows and nc columns nc2 columns
/****************************************************************************/

void PLK::MM_multiply(double* A, double* B, double *AB, int nr, int nc, int nc2)
{
    double temp = 0;
    for (int i=0;i<nr;i++)
    {
        for (int j=0;j<nc2;j++)
        {
            temp = 0;
            for (int k=0;k<nc;k++)
            {
                temp += A[i*nc + k]*B[k*nc2 + j];
            }
            AB[i*nc2 + j] = temp;
        }
    }

}

void PLK::Mcopy(double* A, double* cA, int nr, int nc)
{
    for (int i=0;i<nr*nc;i++)
    {
        cA[i] = A[i];
    }
}

/****************************************************************************/
// Compute matrix - vector multiplication with nr rows and nc columns
/****************************************************************************/

void PLK::MV_multiply(double* A, double* b, double *Ab, int nr, int nc)
{
    double temp = 0;
    for (int i=0;i<nr;i++)
    {
        temp = 0;
        for (int k=0;k<nc;k++)
        {
            temp += A[i*nc + k]*b[k];
        }
        Ab[i] = temp;
    }
}

void PLK::Basic_CG(double* Z, double* X, double* S, int nc, int niter_max)
{
   	double tol=1e-16;
	  int i,j;
	  bool Go_on = true;
	double Acum =0;
	double alpha,beta,deltaold;
    double* R = (double *) malloc(sizeof(double)*nc);
    double* D = (double *) malloc(sizeof(double)*nc);
    double* Q = (double *) malloc(sizeof(double)*nc);
    double* tempS = (double *) malloc(sizeof(double)*nc);

    int numiter = 0;

    double delta = 0;

    for (i=0;i<nc;i++)
    {
        tempS[i] = 0; // Initialization (Could be different)
     	R[i] = X[i];
     	delta += R[i]*R[i];
     	//delta0 += R[i]*R[i];
     	D[i] = X[i];
    }

    double delta0 = delta;
    double bestres = std::sqrt(delta/delta0);

    while (Go_on == true)
    {
        MV_multiply(Z,D,Q,nc,nc);

        Acum =0;
        for (i=0;i<nc;i++) Acum += D[i]*Q[i];
        alpha = delta/Acum;

        deltaold = delta;
        Acum =0;

        for (i=0;i<nc;i++)
        {
            tempS[i] += alpha*D[i];
     	    R[i] -= alpha*Q[i];
        	Acum += R[i]*R[i];
        }

        delta = Acum;

        beta = delta/deltaold;

        for (i=0;i<nc;i++) D[i] = R[i] + beta*D[i];

        numiter++;

        if (sqrt(delta/delta0) < bestres)
        {
            Mcopy(tempS,S,nc,1);
            bestres = std::sqrt(delta/delta0);
        }

        if (numiter > niter_max) Go_on = false;
        //if (numiter > nc) Go_on = false;
        if (delta < tol*tol*delta0) Go_on == false;

    }

    free(R);
    free(D);
    free(Q);
    free(tempS);
}

/****************************************************************************/
// Conjugate gradient (A S St = X St) -- With local allocation
/****************************************************************************/

void PLK::R_CG2(double* A, double* X, double* S, int nr, int nc, int npix,int niter_max)
{

  double tol=1e-3;
	int i,j;
	bool Go_on = true;
	double Acum =0;
	double alpha,beta,deltaold;

    int numiter = 0;

    double* tS = (double *) malloc(sizeof(double)*nc*npix);
    double* SSt = (double *) malloc(sizeof(double)*nc*npix);
    double* R = (double *) malloc(sizeof(double)*nc*nr);
    double* D = (double *) malloc(sizeof(double)*nc*nr);
    double* B = (double *) malloc(sizeof(double)*nc*nr);
    double* Q = (double *) malloc(sizeof(double)*nc*nr);
    double* tempS = (double *) malloc(sizeof(double)*nc*nr);

    transpose(S,tS,nc,npix);
    MM_multiply(X,tS,B,nr,npix,nc);
    MM_multiply(S,tS,SSt,nc,npix,nc);

    // Initialization (Could be different)

    double delta = 0;

    for (i=0;i<nr*nc;i++)
    {
     	tempS[i] = 0;
     	R[i] = B[i];
     	delta += R[i]*R[i];
     	D[i] = B[i];
    }

    double delta0 = delta;
    double bestres = sqrt(delta/delta0);

    while (Go_on == true)
    {
        MM_multiply(D,SSt,Q,nr,nc,nc);

        Acum =0;
        for (i=0;i<nc*nr;i++) Acum += D[i]*Q[i];
        alpha = delta/Acum;

        deltaold = delta;
        Acum =0;

        for (i=0;i<nc*nr;i++)
        {
          tempS[i] += alpha*D[i];
     	    R[i] -= alpha*Q[i];
        	Acum += R[i]*R[i];
        }

        delta = Acum;

        beta = delta/deltaold;

        for (i=0;i<nc*nr;i++) D[i] = R[i] + beta*D[i];

        numiter++;

        if (sqrt(delta/delta0) < bestres)
        {
            for (i=0;i<nc*nr;i++) A[i] = tempS[i];
            bestres = sqrt(delta/delta0);
        }

        if (numiter > niter_max) Go_on = false;
        if (numiter > nc) Go_on = false;
        if (delta < tol*tol*delta0) Go_on == false;

    }

    free(tS);
    free(SSt);
    free(R);
    free(D);
    free(B);
    free(Q);
    free(tempS);
}

/****************************************************************************/

void PLK::R_CG_cpx(double* A, double* Xr, double* Xi, double* Sr, double* Si, int nr, int nc, int npix,int niter_max)
{
    double tol=1e-3;
	int i,j;
	bool Go_on = true;
	double Acum =0;
	double alpha,beta,deltaold;

    int numiter = 0;

    double* tSr = (double *) malloc(sizeof(double)*nc*npix);
    double* tSi = (double *) malloc(sizeof(double)*nc*npix);
    double* SSt = (double *) malloc(sizeof(double)*nc*nc);
    double* SSti = (double *) malloc(sizeof(double)*nc*nc);
    double* B = (double *) malloc(sizeof(double)*nc*nr);
    double* Bi = (double *) malloc(sizeof(double)*nc*nr);
    double* R = (double *) malloc(sizeof(double)*nc*nr);
    double* D = (double *) malloc(sizeof(double)*nc*nr);
    double* Q = (double *) malloc(sizeof(double)*nc*nr);
    double* tempS = (double *) malloc(sizeof(double)*nc*nr);

    transpose(Sr,tSr,nc,npix); // real part
    transpose(Si,tSi,nc,npix); // imaginary part

    MM_multiply(Xr,tSr,B,nr,npix,nc);
    MM_multiply(Xi,tSi,Bi,nr,npix,nc);
    for (i=0;i<nc*nr;i++) B[i] += Bi[i];

    MM_multiply(Sr,tSr,SSt,nc,npix,nc);
    MM_multiply(Si,tSi,SSti,nc,npix,nc);
    for (i=0;i<nc*nc;i++) SSt[i] += SSti[i];

    // Initialization (Could be different)

    double delta = 0;

    for (i=0;i<nr*nc;i++)
    {
     	tempS[i] = 0;
     	R[i] = B[i];
     	delta += R[i]*R[i];
     	D[i] = B[i];
    }

    double delta0 = delta;
    double bestres = sqrt(delta/delta0);

    while (Go_on == true)
    {
        MM_multiply(D,SSt,Q,nr,nc,nc);

        Acum =0;
        for (i=0;i<nc*nr;i++) Acum += D[i]*Q[i];
        alpha = delta/Acum;

        deltaold = delta;
        Acum =0;

        for (i=0;i<nc*nr;i++)
        {
          tempS[i] += alpha*D[i];
     	    R[i] -= alpha*Q[i];
        	Acum += R[i]*R[i];
        }

        delta = Acum;

        beta = delta/deltaold;

        for (i=0;i<nc*nr;i++) D[i] = R[i] + beta*D[i];

        numiter++;

        if (sqrt(delta/delta0) < bestres)
        {
            for (i=0;i<nc*nr;i++) A[i] = tempS[i];
            bestres = sqrt(delta/delta0);
        }

        if (numiter > niter_max) Go_on = false;
        if (numiter > nc) Go_on = false;
        if (delta < tol*tol*delta0) Go_on == false;

    }

    free(tSr);
    free(tSi);
    free(SSt);
    free(SSti);
    free(R);
    free(D);
    free(B);
    free(Bi);
    free(Q);
    free(tempS);
}

/****************************************************************************/

double PLK::PowerMethod(double* AtA,double* AtAx, double* x, int nr, int nc)
{
	double Acum=0;
	int i,k;
	int niter = 50;

	// Pick x at random

	srand( time(0));

	for (i=0;i < nc;i++) x[i] = std::rand() % 10-1;  //We  might need to check if it works or not

	// Compute the transpose of A

	for (k=0; k < niter ; k++)
    {

      MV_multiply(AtA,x,AtAx,nc,nr);

      Acum=0;
      for (i=0;i < nc;i++) Acum += AtAx[i]*AtAx[i];

      if (Acum < 1e-16)  // Re-Run
      {
           srand( time(0));
	       for (i=0;i < nc;i++) x[i] = std::rand() % 10 - 1;
      }

      if (Acum > 1e-16)  // PK
      {
           for (i=0;i < nc;i++) x[i] = AtAx[i]/(std::sqrt(Acum)+1e-12);
      }

    }

    return std::sqrt(Acum);
}
