#include "matrix_utils_omp.h"
#include "omp.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <iostream>

/****************************************************************************/
// Initialization / destructor
/****************************************************************************/

MATRIX_OMP::MATRIX_OMP(int n_Row,int n_Col, int n_Fixed, int npix,int n_Block,int n_CG_NIter, int n_Iter, double maxts, double mints, bool l1, int usep)
{
    NRow=n_Row;
    NCol = n_Col;
    NBlock = n_Block;
    CG_NIter = n_CG_NIter;
    NIter = n_Iter;
    Npix = npix;
    Max_ts = maxts;
    Min_ts = mints;
    L1 = l1;
    UseP = usep;
    NFixed = n_Fixed;
}

MATRIX_OMP::~MATRIX_OMP()
{
  // To be done
}

/****************************************************************************/
// Copying a matrix
/****************************************************************************/

void MATRIX_OMP::Mcopy(double* A, double* cA, int nr, int nc)
{
    for (int i=0;i<nr*nc;i++)
    {
        cA[i] = A[i];
    }
}

/****************************************************************************/
// Adding matrices
/****************************************************************************/

void MATRIX_OMP::Madd(double* A, double* B, double* ApB, int nr, int nc)
{
    for (int i=0;i<nr*nc;i++)
    {
        ApB[i] = A[i] + B[i];
    }
}

/****************************************************************************/
// Substracting matrices
/****************************************************************************/

void MATRIX_OMP::Msubs(double* A, double* B, double* AmB, int nr, int nc)
{
    for (int i=0;i<nr*nc;i++)
    {
        AmB[i] = A[i] - B[i];
    }
}

/****************************************************************************/
// Updating a residual
/****************************************************************************/

void MATRIX_OMP::UpdateResidual(double* X,double* A,double* S,double* Resi)
{
  double * AS = (double *) malloc(sizeof(double)*NRow*Npix);

  MM_multiply(A, S, AS, NRow,NCol,Npix);
  Msubs(X, AS, Resi, NRow, Npix);

  free(AS);

}

/****************************************************************************/
// Hadamard product of matrices
/****************************************************************************/

void MATRIX_OMP::MHprod(double* A, double* B, double* ApB, int nr, int nc)
{
    for (int i=0;i<nr*nc;i++)
    {
        ApB[i] = A[i]*B[i];
    }
}

/****************************************************************************/
// Hadamard division of matrices
/****************************************************************************/

void MATRIX_OMP::MHdiv(double* A, double* B, double* AdB, int nr, int nc)
{
    for (int i=0;i<nr*nc;i++)
    {
        AdB[i] = A[i]/B[i];
    }
}

/****************************************************************************/
// Compute matrix transposition with nr rows and nc columns
/****************************************************************************/

void MATRIX_OMP::transpose(double* A, double* tA, int nr, int nc)
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

void MATRIX_OMP::MM_multiply(double* A, double* B, double *AB, int nr, int nc, int nc2)
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

/****************************************************************************/
// Compute matrix - vector multiplication with nr rows and nc columns
/****************************************************************************/

void MATRIX_OMP::MV_multiply(double* A, double* b, double *Ab, int nr, int nc)
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

/****************************************************************************/
// Conjugate gradient (A S St = X St)
/****************************************************************************/

void MATRIX_OMP::R_CG(double* A, double* X, double* S, int nr, int nc, int npix,int niter_max,  double* tS,double* SSt, double* R, double* D, double* B, double* Q, double* tempS)
{

  double tol=1e-3;
	int i,j;
	bool Go_on = true;
	double Acum =0;
	double alpha,beta,deltaold;

    int numiter = 0;

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
}

/****************************************************************************/
// Conjugate gradient (A S St = X St) -- With local allocation
/****************************************************************************/

void MATRIX_OMP::R_CG2(double* A, double* X, double* S, int nr, int nc, int npix,int niter_max)
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

    //for (i=0;i<nc;i++) SSt[i+nc*i] += 1e-12; // Regularizing

    // Initialization (Could be different)

    double delta = 1e-24;

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
        alpha = delta/(1e-24 + Acum);

        deltaold = delta;
        Acum =0;

        for (i=0;i<nc*nr;i++)
        {
          tempS[i] += alpha*D[i];
     	    R[i] -= alpha*Q[i];
        	Acum += R[i]*R[i];
        }

        delta = Acum;

        beta = delta/(1e-24 + deltaold);

        for (i=0;i<nc*nr;i++) D[i] = R[i] + beta*D[i];

        numiter++;

        if (sqrt(delta/delta0) < bestres)
        {
            for (i=0;i<nc*nr;i++) A[i] = tempS[i];
            bestres = sqrt(delta/(1e-24+delta0));
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



void MATRIX_OMP::R_CG_AMCA(double* A, double* X, double* S, double * Weights, int nr, int nc, int npix,int niter_max,  double* tS,double* SSt, double* R, double* D, double* B, double* Q, double* tempS)
{

    double tol=1e-3;
	int i,j;
	bool Go_on = true;
	double Acum =0;
	double alpha,beta,deltaold;

    int numiter = 0;

    for (int i=0;i<nc;i++)
    {
        for (int j=0;j<npix;j++)
        {
            tS[j*nc + i] = S[j + i*npix]*Weights[j];
        }
    }
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
            Mcopy(tempS,A,nr,nc);
            bestres = sqrt(delta/delta0);
        }

        if (numiter > niter_max) Go_on = false;
        if (numiter > nc) Go_on = false;
        if (delta < tol*tol*delta0) Go_on == false;

    }
}


/****************************************************************************/
// Conjugate gradient (AtA S = AtX)
/****************************************************************************/

void MATRIX_OMP::L_CG(double* A, double* X, double* S, int nr, int nc, int npix, int niter_max, double* tA, double* AtA, double* R, double* D, double* B, double* Q, double* tempS)
{
   	double tol=1e-6;
	int i,j;
	bool Go_on = true;
	double Acum =0;
	double alpha,beta,deltaold;

    //double* tA = (double *) malloc(sizeof(double)*nc*nr);
    //double* AtA = (double *) malloc(sizeof(double)*nc*nc);
    //double* R = (double *) malloc(sizeof(double)*nc*npix);
    //double* D = (double *) malloc(sizeof(double)*nc*npix);
    //double* B = (double *) malloc(sizeof(double)*nc*npix);
    //double* Q = (double *) malloc(sizeof(double)*nc*npix);
    //double* tempS = (double *) malloc(sizeof(double)*nc*npix);

    int numiter = 0;

    transpose(A,tA,nr,nc);
    MM_multiply(tA,X,B,nc,nr,npix);
    MM_multiply(tA,A,AtA,nc,nr,nc);

    //free(tA);

    double delta = 0;

    for (i=0;i<nc*npix;i++)
    {
      tempS[i] = 0; // Initialization (Could be different)
     	R[i] = B[i];
     	delta += R[i]*R[i];
     	//delta0 += R[i]*R[i];
     	D[i] = B[i];
    }

    double delta0 = delta;
    double bestres = sqrt(delta/delta0);

    while (Go_on == true)
    {
        MM_multiply(AtA,D,Q,nc,nc,npix);

        Acum =0;
        for (i=0;i<nc*npix;i++) Acum += D[i]*Q[i];
        alpha = delta/Acum;

        deltaold = delta;
        Acum =0;

        for (i=0;i<nc*npix;i++)
        {
            tempS[i] += alpha*D[i];
     	    R[i] -= alpha*Q[i];
        	Acum += R[i]*R[i];
        }

        delta = Acum;

        beta = delta/deltaold;

        for (i=0;i<nc*npix;i++) D[i] = R[i] + beta*D[i];

        numiter++;

        if (sqrt(delta/delta0) < bestres)
        {
            Mcopy(tempS,S,nc,npix);
            bestres = sqrt(delta/delta0);
        }

        if (numiter > niter_max) Go_on = false;
        if (numiter > nc) Go_on = false;
        if (delta < tol*tol*delta0) Go_on == false;

    }

    //free(AtA);
    //free(R);
    //free(D);
    //free(B);
    //free(Q);
    //free(tempS);
}

/****************************************************************************/
// Conjugate gradient (AtA S = AtX)
/****************************************************************************/

void MATRIX_OMP::L_CG2(double* A, double* X, double* S, int nr, int nc, int npix, int niter_max)
{
   	double tol=1e-6;
    int i,j;
    bool Go_on = true;
    double Acum =0;
    double alpha,beta,deltaold;

    double* tA = (double *) malloc(sizeof(double)*nc*nr);
    double* AtA = (double *) malloc(sizeof(double)*nc*nc);
    double* R = (double *) malloc(sizeof(double)*nc*npix);
    double* D = (double *) malloc(sizeof(double)*nc*npix);
    double* B = (double *) malloc(sizeof(double)*nc*npix);
    double* Q = (double *) malloc(sizeof(double)*nc*npix);
    double* tempS = (double *) malloc(sizeof(double)*nc*npix);

    int numiter = 0;

    transpose(A,tA,nr,nc);
    MM_multiply(tA,X,B,nc,nr,npix);
    MM_multiply(tA,A,AtA,nc,nr,nc);

    //for (i=0;i<nc;i++) AtA[i+nc*i] += 1e-24;

    //free(tA);

    double delta = 1e-24;

    for (i=0;i<nc*npix;i++)
    {
      tempS[i] = 0; // Initialization (Could be different)
     	R[i] = B[i];
     	delta += R[i]*R[i];
     	//delta0 += R[i]*R[i];
     	D[i] = B[i];
    }

    double delta0 = delta;
    double bestres = sqrt(delta/delta0);

    while (Go_on == true)
    {
        MM_multiply(AtA,D,Q,nc,nc,npix);

        Acum =0;
        for (i=0;i<nc*npix;i++) Acum += D[i]*Q[i];
        alpha = delta/(1e-24 + Acum);

        deltaold = delta;
        Acum =0;

        for (i=0;i<nc*npix;i++)
        {
            tempS[i] += alpha*D[i];
     	    R[i] -= alpha*Q[i];
        	Acum += R[i]*R[i];
        }

        delta = Acum;

        beta = delta/(1e-24 + deltaold);

        for (i=0;i<nc*npix;i++) D[i] = R[i] + beta*D[i];

        numiter++;

        if (sqrt(delta/delta0) < bestres)
        {
            Mcopy(tempS,S,nc,npix);
            bestres = sqrt(delta/delta0);
        }

        if (numiter > niter_max) Go_on = false;
        if (numiter > nc) Go_on = false;
        if (delta < tol*tol*delta0) Go_on == false;

    }

    free(tA);
    free(AtA);
    free(R);
    free(D);
    free(B);
    free(Q);
    free(tempS);
}


/****************************************************************************/
// Power method ( for AtA for left application )
/****************************************************************************/

double MATRIX_OMP::PowerMethod(double* A, int nr, int nc, double* x, double* tA, double* Ax, double* AtAx)
{
	float Acum=0;
	//double* x = (double *) malloc(sizeof(double)*nc);
	//double* tA = (double *) malloc(sizeof(double)*nr*nc);
	//double* Ax = (double *) malloc(sizeof(double)*nr);
	//double* AtAx = (double *) malloc(sizeof(double)*nc);
	int i,k;
	int niter = 20;

	// Pick x at random
	for (i=0;i < nc;i++) x[i] = rand() % 2 - 1;

	// Compute the transpose of A
	transpose(A,tA,nr,nc);

	for (k=0; k < niter ; k++)
    {
      // Compute Ax
      MV_multiply(A,x,Ax,nr,nc);

      // Compute AtAx
      MV_multiply(tA,Ax,AtAx,nc,nr);

      Acum=0;
      for (i=0;i < nc;i++) Acum += AtAx[i]*AtAx[i];
      for (i=0;i < nc;i++) x[i] = AtAx[i]/sqrt(Acum);
    }

    //free(x);
    //free(tA);
    //free(Ax);
    //free(AtAx);

    return sqrt(Acum);
}

double MATRIX_OMP::PowerMethod2(double* A, int nr, int nc)
{
	double Acum=0;
  double TotAcum=0;
	double* x = (double *) malloc(sizeof(double)*nc);
	double* tA = (double *) malloc(sizeof(double)*nr*nc);
	double* Ax = (double *) malloc(sizeof(double)*nr);
  double* AtA = (double *) malloc(sizeof(double)*nc*nc);
  double* AtAx = (double *) malloc(sizeof(double)*nc);
	int i,k,z;
	int niter = CG_NIter;
  int nrep = 3;

	// Compute the transpose of A
	transpose(A,tA,nr,nc);
  MM_multiply(tA,A,AtA,nc,nr,nc);

	// Compute the transpose of A

  for (z=0; z < nrep ; z++)
  {
    // Pick x at random
    //srand( time(0));
  	for (i=0;i < nc;i++) x[i] = std::rand() % 1 - 0.5;  //We  might need to check if it works or not

	  for (k=0; k < niter ; k++)
    {
      MV_multiply(AtA,x,AtAx,nc,nc);
      Acum=0;
      for (i=0;i < nc;i++) Acum += std::abs(AtAx[i]*AtAx[i]);
      for (i=0;i < nc;i++) x[i] = AtAx[i]/(std::sqrt(Acum)+1e-16);
    }
    TotAcum += std::sqrt(Acum);
    }

    free(x);
    free(tA);
    free(Ax);
    free(AtAx);
    free(AtA);

    return TotAcum/nrep;
}

/****************************************************************************/
// Vector thresholding
/****************************************************************************/

void MATRIX_OMP::Thresholding(double* V,double Thrd,int npix, bool L1) // We should put some weighting here
{
    for (int i=0;i<npix;i++)
    {
        if (std::abs(V[i]) < Thrd) V[i] = 0;

        if (std::abs(V[i]) > Thrd)
        {
            if (L1 == true)
            {
                if (V[i] < 0)
                {
                    V[i] += Thrd;
                }
                if (V[i] > 0)
                {
                    V[i] -= Thrd;
                }
            }
        }

    }
}

/****************************************************************************/
// Median absolute deviation of a vector
/****************************************************************************/

double MATRIX_OMP::mad(double* V,int npix, double* W)
{
    double mV = median(V,npix,true,W);

    //double* W = (double *) malloc(sizeof(double)*npix);

    // Copying V

    for (int i=0;i<npix;i++)
    {
        W[i] = V[i] - mV;
    }

    return median(W,npix,true,V)/0.6735;
}

/****************************************************************************/
// 2-norm of a vector
/****************************************************************************/

double MATRIX_OMP::TwoNorm(double* V,int npix)
{
    double temp = 0;
    for (int i=0;i<npix;i++)
    {
        temp += V[i]*V[i];
    }
    return temp;
}

/****************************************************************************/
// Mean value of a vector
/****************************************************************************/

double MATRIX_OMP::mean(double* V,int npix)
{
    double temp = 0;
    for (int i=0;i<npix;i++)
    {
        temp += V[i];
    }
    return temp/npix;
}

/****************************************************************************/
// Not quick sort
/****************************************************************************/

void MATRIX_OMP::NoQuickSort(double * array, int n)
{
  double t;
  int i,j;

  for (j=0 ; j<(n-1) ; j++)
  	{
  		for (i=0 ; i<(n-1) ; i++)
  		{
  			if (array[i+1] < array[i])
  			{
  				t = array[i];
  				array[i] = array[i + 1];
  				array[i + 1] = t;
  			}
  		}
  	}
}

void MATRIX_OMP::NoQuickSort_abs(double * array, int n)
{
  double t;
  int i,j;

  for (j=0 ; j<(n-1) ; j++)
  	{
  		for (i=0 ; i<(n-1) ; i++)
  		{
  			if (std::abs(array[i+1]) < std::abs(array[i]))
  			{
  				t = array[i];
  				array[i] = array[i + 1];
  				array[i + 1] = t;
  			}
  		}
  	}
}

int MATRIX_OMP::GetNumElementsThrd(double * array, int n, double thrd)
{
  int i,val;

  for (i=0 ; i<n ; i++)
  	{
      if (std::abs(array[i]) > thrd) val += 1;
  	}

  return val;
}

/****************************************************************************/
// Median value of a vector - second version
/****************************************************************************/

double MATRIX_OMP::median2(double * array, int size)
 {

      NoQuickSort(array, size); // sorting

      return array[size/2];
	}

  /****************************************************************************/
  // L_inf norm
  /****************************************************************************/

  double MATRIX_OMP::max2(double* V,int npix)
  {
      double max = 0;

      // Copying V

      for (int i=0;i<npix;i++)
      {
          if (abs(V[i]) > max) max = abs(V[i]);
      }

      return max;
  }

/****************************************************************************/
// Median absolute deviation of a vector
/****************************************************************************/

double MATRIX_OMP::mad2(double* V,int npix, double* W)
{
    double mV = median2(V,npix);

    // Copying V

    for (int i=0;i<npix;i++)
    {
        W[i] = std::abs(V[i] - mV);
    }

    return median2(W,npix)/0.6735;
}

/****************************************************************************/
// Mean value of a vector
/****************************************************************************/

double MATRIX_OMP::median(double* V,int npix, bool gabs, double* W)
{
    //double* W = (double *) malloc(sizeof(double)*npix);

    // Copying V

    for (int i=0;i<npix;i++)
    {
        if (gabs == false) W[i] = V[i];
        if (gabs == true) W[i] = std::abs(V[i]);
    }

    // Sorting W

    bool desc = false;
    quickSort( W, 0, npix, desc);

    return W[npix/2];
}

/****************************************************************************/
/****************************************************************************/
// SPECIFIC TO GMCA
/****************************************************************************/
/****************************************************************************/


/****************************************************************************/
// Updating the sources
/****************************************************************************/

/**
* @param X : data
* @param A : mixing matrix
* @param S : sources
* @param W : weight matrix
* @param KSuppS : number of elements to be selected in S
* @param L1 : if set, performs soft-thresholding
*/

void MATRIX_OMP::UpdateS(double* X, double* A, double* S, double Kmad, bool L1, int nr, int nc, int npix,double* t_S, double* t_s2, double* t_A, double* t_AtA, double* t_R, double* t_D, double* t_B, double* t_Q, double* t_s, int L0_max)
{

    //double* tempS = (double *) malloc(sizeof(double)*npix);

    double thrd;
    int niter_max = CG_NIter;

    // LS solution using conjugate-gradient

    L_CG(A,X,S,nr,nc,npix,niter_max,t_A, t_AtA, t_R, t_D, t_B, t_Q,t_S); // We need to change the inputs

    // Thresholding the sources

    for (int i=0;i<nc;i++)
    {
       // copying the sources divided by the weights
       for (int j=0;j<npix;j++){
         t_s[j] = S[i*npix+j];
       }

       thrd = mad2(t_s,npix,t_s2);

       if (UseP == 1)
       {
         for (int j=0;j<npix;j++){
           t_s2[j] = S[i*npix+j];
         }
         int nelem = GetNumElementsThrd(t_s2, npix, Min_ts*thrd); // Get the number of non-zero elements
         NoQuickSort_abs(t_s2,npix); // Sort the absolute value of t_s2
         int I = ((double) L0_max)/npix*nelem; // Determine the number of elements to consider
         if (I < 2) I = 2;
         thrd = std::abs(t_s2[npix - I])/Kmad; // it's in ascending order
       }

       // thresholding

       for (int j=0;j<npix;j++)
       {
         if (std::abs(S[i*npix+j]) < Kmad*thrd) S[i*npix+j] = 0;

         if (std::abs(S[i*npix+j]) > Kmad*thrd)
         {
           if (L1 == true)
           {
               if (S[i*npix+j] < 0)
               {
                   S[i*npix+j] += Kmad*thrd;
               }
               if (S[i*npix+j] > 0)
               {
                   S[i*npix+j]-= Kmad*thrd;
               }
           }
         }
       }
    }
}

void UpdatingWeights(double* S,double* t_S, double* W, double q, int npix, int nc)
{
  int i,j;
  double norm;

  // Normalizing the sources

  for (j=0;j<nc;j++)
  {
    norm = 0;
    for (i=0;i<npix;i++){ // look for the max of the amplitude
      if (std::abs(S[i + j*npix])>norm) norm = std::abs(S[i + j*npix]);
    }
    for (i=0;i<npix;i++){ // Normalize
      t_S[i + j*npix] = S[i + j*npix]/norm;
    }
  }

  for (i=0;i<npix;i++){
    norm = 0;
    for (j=0;j<nc;j++)
    {
      norm += std::pow(std::abs(t_S[i + j*npix]),q);
    }
    t_S[i] = std::pow(norm,1./q);
    W[i] = nc/(1e-16+std::pow(norm,1./q));
    if (t_S[i] == 0) W[i] = 0.;
  }
}

void MATRIX_OMP::UpdateS_AMCA(double* X, double* A, double* S, double* Weights, double alpha, double Kmad, bool L1, int nr, int nc, int npix,double* t_S, double* t_s2, double* t_A, double* t_AtA, double* t_R, double* t_D, double* t_B, double* t_Q, double* t_s, int L0_max)
{

    //double* tempS = (double *) malloc(sizeof(double)*npix);

    double thrd;
    int niter_max = CG_NIter;

    // LS solution using conjugate-gradient

    L_CG(A,X,S,nr,nc,npix,niter_max,t_A, t_AtA, t_R, t_D, t_B, t_Q,t_S); // We need to change the inputs

    // Updating the weights

    int i,j;
    double norm;

    // Normalizing the sources

    for (j=0;j<nc;j++)
    {
      norm = 0;
      for (i=0;i<npix;i++){ // look for the max of the amplitude
        if (std::abs(S[i + j*npix])>norm) norm = std::abs(S[i + j*npix]);
      }
      for (i=0;i<npix;i++){ // Normalize
        t_S[i + j*npix] = S[i + j*npix]/norm;
      }
    }

    for (i=0;i<npix;i++){
      norm = 0;
      for (j=0;j<nc;j++)
      {
        norm += std::pow(std::abs(t_S[i + j*npix]),alpha);
      }
      t_R[i] = std::pow(norm,1./alpha);
      Weights[i] = nc/(1e-16+t_R[i]);
      if (t_R[i] == 0) Weights[i] = 0.;
    }

    // Thresholding the sources

    for (int i=0;i<nc;i++)
    {
       // copying the sources divided by the weights
       for (int j=0;j<npix;j++){
         t_s[j] = S[i*npix+j];
       }

       thrd = mad2(t_s,npix,t_s2);

       if (UseP == 1)
       {
         for (int j=0;j<npix;j++){
           t_s2[j] = S[i*npix+j];
         }
         int nelem = GetNumElementsThrd(t_s2, npix, Min_ts*thrd); // Get the number of non-zero elements
         NoQuickSort_abs(t_s2,npix); // Sort the absolute value of t_s2
         int I = ((double) L0_max)/npix*nelem; // Determine the number of elements to consider
         if (I < 2) I = 2;
         thrd = std::abs(t_s2[npix - I])/Kmad; // it's in ascending order
       }

       // thresholding

       for (int j=0;j<npix;j++)
       {
         if (std::abs(S[i*npix+j]) < Kmad*thrd) S[i*npix+j] = 0;

         if (std::abs(S[i*npix+j]) > Kmad*thrd)
         {
           if (L1 == true)
           {
               if (S[i*npix+j] < 0)
               {
                   S[i*npix+j] += Kmad*thrd;
               }
               if (S[i*npix+j] > 0)
               {
                   S[i*npix+j]-= Kmad*thrd;
               }
           }
         }
       }
    }
}

/****************************************************************************/
// Updating the mixing matrix
/****************************************************************************/

void MATRIX_OMP::UpdateA(double* X, double* A, double* S,int nr, int nc, int npix, double* t_A, double* t_S, double* t_SSt, double* t_R, double* t_D, double* t_B, double* t_Q, double* t_S2)
{

    //double* tempA = (double *) malloc(sizeof(double)*nr);
    double L2n = 0;
    int niter_max = CG_NIter;

    // LS solution using conjugate-gradient

    R_CG(A,X,S,nr,nc,npix, niter_max,t_S,t_SSt,  t_R,  t_D,  t_B,  t_Q, t_S2);

    // L2 constraint

    for (int i=0;i<nc;i++)
    {
       // copying the sources divided by the weights

       L2n = 0;
       for (int j=0;j<nr;j++) L2n += A[j*nc+i]*A[j*nc+i];

       // copying back the sources
       for (int j=0;j<nr;j++) A[j*nc+i] =A[j*nc+i]/(sqrt(L2n) + 1e-12);

    }

    //free(tempA);
}


void MATRIX_OMP::UpdateA_AMCA(double* X, double* A, double* S,double* Weights,int nr, int nc, int npix, double* t_A, double* t_S, double* t_SSt, double* t_R, double* t_D, double* t_B, double* t_Q, double* t_S2)
{

    //double* tempA = (double *) malloc(sizeof(double)*nr);
    double L2n = 0;
    int niter_max = CG_NIter;

    // LS solution using conjugate-gradient

    R_CG_AMCA(A,X,S,Weights,nr,nc,npix, niter_max,t_S,t_SSt,  t_R,  t_D,  t_B,  t_Q, t_S2);

    // L2 constraint

    for (int i=0;i<nc;i++)
    {
       // copying the sources divided by the weights

       L2n = 0;
       for (int j=0;j<nr;j++) L2n += A[j*nc+i]*A[j*nc+i];

       // copying back the sources
       for (int j=0;j<nr;j++) A[j*nc+i] =A[j*nc+i]/(sqrt(L2n) + 1e-12);

    }

    //free(tempA);
}

/****************************************************************************/
// PALM
/****************************************************************************/

double MATRIX_OMP::GradientDescent_S(double* X, double* A, double* S, double* iSigma,double alpha)
{
  double * tA = (double *) malloc(sizeof(double)*NRow*NCol);
  double * Resi = (double *) malloc(sizeof(double)*NRow*Npix);
  double * Gs = (double *) malloc(sizeof(double)*NCol*Npix); // Gradient for S
  double * As = (double *) malloc(sizeof(double)*NRow*NCol);
  double * As2 = (double *) malloc(sizeof(double)*NRow*NCol);

  for (int i=0;i<NCol;i++) for (int j=0;j<NRow;j++) As[j*NCol+i] = A[j*NCol+i]*sqrt(iSigma[j]);
  for (int i=0;i<NCol;i++) for (int j=0;j<NRow;j++) As2[j*NCol+i] = A[j*NCol+i]*iSigma[j];

  UpdateResidual(X,A,S,Resi);
  transpose(As2,tA,NRow,NCol);
  MM_multiply(tA, Resi, Gs, NCol, NRow, Npix);
  double L = PowerMethod2(As, NRow, NCol);

  for (int r=0;r<Npix*NCol;r++) S[r] += alpha/L*Gs[r];

  free(As);
  free(As2);
  free(Resi);
  free(tA);
  free(Gs);

  return L;
}

double MATRIX_OMP::GradientDescent_A(double* X, double* A, double* S,double * iSigma, double alpha)
{
  double * tS = (double *) malloc(sizeof(double)*Npix*NCol);
  double * Resi = (double *) malloc(sizeof(double)*NRow*Npix);
  double * Ga = (double *) malloc(sizeof(double)*NRow*NCol); // Gradient for S
  double * Aold = (double *) malloc(sizeof(double)*NRow*NCol);

  UpdateResidual(X,A,S,Resi);
  for (int i=0;i<NRow;i++) for (int j=0;j<Npix;j++) Resi[j*NRow+i] = Resi[j*NRow+i]*iSigma[i];
  transpose(S,tS,NCol,Npix);
  MM_multiply(Resi, tS, Ga, NRow, Npix, NCol);
  double L = PowerMethod2(tS, Npix, NCol);

  for (int r=0;r<NRow*NCol;r++){
    Aold[r] = A[r];
    A[r] += alpha/(1e-12 + L)*Ga[r];
  }

  double NormDiff = 0;
  double NormTot = 0;

  if (NFixed > 0) // Some columns are fixed
  {
    for (int i=0;i<NFixed;i++) for (int j=0;j<NRow;j++) A[j*NCol+i] = Aold[j*NCol+i];
  }

  for (int i=0;i<NCol;i++)
  {
     double L2n = 0;
     for (int j=0;j<NRow;j++) L2n += A[j*NCol+i]*A[j*NCol+i];
     for (int j=0;j<NRow;j++)
     {
       A[j*NCol+i] =A[j*NCol+i]/(sqrt(L2n) + 1e-12);
       NormDiff += (A[j*NCol+i]-Aold[j*NCol+i])*(A[j*NCol+i]-Aold[j*NCol+i]);
       NormTot += Aold[j*NCol+i]*Aold[j*NCol+i];
     }
  }

  free(Resi);
  free(tS);
  free(Ga);
  free(Aold);

  return sqrt(NormDiff/(1e-16 + NormTot));
}


void MATRIX_OMP::PALM_MAIN(double* X, double* A, double* S, double* iSigma)
{

  bool GoOn = true;
  int it = 0;
  double * t_s = (double *) malloc(sizeof(double)*Npix);
  double * t_s2 = (double *) malloc(sizeof(double)*Npix);
  double alpha = 0.9;

  //double normA = TwoNorm(A,NRow*NCol);
  //std::cout << "normA = " << normA << "\n";
  //std::cout << NRow << " - " << NCol << "\n";

  while (GoOn == true){ // main loop

    it += 1;

    // Gradient descent for S

    double L = GradientDescent_S(X, A, S,iSigma, alpha);
    // Prox

    for (int i=0;i<NCol;i++)
    {
       // copying the sources divided by the weights
       for (int j=0;j<Npix;j++){
         t_s[j] = S[i*Npix+j];
       }

       double thrd = mad2(t_s,Npix,t_s2);

       for (int j=0;j<Npix;j++)
       {
         if (std::abs(S[i*Npix+j]) < Max_ts*thrd) S[i*Npix+j] = 0;

         if (std::abs(S[i*Npix+j]) > Max_ts*thrd)
         {
           if (L1 == true)
           {
               if (S[i*Npix+j] < 0)
               {
                   S[i*Npix+j] += Max_ts*thrd;
               }
               if (S[i*Npix+j] > 0)
               {
                   S[i*Npix+j] -= Max_ts*thrd;
               }
           }
         }
       }
    }

    // Gradient descent for A

    double relvar = GradientDescent_A(X,A,S,iSigma,alpha);

    if (it > NIter) GoOn = false;
    if (it > 100){
      if (relvar < 1e-9)
      {
        GoOn = false;
      }
    }
  }  // Main loop

  free(t_s);
  free(t_s2);

}
/****************************************************************************/
// BASIC GMCA
/****************************************************************************/

void MATRIX_OMP::GMCA_BASIC(double* X, double* A, double* S, double* OldA,int nr, int nc, int npix, int nmax, int n_iter,int maxts, int mints, bool L1,double* t_A,double* t_S,double* t_NcNc,double* t_Rs,double* t_Ds,double* t_Bs,double* t_Qs,double* t_Ra,double* t_Da,double* t_Ba,double* t_Qa,double* t_Za, double* t_s, double* t_s2)
{
    // LS solution using conjugate-gradient

    int niter_stat = 250;
    int GoOn = 1;
    int it = 0;
    int i,j;
    int L0_max = 2.;
    int nf = NFixed;
    double dL0 = 1./niter_stat*npix;
    double RelVar = 0;
    double Kmad = maxts;
    double dk = (maxts - mints)/(n_iter-1.);

    for (int i=0;i<nc;i++) for (int j=0;j<nr;j++) OldA[j*nc+i]= A[j*nc+i];

    while (GoOn == 1)
    {

      it += 1;

       // Updating the sources -- What about zero sources ???
       UpdateS(X, A, S, Kmad, L1, nr, nc, npix, t_S, t_s2,t_A,t_NcNc, t_Rs, t_Ds, t_Bs, t_Qs, t_s,L0_max);

       // Updating the mixing matrix
       UpdateA(X, A, S, nr, nc, npix, t_A, t_S, t_NcNc, t_Ra, t_Da,t_Ba, t_Qa, t_Za); // We should also implement a PALM-based version

       if (nf > 0) // Some columns are fixed
       {
         for (int i=0;i<nf;i++) for (int j=0;j<nr;j++) A[j*nc+i] = OldA[j*nc+i];
       }

       // copying back the sources

       RelVar = 0;
       for (int i=0;i<nc;i++) for (int j=0;j<nr;j++) RelVar += abs(OldA[j*nc+i] - A[j*nc+i])/(abs(OldA[j*nc+i]) + 1e-12);

       if (it > 250)
       {
         if (RelVar < 1e-12){GoOn=0;}
       }
       if (it > n_iter){GoOn = 0;}

       // Copying the old matrix

       for (int i=0;i<nc;i++) for (int j=0;j<nr;j++) OldA[j*nc+i]= A[j*nc+i];

       // Update the kmad

       Kmad -= dk;
       L0_max += dL0;
       if (L0_max > npix-1) L0_max = npix-1;
    }
}

/****************************************************************************/
// BASIC GMCA
/****************************************************************************/

void MATRIX_OMP::AMCA_BASIC(double* X, double* A, double* S, double* OldA, double* Weights,double alpha,int nr, int nc, int npix, int nmax, int n_iter,int maxts, int mints, bool L1,double* t_A,double* t_S,double* t_NcNc,double* t_Rs,double* t_Ds,double* t_Bs,double* t_Qs,double* t_Ra,double* t_Da,double* t_Ba,double* t_Qa,double* t_Za, double* t_s, double* t_s2)
{
    // LS solution using conjugate-gradient

    int niter_stat = 250;
    int GoOn = 1;
    int it = 0;
    int i,j;
    int L0_max = 2.;
    double dL0 = 1./niter_stat*npix;
    double RelVar = 0;
    double Kmad = maxts;
    double dk = (maxts - mints)/(n_iter-1.);
    double alpha_c = 1.;
    double dalpha = (alpha_c - alpha)/(n_iter-1.);

    for (int i=0;i<nc;i++) for (int j=0;j<nr;j++) OldA[j*nc+i]= A[j*nc+i];

    while (GoOn == 1)
    {

      it += 1;

       // Updating the sources -- What about zero sources ???
       UpdateS_AMCA(X, A, S, Weights, alpha_c, Kmad, L1, nr, nc, npix, t_S, t_s2,t_A,t_NcNc, t_Rs, t_Ds, t_Bs, t_Qs, t_s,L0_max);

       // Updating the mixing matrix
       UpdateA_AMCA(X, A, S, Weights, nr, nc, npix, t_A, t_S, t_NcNc, t_Ra, t_Da,t_Ba, t_Qa, t_Za); // We should also implement a PALM-based version

       // copying back the sources

       RelVar = 0;
       for (int i=0;i<nc;i++) for (int j=0;j<nr;j++) RelVar += abs(OldA[j*nc+i] - A[j*nc+i])/(abs(OldA[j*nc+i]) + 1e-12);

       if (it > 250)
       {
         if (RelVar < 1e-12){GoOn=0;}
       }
       if (it > n_iter){GoOn = 0;}

       // Copying the old matrix

       for (int i=0;i<nc;i++) for (int j=0;j<nr;j++) OldA[j*nc+i]= A[j*nc+i];

       // Update the kmad

       Kmad -= dk;
       L0_max += dL0;
       alpha_c -= dalpha;
       if (L0_max > npix-1) L0_max = npix-1;
    }
}

/****************************************************************************/
/****************************************************************************/
// QUICKSORT
/****************************************************************************/
/****************************************************************************/

/**
 * Quicksort.
 * @param a - The array to be sorted.
 * @param first - The start of the sequence to be sorted.
 * @param last - The end of the sequence to be sorted.
 * @param descend - If true in descending order
*/

void MATRIX_OMP::quickSort( double* a, int first, int last, bool descend)
{
    int pivotElement;

    if(first < last)
    {
        pivotElement = pivot(a, first, last,descend);
        quickSort(a, first, pivotElement-1,descend);
        quickSort(a, pivotElement+1, last,descend);
    }
}

/**
 * Find and return the index of pivot element.
 * @param a - The array.
 * @param first - The start of the sequence.
 * @param last - The end of the sequence.
 * @return - the pivot element
*/

int MATRIX_OMP::pivot(double* a, int first, int last, bool descend)
{
    int  p = first;
    double pivotElement = a[first];

    for(int i = first+1 ; i <= last ; i++)
    {
        /* If you want to sort the list in the other order, change "<=" to ">" */

        if (descend == true)
        {
            if(a[i] > pivotElement)
            {
                p++;
                swap(a[i], a[p]);
            }
        }

        if (descend == false)
        {
            if(a[i] <= pivotElement)
            {
                p++;
                swap(a[i], a[p]);
            }
        }
    }

    swap(a[p], a[first]);

    return p;
}

/**
 * Swap the parameters.
 * @param a - The first parameter.
 * @param b - The second parameter.
*/
void MATRIX_OMP::swap(double& a, double& b)
{
    double temp = a;
    a = b;
    b = temp;
}

/**
 * Swap the parameters without a temp variable.
 * Warning! Prone to overflow/underflow.
 * @param a - The first parameter.
 * @param b - The second parameter.
*/
void MATRIX_OMP::swapNoTemp(double& a, double& b)
{
    a -= b;
    b += a;// b gets the original value of a
    a = (b - a);// a gets the original value of b
}



/****************************************************************************/
// PALM
/****************************************************************************/

void MATRIX_OMP::CorrectPERM(double* Aref, double* A,double* Aout)
{
  double* Diff = (double *) malloc(sizeof(double)*NCol*NCol);
  int* MaxInd = (int *) malloc(sizeof(int)*NCol);
  int i,j,k,max_i;
  double amax_val;

  L_CG2(Aref, A, Diff, NRow, NCol, NCol, CG_NIter); // Get Diff

  //for (i=0;i<NCol;i++){
  //  for (j=0;j<NCol;j++){
  //    val = 0;
  //    for (k=0;k<NRow;k++){val += Aref[i + k*NCol]*A[j + k*NCol];}
  //    Diff[j+i*NCol] = std::abs(val);
  //  }
  //}

  // Check the max

  for (i=0;i<NCol;i++){
    max_i = 0;
    amax_val = 0;
    for (j=0;j<NCol;j++){
      if (std::abs(Diff[j+i*NCol]) > amax_val){
        max_i = j;
        amax_val = std::abs(Diff[j+i*NCol]);
      }
    }
    MaxInd[i] = max_i;
    // Change sign
    if (amax_val > Diff[max_i+i*NCol]){
       for (j=0;j<NCol;j++){A[max_i+j*NCol] = -A[max_i+j*NCol];}
    }
  }

  for (i=0;i<NCol;i++){
    for (j=0;j<NCol;j++){
      Aout[j+i*NCol] = A[MaxInd[j]+i*NCol];
    }
  }

  free(Diff);
  free(MaxInd);

}
