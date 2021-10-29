#include "matrix_utils.h"
//#include <gsl/gsl_rng.h>
//#include <gsl/gsl_randist.h>
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

MATRIX::MATRIX(int n_Row,int n_Col, int npix,int n_Block,int n_CG_NIter, int n_Iter)
{
    NRow=n_Row;
    NCol = n_Col;
    NBlock = n_Block;
    CG_NIter = n_CG_NIter;
    NIter = n_Iter;
    Npix = npix;
}

MATRIX::~MATRIX()
{
    // Nothing to be done
}

/****************************************************************************/
// Copying a matrix
/****************************************************************************/

void MATRIX::Mcopy(double* A, double* cA, int nr, int nc)
{
    for (int i=0;i<nr*nc;i++)
    {
        cA[i] = A[i];
    }
}

/****************************************************************************/
// Adding matrices
/****************************************************************************/

void MATRIX::Madd(double* A, double* B, double* ApB, int nr, int nc)
{
    for (int i=0;i<nr*nc;i++)
    {
        ApB[i] = A[i] + B[i];
    }
}

/****************************************************************************/
// Substracting matrices
/****************************************************************************/

void MATRIX::Msubs(double* A, double* B, double* AmB, int nr, int nc)
{
    for (int i=0;i<nr*nc;i++)
    {
        AmB[i] = A[i] - B[i];
    }
}

/****************************************************************************/
// Hadamard product of matrices
/****************************************************************************/

void MATRIX::MHprod(double* A, double* B, double* ApB, int nr, int nc)
{
    for (int i=0;i<nr*nc;i++)
    {
        ApB[i] = A[i]*B[i];
    }
}

/****************************************************************************/
// Hadamard division of matrices
/****************************************************************************/

void MATRIX::MHdiv(double* A, double* B, double* AdB, int nr, int nc)
{
    for (int i=0;i<nr*nc;i++)
    {
        AdB[i] = A[i]/B[i];
    }
}

/****************************************************************************/
// Compute matrix transposition with nr rows and nc columns
/****************************************************************************/

void MATRIX::transpose(double* A, double* tA, int nr, int nc)
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

void MATRIX::MM_multiply(double* A, double* B, double *AB, int nr, int nc, int nc2)
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

void MATRIX::MV_multiply(double* A, double* b, double *Ab, int nr, int nc)
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

void MATRIX::R_CG(double* A, double* X, double* S, int nr, int nc, int npix,int niter_max)
{ 
 
    double tol=1e-3;
	int i,j;
	bool Go_on = true;
	double Acum =0;
	double alpha,beta,deltaold;
    double* tS = (double *) malloc(sizeof(double)*nc*npix);
    double* SSt = (double *) malloc(sizeof(double)*nc*nc);
    double* R = (double *) malloc(sizeof(double)*nc*nr);
    double* D = (double *) malloc(sizeof(double)*nc*nr);
    double* B = (double *) malloc(sizeof(double)*nc*nr);
    double* Q = (double *) malloc(sizeof(double)*nc*nr); 
    double* tempS = (double *) malloc(sizeof(double)*nc*nr); 

    int numiter = 0;
    
    transpose(S,tS,nc,npix);
    MM_multiply(X,tS,B,nr,npix,nc);
    MM_multiply(S,tS,SSt,nc,npix,nc);
    
    free(tS);

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
    
    free(SSt);
    free(R);
    free(D);
    free(B);
    free(Q);
    free(tempS);

    
}

/****************************************************************************/
// Conjugate gradient (AtA S = AtX) 
/****************************************************************************/

void MATRIX::L_CG(double* A, double* X, double* S, int nr, int nc, int npix, int niter_max)
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
    
    free(tA);
    
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

double MATRIX::PowerMethod(double* A, int nr, int nc)
{  
	float Acum=0;
	double* x = (double *) malloc(sizeof(double)*nc); 
	double* tA = (double *) malloc(sizeof(double)*nr*nc); 
	double* Ax = (double *) malloc(sizeof(double)*nr);
	double* AtAx = (double *) malloc(sizeof(double)*nc);
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
    
    free(x);
    free(tA);
    free(Ax);
    free(AtAx);
    
    return sqrt(Acum);
}

/****************************************************************************/
// Vector thresholding 
/****************************************************************************/

void MATRIX::Thresholding(double* V,double Thrd,int npix, bool L1) // We should put some weighting here
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

double MATRIX::mad(double* V,int npix)
{ 
    double mV = median(V,npix,true);
    
    double* W = (double *) malloc(sizeof(double)*npix);
    
    // Copying V
    
    for (int i=0;i<npix;i++)
    {
        W[i] = V[i] - mV;
    }
    
    return median(W,npix,true)/0.6735;
}

/****************************************************************************/
// 2-norm of a vector
/****************************************************************************/

double MATRIX::TwoNorm(double* V,int npix)
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

double MATRIX::mean(double* V,int npix)
{  
    double temp = 0;
    for (int i=0;i<npix;i++)
    {
        temp += V[i];
    }
    return temp/npix;
}

/****************************************************************************/
// Mean value of a vector
/****************************************************************************/

double MATRIX::median(double* V,int npix, bool gabs)
{  
    double* W = (double *) malloc(sizeof(double)*npix);
    
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

void MATRIX::UpdateS(double* X, double* A, double* S, double* W, long KSuppS, bool L1,int nr, int nc, int npix)
{
    
    double* tempS = (double *) malloc(sizeof(double)*npix);
    double* OtempS = (double *) malloc(sizeof(double)*npix);
    double thrd;
    int niter_max = CG_NIter;
    
    // LS solution using conjugate-gradient
    
    L_CG(A,X,S,nr,nc,npix,niter_max);
    
    // Thresholding the sources
    
    for (int i=0;i<nc;i++)
    {
       // copying the sources divided by the weights
       for (int j=0;j<npix;j++) 
       {
           tempS[j] = S[i*npix+j]/W[i*npix + j]; 
           OtempS[j] = std::abs(tempS[j]);
       }
              
       // ordering the source
       quickSort(OtempS, 0, npix-1, true);
       thrd = OtempS[KSuppS];
       
       // thresholding
       Thresholding(tempS,thrd,npix, L1);
       
       // copying back the sources
       for (int j=0;j<npix;j++) S[i*npix+j] = tempS[j]*W[i*npix + j];
       
    }
    
    free(tempS);
    free(OtempS);
}

/****************************************************************************/
// Updating the mixing matrix
/****************************************************************************/

/**
* @param X : data
* @param A : mixing matrix
* @param S : sources
* @param W : weight matrix
*/

void MATRIX::UpdateA(double* X, double* A, double* S,int nr, int nc, int npix)
{
    
    double* tempA = (double *) malloc(sizeof(double)*nr);
    double L2n = 0;
    int niter_max = CG_NIter;
    
    // LS solution using conjugate-gradient
    
    R_CG(A,X,S,nr,nc,npix, niter_max);
    
    // L2 constraint
    
    for (int i=0;i<nc;i++)
    {
       // copying the sources divided by the weights
       for (int j=0;j<nr;j++) tempA[j] = A[j*nc+i]; 
       
       // computing the L2 norm
       L2n = TwoNorm(tempA,nr);
              
       // copying back the sources
       for (int j=0;j<nr;j++) A[j*nc+i] = tempA[j]/(sqrt(L2n) + 1e-12);
       
    }
    
    free(tempA);
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

void MATRIX::quickSort( double* a, int first, int last, bool descend) 
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

int MATRIX::pivot(double* a, int first, int last, bool descend) 
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
void MATRIX::swap(double& a, double& b)
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
void MATRIX::swapNoTemp(double& a, double& b)
{
    a -= b;
    b += a;// b gets the original value of a
    a = (b - a);// a gets the original value of b
}
