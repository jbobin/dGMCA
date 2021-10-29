/*
 * cln.h - This file is part of pyplanck
 * Created on 03/12/16
 * Contributor : 
 *
 * Copyright 2016 CEA
 *
 * DESCRIPTION
 *
 * This software is governed by the CeCILL  license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading,  using,  modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean  that it is complicated to manipulate,  and  that  also
 * therefore means  that it is reserved for developers  and  experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and,  more generally, to use and operate it in the
 * same conditions as regards security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
 *
 */

#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "omp.h"
#include "NumPyArrayData.h"

namespace bp = boost::python;
namespace np = boost::python::numpy;

class MATRIX
{
    
public:
    MATRIX(int n_Row,int n_Col, int Npix, int n_Block,int n_CG_NIter, int n_Iter);
    ~MATRIX();
    
    //  Main functions    
        
    void transpose(double* A, double* tA, int nr, int nc);
    void MM_multiply(double* A, double* B, double *AB, int nr, int nc, int ncB);
    void MV_multiply(double* A, double* b, double *Ab, int nr, int nc);
    void L_CG(double* A, double* X, double* S, int nr, int nc, int npix,int niter_max);
    void R_CG(double* A, double* X, double* S, int nr, int nc, int npix,int niter_max);
    double PowerMethod(double* A, int nr, int nc);
    void Thresholding(double* V, double Thrd,int npix, bool L1);
    double mad(double* V,int npix);
    double TwoNorm(double* V,int npix);
    double mean(double* V,int npix);
    double median(double* V,int npix, bool gabs);
    void quickSort( double* a, int first, int last,bool descend);
    int pivot(double* a, int first, int last,bool descend);
    void swap(double& a, double& b);
    void swapNoTemp(double& a, double& b);
    void Mcopy(double* A, double* cA, int nr, int nc);
    void Madd(double* A, double* B, double* ApB, int nr, int nc);
    void Msubs(double* A, double* B, double* AmB, int nr, int nc);
    void MHprod(double* A, double* B, double* ApB, int nr, int nc);
    void MHdiv(double* A, double* B, double* AdB, int nr, int nc);
    void UpdateS(double* X, double* A, double* S, double* W, long KSuppS, bool L1,int nr, int nc, int npix);
    void UpdateA(double* X, double* A, double* S, int nr, int nc, int npix);
    
    //**********************
    // Interface with python
    //**********************
    
    // QUIKSORT
    np::ndarray quicksort_numpy(np::ndarray &ar, bool descend){
        
        NumPyArrayData<double> V(ar);

        np::ndarray Out = np::zeros(bp::make_tuple(Npix), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        double* a = (double *) malloc(sizeof(double)*Npix);
        
        for (int i =0; i<Npix; i++)
        {
            a[i] = V(i);
        }
        
        quickSort(a,0,Npix-1,descend);
        
        for (int j =0; j<Npix; j++)
        {
            Out_data(j) = a[j];
        }

        return Out;
        
    }
    
    // BASIC GMCA ALGORITHM
    
    //np::ndarray gmca_numpy(np::ndarray &Xin, long KSuppS, bool L1){
    //
    //}
    
    // TEST ALLOC MATRIX
    
    np::ndarray mm_prod_numpy(np::ndarray &Ain,np::ndarray &Bin, long KSuppS, bool L1, int nr, int nc, int npix){
        
        NumPyArrayData<double> A(Ain);
        NumPyArrayData<double> B(Bin);
        
        np::ndarray Out = np::zeros(bp::make_tuple(nr,nc), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        double* pA = (double *) malloc(sizeof(double)*nr*nc);
        double* pB = (double *) malloc(sizeof(double)*nr*npix);
        double* pS = (double *) malloc(sizeof(double)*nc*npix);
        double* pW = (double *) malloc(sizeof(double)*nc*npix);
        
        //* feed the pointers   
             
        for (int i =0; i<nc; i++)
        {
            for (int j =0; j <npix; j++)
            {
                pS[i*npix + j] = A(i,j);
            }
        }
        
        for (int i =0; i<nr; i++)
        {
            for (int j =0; j <npix; j++)
            {
                pB[i*npix + j] = B(i,j);
            }
        }
        
        for (int i =0; i<nc*npix; i++) pW[i] = 1;
        
        //* matrix multiplication
                
        //UpdateS(pB, pA, pS, pW, KSuppS, L1, nr, nc, npix);
        
        UpdateA(pB, pA, pS,nr,nc,npix);
        
        //* get the data back
        
        for (int i = 0; i < nr; i++)
        {
            for (int j =0; j < nc; j++)
            {
                Out_data(i,j) = pA[i*nc + j];
            }
        }
        
        return Out;
    }
    
private:
    
    int NRow, NCol, NBlock, Npix;
    int CG_NIter;  //
    int NIter;  // Number of iterations

};

#endif // MATRIX_UTILS_H