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

#ifndef MATRIX_UTILS_OMP_H
#define MATRIX_UTILS_OMP_H

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "omp.h"
#include "NumPyArrayData.h"
#include <stdlib.h>

namespace bp = boost::python;
namespace np = boost::python::numpy;

class MATRIX_OMP
{

public:

    MATRIX_OMP(int n_Row,int n_Col,int n_Fixed, int Npix, int n_Block,int n_CG_NIter, int n_Iter, double Max_ts, double Min_ts,bool l1,int usep);
    ~MATRIX_OMP();

    //  Main functions

    void Mcopy(double* A, double* cA, int nr, int nc);
    void Madd(double* A, double* B, double* ApB, int nr, int nc);
    void Msubs(double* A, double* B, double* AmB, int nr, int nc);
    void MHprod(double* A, double* B, double* ApB, int nr, int nc);
    void MHdiv(double* A, double* B, double* AdB, int nr, int nc);
    void transpose(double* A, double* tA, int nr, int nc);
    void MM_multiply(double* A, double* B, double *AB, int nr, int nc, int nc2);
    void MV_multiply(double* A, double* b, double *Ab, int nr, int nc);
    void R_CG(double* A, double* X, double* S, int nr, int nc, int npix,int niter_max,  double* tS,double* SSt, double* R, double* D, double* B, double* Q, double* tempS);
    void L_CG(double* A, double* X, double* S, int nr, int nc, int npix, int niter_max, double* tA, double* AtA, double* R, double* D, double* B, double* Q, double* tempS);
    void L_CG2(double* A, double* X, double* S, int nr, int nc, int npix, int niter_max);
    void R_CG2(double* A, double* X, double* S, int nr, int nc, int npix, int niter_max);
    void Thresholding(double* V,double Thrd,int npix, bool L1);
    void quickSort( double* a, int first, int last, bool descend);
    void swap(double& a, double& b);
    void swapNoTemp(double& a, double& b);
    void NoQuickSort(double * array, int n);
    void NoQuickSort_abs(double * array, int n);
    void UpdatingWeights(double* S, double* t_S, double* W, double q, int npix, int nc);
    void R_CG_AMCA(double* A, double* X, double* S, double * Weights, int nr, int nc, int npix,int niter_max,  double* tS,double* SSt, double* R, double* D, double* B, double* Q, double* tempS);
    void UpdateResidual(double* X,double* A,double* S,double* Resi);
    void UpdateS(double* X, double* A, double* S, double Kmad, bool L1, int nr, int nc, int npix,double* t_S, double* t_s2, double* t_A, double* t_AtA, double* t_R, double* t_D, double* t_B, double* t_Q, double* t_s,int L0_max);
    void UpdateA(double* X, double* A, double* S,int nr, int nc, int npix, double* t_A, double* t_S, double* t_SSt, double* t_R, double* t_D, double* t_B, double* t_Q, double* t_S2);
    void UpdateS_AMCA(double* X, double* A, double* S,double* Weights,double alpha, double Kmad, bool L1, int nr, int nc, int npix,double* t_S, double* t_s2, double* t_A, double* t_AtA, double* t_R, double* t_D, double* t_B, double* t_Q, double* t_s,int L0_max);
    void UpdateA_AMCA(double* X, double* A, double* S,double* Weights,int nr, int nc, int npix, double* t_A, double* t_S, double* t_SSt, double* t_R, double* t_D, double* t_B, double* t_Q, double* t_S2);
    void GMCA_BASIC(double* X, double* A, double* S, double* OldA,int nr, int nc, int npix, int nmax, int n_iter,int maxts, int mints, bool L1,double* t_A,double* t_S,double* t_NcNc,double* t_Rs,double* t_Ds,double* t_Bs,double* t_Qs,double* t_Ra,double* t_Da,double* t_Ba,double* t_Qa,double* t_Za, double* t_s, double* t_s2);
    void AMCA_BASIC(double* X, double* A, double* S, double* OldA, double* Weights, double alpha,int nr, int nc, int npix, int nmax, int n_iter,int maxts, int mints, bool L1,double* t_A,double* t_S,double* t_NcNc,double* t_Rs,double* t_Ds,double* t_Bs,double* t_Qs,double* t_Ra,double* t_Da,double* t_Ba,double* t_Qa,double* t_Za, double* t_s, double* t_s2);
    double GradientDescent_S(double* X, double* A, double* S, double* iSigma,double alpha);
    double GradientDescent_A(double* X, double* A, double* S, double* iSigma,double alpha);
    void PALM_MAIN(double* X, double* A, double* S, double* iSigma);

    double PowerMethod2(double* A, int nr, int nc);
    double PowerMethod(double* A, int nr, int nc, double* x, double* tA, double* Ax, double* AtAx);
    double mad(double* V,int npix, double* W);
    double mad2(double* V,int npix, double* W);
    double TwoNorm(double* V,int npix);
    double mean(double* V,int npix);
    double median(double* V,int npix, bool gabs, double* W);
    double median2(double* V,int npix);
    double max2(double* V,int npix);
    void CorrectPERM(double* Aref, double* A,double* Aout);


    int pivot(double* a, int first, int last, bool descend);
    int GetNumElementsThrd(double * array, int n, double thrd);

    //********************************************************************************************************
    //******************************************* GMCA on batches ********************************************
    //********************************************************************************************************

    np::ndarray GMCA_Batches(np::ndarray &X,np::ndarray &Ain){

      NumPyArrayData<double> X_data(X);
      NumPyArrayData<double> Ain_data(Ain);

      np::ndarray A_out = np::zeros(bp::make_tuple(NRow,NCol,NBlock), np::dtype::get_builtin<double>());
      NumPyArrayData<double> A_out_data(A_out);

      // Defining outer-loop variables

      for (int batch=0;batch<NBlock;batch++)
      {
        // Allocating what needs to be allocated

        double * Xloop = (double *) malloc(sizeof(double)*NRow*Npix);
        double * Aloop = (double *) malloc(sizeof(double)*(NRow*NCol));
        double * Sloop = (double *) malloc(sizeof(double)*NCol*Npix);
        double * OldA = (double *) malloc(sizeof(double)*NRow*NCol);

        double * t_A = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_S = (double *) malloc(sizeof(double)*NCol*Npix);
        double * t_NcNc = (double *) malloc(sizeof(double)*NCol*NCol);

        double * t_s2 = (double *) malloc(sizeof(double)*Npix);
        double * t_s = (double *) malloc(sizeof(double)*Npix);

        double * t_Rs = (double *) malloc(sizeof(double)*NCol*Npix);
        double * t_Ds = (double *) malloc(sizeof(double)*NCol*Npix);
        double * t_Bs = (double *) malloc(sizeof(double)*NCol*Npix);
        double * t_Qs = (double *) malloc(sizeof(double)*NCol*Npix);

        double * t_Ra = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_Da = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_Ba = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_Qa = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_Za = (double *) malloc(sizeof(double)*NRow*NCol);

        for (int i=0;i<NRow;i++)
        {
          for (int j=0;j<Npix;j++) Xloop[j + i*Npix] = X_data(i,j,batch);
          for (int k=0;k<NCol;k++) Aloop[k + i*NCol] = Ain_data(i,k); // should be made only once
        }

        // Running GMCA for a single batch

        GMCA_BASIC(Xloop,Aloop,Sloop,OldA,NRow,NCol,Npix,CG_NIter,NIter,Max_ts,Min_ts,L1,t_A,t_S,t_NcNc,t_Rs,t_Ds,t_Bs,t_Qs,t_Ra,t_Da,t_Ba,t_Qa,t_Za,t_s,t_s2);

        for (int i=0;i<NRow;i++)
        {
          for (int k=0;k<NCol;k++)  A_out_data(i,k,batch) = Aloop[k + i*NCol] ;
        }

        // Freeing what needs to be freed

        free(Xloop);
        free(Aloop);
        free(Sloop);
        free(OldA);

        free(t_A);
        free(t_S);
        free(t_NcNc);

        free(t_s);
        free(t_s2);

        free(t_Rs);
        free(t_Ds);
        free(t_Bs);
        free(t_Qs);

        free(t_Ra);
        free(t_Da);
        free(t_Ba);
        free(t_Qa);
        free(t_Za);

      }

      return A_out;

    }

    //********************************************************************************************************
    //************************************** GMCA on batches - OMP - SINGLE STEP *****************************
    //********************************************************************************************************

    np::ndarray GMCA_OneIteration_Batches_omp(np::ndarray &X,np::ndarray &Ain,np::ndarray &Thresholds){

      NumPyArrayData<double> X_data(X);
      NumPyArrayData<double> Ain_data(Ain);
      NumPyArrayData<double> Thrd_data(Thresholds);

      np::ndarray A_out = np::zeros(bp::make_tuple(NRow+3,NCol,NBlock), np::dtype::get_builtin<double>());
      NumPyArrayData<double> A_out_data(A_out);

      // Defining outer-loop variables

      int batch;
      int nrow = NRow;
      int ncol = NCol;
      int npix = Npix;
      int cgniter = CG_NIter;
      int niter = NIter;
      int maxts = Max_ts;
      int mints = Min_ts;
      bool l1 = L1;
      int usep = UseP;

      #pragma omp parallel for shared(X_data, Ain_data,Thrd_data, A_out_data, batch,nrow,ncol,npix,cgniter,niter,maxts,mints,l1,usep)

      for (batch=0;batch<NBlock;batch++)
      {
        // Allocating what needs to be allocated

        double * Xloop = (double *) malloc(sizeof(double)*NRow*Npix);
        double * Aloop = (double *) malloc(sizeof(double)*(NRow*NCol));
        double * Aloop_in = (double *) malloc(sizeof(double)*(NRow*NCol));
        double * Aloop_out = (double *) malloc(sizeof(double)*(NRow*NCol));
        double * Sloop = (double *) malloc(sizeof(double)*NCol*Npix);

        for (int i=0;i<NRow;i++)
        {
          for (int j=0;j<Npix;j++) Xloop[j + i*Npix] = X_data(i,j,batch);
          for (int k=0;k<NCol;k++) {
            Aloop[k + i*NCol] = Ain_data(i,k); // should be made only once
            Aloop_in[k + i*NCol] = Ain_data(i,k);
          }
        }

        // Updating the sources

        L_CG2(Aloop, Xloop, Sloop, NRow, NCol, Npix, CG_NIter);

        // Thresholding

        double * vloop = (double *) malloc(sizeof(double)*Npix);
        double * wloop = (double *) malloc(sizeof(double)*Npix);
        double val_max = 0;

        for (int i=0;i<NCol;i++){
            for (int k=0;k<Npix;k++)  vloop[k] = Sloop[k + i*Npix];

            A_out_data(NRow,i,batch) = mad2(vloop,Npix, wloop); // Value of the mad

            val_max = std::abs(Sloop[i*Npix]);

            // Copying V

            for (int k=0;k<Npix;k++) if (std::abs(Sloop[k + i*Npix]) > val_max) val_max = std::abs(Sloop[k + i*Npix]);
            A_out_data(NRow+1,i,batch) = val_max; // get the max

           // thresholding

           for (int k=0;k<Npix;k++)
           {
             if (std::abs(Sloop[k + i*Npix]) < Thrd_data(i)) Sloop[k + i*Npix] = 0;

             if (std::abs(Sloop[k + i*Npix]) > Thrd_data(i))
             {
               if (L1 == true)
               {
                   if (Sloop[k + i*Npix] < 0)
                   {
                       Sloop[k + i*Npix] += Thrd_data(i);
                   }
                   if (Sloop[k + i*Npix] > 0)
                   {
                       Sloop[k + i*Npix]-= Thrd_data(i);
                   }
               }
             }
           }
        }

        // Checking which sources are non-negative

        int NNz = 0;
        int * NNz_loop = (int *) malloc(sizeof(int)*NCol);

        for (int i=0;i<NCol;i++){
            double L2s = 0;
            for (int k=0;k<Npix;k++){
              L2s += Sloop[k + i*Npix]*Sloop[k + i*Npix];
            }
            if (L2s > 1e-12){
              NNz_loop[NNz] = i;
              NNz += 1;
            }
          }

        if (NNz > 1){

          // Update the sources that need to be updated

          double * Sloop_small = (double *) malloc(sizeof(double)*NNz*Npix);
          double * Aloop_small = (double *) malloc(sizeof(double)*NNz*NRow);

          for (int k=0;k<NNz;k++){
            //for (int i=0;i<NRow;i++) Aloop_small[k + i*NNz] = Aloop[NNz_loop[k] + i*NNz];
            for (int i=0;i<Npix;i++) Sloop_small[i + k*Npix] = Sloop[i + NNz_loop[k]*Npix] ;
          }

          // Updating the mixing matrix

          R_CG2(Aloop_small, Xloop, Sloop_small, NRow, NNz, Npix, CG_NIter);

          // Projection onto the oblique ensemble

          for (int k=0;k<NNz;k++)
          {
            double L2n = 0;
            for (int i=0;i<NRow;i++) L2n += Aloop_small[k + i*NNz]*Aloop_small[k + i*NNz];
            for (int i=0;i<NRow;i++) Aloop[NNz_loop[k] + i*NCol]  = Aloop_small[k + i*NNz]/(std::sqrt(L2n) + 1e-12) ;
          }

          free(NNz_loop);
          free(Aloop_small);
          free(Sloop_small);

        }

        // If there's no source to be updated
        if (NNz < 2){
          for (int k=0;k<NCol;k++)
          {
            for (int i=0;i<NRow;i++) A_out_data(i,k,batch) = Aloop_in[k + i*NCol]; // Put back the initial mixing matrix
          }
        }

        // Correct for permutations

        CorrectPERM(Aloop_in, Aloop,Aloop_out);

        // Aggregation weights

        for (int k=0;k<NCol;k++){
          double L2s = 0;
          for (int i=0;i<Npix;i++) L2s += std::abs(Sloop[i + k*Npix]);
          A_out_data(NRow+2,k,batch) = L2s;
        }

        for (int i=0;i<NRow;i++) for (int k=0;k<NCol;k++) A_out_data(i,k,batch) = Aloop[k + i*NCol];

        // Freeing what needs to be freed

        free(Xloop);
        free(Aloop);
        free(Aloop_in);
        free(Aloop_out);
        free(Sloop);
        free(vloop);
        free(wloop);

      }

      return A_out;

    }

    np::ndarray GMCA_OneIteration_Batches_omp_TEMP(np::ndarray &X,np::ndarray &Ain,np::ndarray &Thresholds,np::ndarray &SigmaNoise){

      NumPyArrayData<double> X_data(X);
      NumPyArrayData<double> Ain_data(Ain);
      NumPyArrayData<double> Thrd_data(Thresholds);
      NumPyArrayData<double> SigmaNoise_data(SigmaNoise);

      np::ndarray A_out = np::zeros(bp::make_tuple(NRow+3,NCol,NBlock), np::dtype::get_builtin<double>());
      NumPyArrayData<double> A_out_data(A_out);

      // Defining outer-loop variables

      int batch;
      int nrow = NRow;
      int ncol = NCol;
      int npix = Npix;
      int cgniter = CG_NIter;
      int niter = NIter;
      int maxts = Max_ts;
      int mints = Min_ts;
      bool l1 = L1;
      int usep = UseP;

      #pragma omp parallel for shared(X_data, Ain_data,Thrd_data,SigmaNoise_data, A_out_data, batch,nrow,ncol,npix,cgniter,niter,maxts,mints,l1,usep)

      for (batch=0;batch<NBlock;batch++)
      {
        // Allocating what needs to be allocated

        double * Xloop = (double *) malloc(sizeof(double)*NRow*Npix);
        double * Aloop = (double *) malloc(sizeof(double)*(NRow*NCol));
        double * Aloop_in = (double *) malloc(sizeof(double)*(NRow*NCol));
        double * Aloop_out = (double *) malloc(sizeof(double)*(NRow*NCol));
        double * Sloop = (double *) malloc(sizeof(double)*NCol*Npix);
        double * Noiseloop = (double *) malloc(sizeof(double)*NRow*NRow);

        for (int i=0;i<NRow;i++)
        {
          for (int j=0;j<Npix;j++) Xloop[j + i*Npix] = X_data(i,j,batch);
          for (int k=0;k<NRow;k++) Noiseloop[k + i*NRow] = 0;
          for (int k=0;k<NCol;k++) {
            Aloop[k + i*NCol] = Ain_data(i,k); // should be made only once
            Aloop_in[k + i*NCol] = Ain_data(i,k);
          }
          Noiseloop[i + i*NRow] = SigmaNoise_data(i); // Defines the noise covariance matrix
        }

        // Updating the sources

        L_CG2(Aloop, Xloop, Sloop, NRow, NCol, Npix, CG_NIter);

        // Thresholding

        double * vloop = (double *) malloc(sizeof(double)*Npix);
        double * wloop = (double *) malloc(sizeof(double)*Npix);
        double val_max = 0;

        for (int i=0;i<NCol;i++){
            for (int k=0;k<Npix;k++)  vloop[k] = Sloop[k + i*Npix];

            A_out_data(NRow,i,batch) = mad2(vloop,Npix, wloop); // Value of the mad

            val_max = std::abs(Sloop[i*Npix]);

            // Copying V

            for (int k=0;k<Npix;k++) if (std::abs(Sloop[k + i*Npix]) > val_max) val_max = std::abs(Sloop[k + i*Npix]);
            A_out_data(NRow+1,i,batch) = val_max; // get the max

           // thresholding

           for (int k=0;k<Npix;k++)
           {
             if (std::abs(Sloop[k + i*Npix]) < Thrd_data(i)) Sloop[k + i*Npix] = 0;

             if (std::abs(Sloop[k + i*Npix]) > Thrd_data(i))
             {
               if (L1 == true)
               {
                   if (Sloop[k + i*Npix] < 0)
                   {
                       Sloop[k + i*Npix] += Thrd_data(i);
                   }
                   if (Sloop[k + i*Npix] > 0)
                   {
                       Sloop[k + i*Npix]-= Thrd_data(i);
                   }
               }
             }
           }
        }

        // Checking which sources are not zero

        int NNz = 0;
        int * NNz_loop = (int *) malloc(sizeof(int)*NCol);

        for (int i=0;i<NCol;i++){
            double L2s = 0;
            for (int k=0;k<Npix;k++){
              L2s += Sloop[k + i*Npix]*Sloop[k + i*Npix];
            }
            if (L2s > 1e-12){
              NNz_loop[NNz] = i;
              NNz += 1;
            }
          }

        if (NNz > 1){

          // Update the sources that need to be updated

          double * Sloop_small = (double *) malloc(sizeof(double)*NNz*Npix);
          double * Aloop_small = (double *) malloc(sizeof(double)*NNz*NRow);

          for (int k=0;k<NNz;k++){
            //for (int i=0;i<NRow;i++) Aloop_small[k + i*NNz] = Aloop[NNz_loop[k] + i*NNz];
            for (int i=0;i<Npix;i++) Sloop_small[i + k*Npix] = Sloop[i + NNz_loop[k]*Npix] ;
          }

          // Updating the mixing matrix

          R_CG2(Aloop_small, Xloop, Sloop_small, NRow, NNz, Npix, CG_NIter);

          // Projection onto the oblique ensemble

          for (int k=0;k<NNz;k++)
          {
            double L2n = 0;
            for (int i=0;i<NRow;i++) L2n += Aloop_small[k + i*NNz]*Aloop_small[k + i*NNz];
            for (int i=0;i<NRow;i++) Aloop[NNz_loop[k] + i*NCol]  = Aloop_small[k + i*NNz]/(std::sqrt(L2n) + 1e-12) ;
          }

          free(NNz_loop);
          free(Aloop_small);
          free(Sloop_small);

        }

        // If there's no source to be updated
        if (NNz < 2){
          for (int k=0;k<NCol;k++)
          {
            for (int i=0;i<NRow;i++) A_out_data(i,k,batch) = Aloop_in[k + i*NCol]; // Put back the initial mixing matrix
          }
        }

        // Correct for permutations

        CorrectPERM(Aloop_in, Aloop,Aloop_out);

        // Aggregation weights

        double * tAloop = (double *) malloc(sizeof(double)*(NRow*NCol));
        double * temp1 = (double *) malloc(sizeof(double)*(NRow*NCol));
        double * temp2 = (double *) malloc(sizeof(double)*(NCol*NCol));
        transpose(Aloop, tAloop, NRow, NCol); // Get the transpose of A

        // Put back the original mm

        for (int i=0;i<NRow;i++)for (int k=0;k<NCol;k++) Aloop[k + i*NCol] = Ain_data(i,k); // Not quite optimal ...

        L_CG2(Aloop, Noiseloop, temp1, NRow, NCol, NRow, CG_NIter); // Check that
        R_CG2(temp2, temp1, tAloop, NCol, NCol, NRow, CG_NIter); // Check that

        for (int k=0;k<NCol;k++){
          //double L2s = 0;
          //for (int i=0;i<Npix;i++) L2s += std::abs(Sloop[i + k*Npix]);
          A_out_data(NRow+2,k,batch) = temp2[k + k*NCol]; // Just take the diagonal elements
        }

        free(temp1);
        free(temp2);
        free(tAloop);



        for (int i=0;i<NRow;i++) for (int k=0;k<NCol;k++) A_out_data(i,k,batch) = Aloop[k + i*NCol];

        // Freeing what needs to be freed

        free(Xloop);
        free(Noiseloop);
        free(Aloop);
        free(Aloop_in);
        free(Aloop_out);
        free(Sloop);
        free(vloop);
        free(wloop);

      }

      return A_out;

    }

    //********************************************************************************************************
    //************************************** bGMCA on batches - OMP - SINGLE STEP ****************************
    //********************************************************************************************************

    np::ndarray bGMCA_OneIteration_RandBlock_Batches_omp(np::ndarray &X,np::ndarray &Ain,np::ndarray &Sin,np::ndarray &Thresholds,np::ndarray &BlockSize){

      NumPyArrayData<double> X_data(X);
      NumPyArrayData<double> Ain_data(Ain);
      NumPyArrayData<double> Sin_data(Sin);
      NumPyArrayData<double> Thrd_data(Thresholds);
      NumPyArrayData<double> BS_data(BlockSize);

      np::ndarray A_out = np::zeros(bp::make_tuple(NRow+3,NCol,NBlock), np::dtype::get_builtin<double>());
      NumPyArrayData<double> A_out_data(A_out);

      // Defining outer-loop variables

      int batch;
      int nrow = NRow;
      int ncol = NCol;
      int npix = Npix;
      int cgniter = CG_NIter;
      int niter = NIter;
      int maxts = Max_ts;
      int mints = Min_ts;
      bool l1 = L1;
      int usep = UseP;
      int BS = BS_data(0);

      #pragma omp parallel for shared(X_data, Ain_data,Sin_data,Thrd_data, A_out_data, batch,nrow,ncol,npix,cgniter,niter,maxts,mints,l1,usep,BS)

      for (batch=0;batch<NBlock;batch++)
      {
        // Allocating what needs to be allocated

        int cBS = NCol-BS;
        double * Xloop = (double *) malloc(sizeof(double)*NRow*Npix);
        double * ASloop = (double *) malloc(sizeof(double)*NRow*Npix);
        double * Aloop = (double *) malloc(sizeof(double)*(NRow*BS));
        double * Aloop_c = (double *) malloc(sizeof(double)*(NRow*cBS));
        double * Aloop_in = (double *) malloc(sizeof(double)*(NRow*NCol));
        double * Aloop_out = (double *) malloc(sizeof(double)*(NRow*NCol));
        double * Sloop = (double *) malloc(sizeof(double)*BS*Npix);
        double * Sloop_c = (double *) malloc(sizeof(double)*cBS*Npix);
        int * RandPerm = (int *) malloc(sizeof(int)*NCol);


        // RANDOM PERMUTATION

        //for (int i = 0; i < NCol; i++) RandPerm[i] = (int) RandInd_data(i);

        for (int i = 0; i < NCol; i++) RandPerm[i] = i;
        for (int i = 0; i < NCol; i++){
          int randIdx = rand() % NCol;
          int t = RandPerm[i];
          RandPerm[i] = RandPerm[randIdx];
          RandPerm[randIdx] = t;
        }

        //

        for (int j=0;j<Npix;j++){
          for (int i=0;i<NRow;i++) Xloop[j + i*Npix] = X_data(i,j,batch);
          for (int i=BS;i<NCol;i++) Sloop_c[j + (i-BS)*Npix] = Sin_data(RandPerm[i],j,batch);
        }

        for (int i=0;i<NRow;i++)
        {
          for (int k=0;k<NCol;k++)
          {
            Aloop_in[k + i*NCol] = Ain_data(i,k);
            Aloop_out[k + i*NCol] = Ain_data(i,k);
            if (k < BS) Aloop[k + i*BS] = Ain_data(i,RandPerm[k]);
            if (k > BS-1) Aloop_c[k-BS + i*cBS] = Ain_data(i,RandPerm[k]);
          }
        }

        // Compute the residual

        MM_multiply(Aloop_c,Sloop_c,ASloop,NRow,cBS,Npix);
        for (int j=0;j<Npix;j++) for (int i=0;i<NRow;i++) Xloop[j + i*Npix] = Xloop[j + i*Npix] - ASloop[j + i*Npix];

        // Update S

        L_CG2(Aloop, Xloop, Sloop, NRow, BS, Npix, CG_NIter);

        // Thresholding

        // Thresholding

        double * vloop = (double *) malloc(sizeof(double)*Npix);
        double * wloop = (double *) malloc(sizeof(double)*Npix);
        double val_max = 0;

        for (int i=0;i<BS;i++){

            for (int k=0;k<Npix;k++)  vloop[k] = Sloop[k + i*Npix];

            A_out_data(NRow,RandPerm[i],batch) = mad2(vloop,Npix, wloop); // Value of the mad

            val_max = std::abs(Sloop[i*Npix]);

            // Copying V

            for (int k=0;k<Npix;k++) if (std::abs(Sloop[k + i*Npix]) > val_max) val_max = std::abs(Sloop[k + i*Npix]);
            A_out_data(NRow+1,RandPerm[i],batch) = val_max; // get the max

           // thresholding

           double thrd = Thrd_data(RandPerm[i]);

           for (int k=0;k<Npix;k++)
           {
             if (std::abs(Sloop[k + i*Npix]) < thrd) Sloop[k + i*Npix] = 0;

             if (std::abs(Sloop[k + i*Npix]) > thrd)
             {
               if (L1 == true)
               {
                   if (Sloop[k + i*Npix] < 0)
                   {
                       Sloop[k + i*Npix] += thrd;
                   }
                   if (Sloop[k + i*Npix] > 0)
                   {
                       Sloop[k + i*Npix]-= thrd;
                   }
               }
             }
           }
        }

        for (int i=BS;i<NCol;i++){
            val_max = std::abs(Sloop[(i-BS)*Npix]);
            for (int k=0;k<Npix;k++){
              vloop[k] = Sloop[k + (i-BS)*Npix];
              if (std::abs(Sloop[k + (i-BS)*Npix]) > val_max) val_max = std::abs(Sloop[k + (i-BS)*Npix]);
            }
            A_out_data(NRow,RandPerm[i],batch) = mad2(vloop,Npix, wloop); // Value of the mad
            A_out_data(NRow+1,RandPerm[i],batch) = val_max; // get the max
          }

        free(vloop);
        free(wloop);

        // Update A

        // Checking which sources are non-negative

        int NNz = 0;
        int * NNz_loop = (int *) malloc(sizeof(int)*BS);

        for (int i=0;i<BS;i++){
            double L2s = 0;
            for (int k=0;k<Npix;k++){
              L2s += Sloop[k + i*Npix]*Sloop[k + i*Npix];
            }
            if (L2s > 1e-12){
              NNz_loop[NNz] = i;
              NNz += 1;
            }
          }

        if (NNz > 1){

          // Update the sources that need to be updated

          double * Sloop_small = (double *) malloc(sizeof(double)*NNz*Npix);
          double * Aloop_small = (double *) malloc(sizeof(double)*NNz*NRow);
          double * Aloop_big = (double *) malloc(sizeof(double)*NCol*NRow);

          for (int k=0;k<NNz;k++){
            for (int i=0;i<NRow;i++) Aloop_small[k + i*NNz] = Aloop[NNz_loop[k] + i*NNz];
            for (int i=0;i<Npix;i++) Sloop_small[i + k*Npix] = Sloop[i + NNz_loop[k]*Npix] ;
          }

          // Updating the mixing matrix

          R_CG2(Aloop_small, Xloop, Sloop_small, NRow, NNz, Npix, CG_NIter);

          // Projection onto the oblique ensemble

          for (int k=0;k<NCol*NRow;k++) Aloop_big[k] = Aloop_in[k];

          for (int k=0;k<NNz;k++)
          {
            double L2n = 0;
            for (int i=0;i<NRow;i++) L2n += Aloop_small[k + i*NNz]*Aloop_small[k + i*NNz];
            for (int i=0;i<NRow;i++) Aloop_big[RandPerm[NNz_loop[k]] + i*NCol]  = Aloop_small[k + i*NNz]/(std::sqrt(L2n) + 1e-12) ;
          }


          free(Aloop_small);
          free(Sloop_small);

          // Correct for permutations

          CorrectPERM(Aloop_in, Aloop_big,Aloop_out);

          free(Aloop_big);

          // Aggregation weights

          for (int k=0;k<BS;k++){
            double L2s = 0;
            for (int i=0;i<Npix;i++) L2s += std::abs(Sloop[i + k*Npix]);
            A_out_data(NRow+2,RandPerm[k],batch) = L2s*L2s;
          }

          for (int i=0;i<NRow;i++) for (int k=0;k<NCol;k++) A_out_data(i,k,batch) = Aloop_out[k + i*NCol];
        }

        // If there's no source to be updated
        if (NNz < 2){
          for (int k=0;k<NCol;k++)
          {
            for (int i=0;i<NRow;i++) A_out_data(i,k,batch) = Aloop_in[k + i*NCol]; // Put back the initial mixing matrix
          }
        }

        // Freeing what needs to be freed

        free(Xloop);
        free(Aloop);
        free(Aloop_in);
        free(Aloop_out);
        free(RandPerm);
        free(ASloop);
        free(Sloop);
        free(Sloop_c);
        free(Aloop_c);

      }

      return A_out;

    }

    //********************************************************************************************************
    //************************************** bGMCA on batches - OMP - SINGLE STEP ****************************
    //********************************************************************************************************

    np::ndarray bGMCA_OneIteration_Batches_omp(np::ndarray &X,np::ndarray &Ain,np::ndarray &Sin,np::ndarray &Thresholds,np::ndarray &BlockSize,np::ndarray &RandInd){

      NumPyArrayData<double> X_data(X);
      NumPyArrayData<double> Ain_data(Ain);
      NumPyArrayData<double> Sin_data(Sin);
      NumPyArrayData<double> Thrd_data(Thresholds);
      NumPyArrayData<double> BS_data(BlockSize);
      NumPyArrayData<double> RandInd_data(RandInd);

      np::ndarray A_out = np::zeros(bp::make_tuple(NRow+3,NCol,NBlock), np::dtype::get_builtin<double>());
      NumPyArrayData<double> A_out_data(A_out);

      // Defining outer-loop variables

      int batch;
      int nrow = NRow;
      int ncol = NCol;
      int npix = Npix;
      int cgniter = CG_NIter;
      int niter = NIter;
      int maxts = Max_ts;
      int mints = Min_ts;
      bool l1 = L1;
      int usep = UseP;
      int BS = BS_data(0);

      #pragma omp parallel for shared(X_data,RandInd_data, Ain_data,Sin_data,Thrd_data, A_out_data, batch,nrow,ncol,npix,cgniter,niter,maxts,mints,l1,usep,BS)

      for (batch=0;batch<NBlock;batch++)
      {
        // Allocating what needs to be allocated

        int cBS = NCol-BS;
        double * Xloop = (double *) malloc(sizeof(double)*NRow*Npix);
        double * ASloop = (double *) malloc(sizeof(double)*NRow*Npix);
        double * Aloop = (double *) malloc(sizeof(double)*(NRow*BS));
        double * Aloop_c = (double *) malloc(sizeof(double)*(NRow*cBS));
        double * Aloop_in = (double *) malloc(sizeof(double)*(NRow*NCol));
        double * Aloop_out = (double *) malloc(sizeof(double)*(NRow*NCol));
        double * Sloop = (double *) malloc(sizeof(double)*BS*Npix);
        double * Sloop_c = (double *) malloc(sizeof(double)*cBS*Npix);
        int * RandPerm = (int *) malloc(sizeof(int)*NCol);


        // RANDOM PERMUTATION

        for (int i = 0; i < NCol; i++) RandPerm[i] = (int) RandInd_data(i);

        //

        for (int j=0;j<Npix;j++){
          for (int i=0;i<NRow;i++) Xloop[j + i*Npix] = X_data(i,j,batch);
          for (int i=BS;i<NCol;i++) Sloop_c[j + (i-BS)*Npix] = Sin_data(RandPerm[i],j,batch);
        }

        for (int i=0;i<NRow;i++)
        {
          for (int k=0;k<NCol;k++)
          {
            Aloop_in[k + i*NCol] = Ain_data(i,k);
            Aloop_out[k + i*NCol] = Ain_data(i,k);
            if (k < BS) Aloop[k + i*BS] = Ain_data(i,RandPerm[k]);
            if (k > BS-1) Aloop_c[k-BS + i*cBS] = Ain_data(i,RandPerm[k]);
          }
        }

        // Compute the residual

        MM_multiply(Aloop_c,Sloop_c,ASloop,NRow,cBS,Npix);
        for (int j=0;j<Npix;j++) for (int i=0;i<NRow;i++) Xloop[j + i*Npix] = Xloop[j + i*Npix] - ASloop[j + i*Npix];

        // Update S

        L_CG2(Aloop, Xloop, Sloop, NRow, BS, Npix, CG_NIter);

        // Thresholding

        // Thresholding

        double * vloop = (double *) malloc(sizeof(double)*Npix);
        double * wloop = (double *) malloc(sizeof(double)*Npix);
        double val_max = 0;

        for (int i=0;i<BS;i++){

            for (int k=0;k<Npix;k++)  vloop[k] = Sloop[k + i*Npix];

            A_out_data(NRow,RandPerm[i],batch) = mad2(vloop,Npix, wloop); // Value of the mad

            val_max = std::abs(Sloop[i*Npix]);

            // Copying V

            for (int k=0;k<Npix;k++) if (std::abs(Sloop[k + i*Npix]) > val_max) val_max = std::abs(Sloop[k + i*Npix]);
            A_out_data(NRow+1,RandPerm[i],batch) = val_max; // get the max

           // thresholding

           double thrd = Thrd_data(RandPerm[i]);

           for (int k=0;k<Npix;k++)
           {
             if (std::abs(Sloop[k + i*Npix]) < thrd) Sloop[k + i*Npix] = 0;

             if (std::abs(Sloop[k + i*Npix]) > thrd)
             {
               if (L1 == true)
               {
                   if (Sloop[k + i*Npix] < 0)
                   {
                       Sloop[k + i*Npix] += thrd;
                   }
                   if (Sloop[k + i*Npix] > 0)
                   {
                       Sloop[k + i*Npix]-= thrd;
                   }
               }
             }
           }
        }

        for (int i=BS;i<NCol;i++){
            val_max = std::abs(Sloop[(i-BS)*Npix]);
            for (int k=0;k<Npix;k++){
              vloop[k] = Sloop[k + (i-BS)*Npix];
              if (std::abs(Sloop[k + (i-BS)*Npix]) > val_max) val_max = std::abs(Sloop[k + (i-BS)*Npix]);
            }
            A_out_data(NRow,RandPerm[i],batch) = mad2(vloop,Npix, wloop); // Value of the mad
            A_out_data(NRow+1,RandPerm[i],batch) = val_max; // get the max
          }

        free(vloop);
        free(wloop);

        // Update A

        // Checking which sources are non-negative

        int NNz = 0;
        int * NNz_loop = (int *) malloc(sizeof(int)*BS);

        for (int i=0;i<BS;i++){
            double L2s = 0;
            for (int k=0;k<Npix;k++){
              L2s += Sloop[k + i*Npix]*Sloop[k + i*Npix];
            }
            if (L2s > 1e-12){
              NNz_loop[NNz] = i;
              NNz += 1;
            }
          }

        if (NNz > 1){

          // Update the sources that need to be updated

          double * Sloop_small = (double *) malloc(sizeof(double)*NNz*Npix);
          double * Aloop_small = (double *) malloc(sizeof(double)*NNz*NRow);
          double * Aloop_big = (double *) malloc(sizeof(double)*NCol*NRow);

          for (int k=0;k<NNz;k++){
            for (int i=0;i<NRow;i++) Aloop_small[k + i*NNz] = Aloop[NNz_loop[k] + i*NNz];
            for (int i=0;i<Npix;i++) Sloop_small[i + k*Npix] = Sloop[i + NNz_loop[k]*Npix] ;
          }

          // Updating the mixing matrix

          R_CG2(Aloop_small, Xloop, Sloop_small, NRow, NNz, Npix, CG_NIter);

          // Projection onto the oblique ensemble

          for (int k=0;k<NCol*NRow;k++) Aloop_big[k] = Aloop_in[k];

          for (int k=0;k<NNz;k++)
          {
            double L2n = 0;
            for (int i=0;i<NRow;i++) L2n += Aloop_small[k + i*NNz]*Aloop_small[k + i*NNz];
            for (int i=0;i<NRow;i++) Aloop_big[RandPerm[NNz_loop[k]] + i*NCol]  = Aloop_small[k + i*NNz]/(std::sqrt(L2n) + 1e-12) ;
          }


          free(Aloop_small);
          free(Sloop_small);

          // Correct for permutations

          CorrectPERM(Aloop_in, Aloop_big,Aloop_out);

          free(Aloop_big);

          // Aggregation weights

          for (int k=0;k<BS;k++){
            double L2s = 0;
            for (int i=0;i<Npix;i++) L2s += std::abs(Sloop[i + k*Npix]);
            A_out_data(NRow+2,RandPerm[k],batch) = L2s*L2s;
          }

          for (int i=0;i<NRow;i++) for (int k=0;k<NCol;k++) A_out_data(i,k,batch) = Aloop_out[k + i*NCol];
        }

        // If there's no source to be updated
        if (NNz < 2){
          for (int k=0;k<NCol;k++)
          {
            for (int i=0;i<NRow;i++) A_out_data(i,k,batch) = Aloop_in[k + i*NCol]; // Put back the initial mixing matrix
          }
        }

        // Freeing what needs to be freed

        free(Xloop);
        free(Aloop);
        free(Aloop_in);
        free(Aloop_out);
        free(RandPerm);
        free(ASloop);
        free(Sloop);
        free(Sloop_c);
        free(Aloop_c);

      }

      return A_out;

    }

    //********************************************************************************************************
    //************************************** bbGMCA - Compute Residual ****************************
    //********************************************************************************************************

    np::ndarray bGMCA_Residual_Batches_omp(np::ndarray &X,np::ndarray &Ain,np::ndarray &Sin,np::ndarray &BlockSize,np::ndarray &RandInd){

      NumPyArrayData<double> X_data(X);
      NumPyArrayData<double> Ain_data(Ain);
      NumPyArrayData<double> Sin_data(Sin);
      NumPyArrayData<double> BS_data(BlockSize);
      NumPyArrayData<double> RandInd_data(RandInd);

      np::ndarray A_out = np::zeros(bp::make_tuple(NRow,Npix,NBlock), np::dtype::get_builtin<double>());
      NumPyArrayData<double> A_out_data(A_out);

      // Defining outer-loop variables

      int batch;
      int nrow = NRow;
      int ncol = NCol;
      int npix = Npix;
      int cgniter = CG_NIter;
      int niter = NIter;
      int maxts = Max_ts;
      int mints = Min_ts;
      bool l1 = L1;
      int usep = UseP;


      #pragma omp parallel for shared(X_data,RandInd_data, Ain_data,Sin_data, A_out_data, batch,nrow,ncol,npix,cgniter,niter,maxts,mints,l1,usep,BS_data)

      for (batch=0;batch<NBlock;batch++)
      {
        // Allocating what needs to be allocated

        int BS = (int) BS_data(0);
        int cBS = NCol - BS;
        double * Xloop = (double *) malloc(sizeof(double)*NRow*Npix);
        double * ASloop = (double *) malloc(sizeof(double)*NRow*Npix);
        double * Aloop = (double *) malloc(sizeof(double)*(NRow*BS));
        double * Aloop_c = (double *) malloc(sizeof(double)*(NRow*cBS));
        double * Sloop_c = (double *) malloc(sizeof(double)*cBS*Npix);
        int * RandPerm = (int *) malloc(sizeof(int)*NCol);


        // RANDOM PERMUTATION

        for (int i = 0; i < NCol; i++) RandPerm[i] = (int) RandInd_data(i);

        //

        for (int j=0;j<Npix;j++){
          for (int i=0;i<NRow;i++) Xloop[j + i*Npix] = X_data(i,j,batch);
          for (int i=BS;i<NCol;i++) Sloop_c[j + (i-BS)*Npix] = Sin_data(RandPerm[i],j,batch);
        }

        for (int i=0;i<NRow;i++)
        {
          for (int k=0;k<NCol;k++)
          {
            if (k < BS) Aloop[k + i*BS] = Ain_data(i,RandPerm[k]);
            if (k > BS-1) Aloop_c[k-BS + i*cBS] = Ain_data(i,RandPerm[k]);
          }
        }

        // Compute the residual

        MM_multiply(Aloop_c,Sloop_c,ASloop,NRow,cBS,Npix);
        for (int j=0;j<Npix;j++) for (int i=0;i<NRow;i++) A_out_data(i,j,batch) = Xloop[j + i*Npix] - ASloop[j + i*Npix];

        // Freeing what needs to be freed

        free(Xloop);
        free(Aloop);
        free(RandPerm);
        free(ASloop);
        free(Sloop_c);
        free(Aloop_c);

      }

      return A_out;

    }

    //********************************************************************************************************
    //************************************** bbGMCA - Compute Residual ****************************
    //********************************************************************************************************

    np::ndarray bGMCA_UpdateS_Batches_omp(np::ndarray &X,np::ndarray &Ain,np::ndarray &Sin,np::ndarray &BlockSize,np::ndarray &RandInd){

      NumPyArrayData<double> X_data(X);
      NumPyArrayData<double> Ain_data(Ain);
      NumPyArrayData<double> Sin_data(Sin);
      NumPyArrayData<double> BS_data(BlockSize);
      NumPyArrayData<double> RandInd_data(RandInd);

      np::ndarray A_out = np::zeros(bp::make_tuple(NRow,Npix,NBlock), np::dtype::get_builtin<double>());
      NumPyArrayData<double> A_out_data(A_out);

      // Defining outer-loop variables

      int batch;
      int nrow = NRow;
      int ncol = NCol;
      int npix = Npix;
      int cgniter = CG_NIter;
      int niter = NIter;
      int maxts = Max_ts;
      int mints = Min_ts;
      bool l1 = L1;
      int usep = UseP;


      #pragma omp parallel for shared(X_data,RandInd_data, Ain_data,Sin_data, A_out_data, batch,nrow,ncol,npix,cgniter,niter,maxts,mints,l1,usep,BS_data)

      for (batch=0;batch<NBlock;batch++)
      {
        // Allocating what needs to be allocated

        int BS = (int) BS_data(0);
        int cBS = NCol - BS;
        double * Xloop = (double *) malloc(sizeof(double)*NRow*Npix);
        double * ASloop = (double *) malloc(sizeof(double)*NRow*Npix);
        double * Aloop = (double *) malloc(sizeof(double)*(NRow*BS));
        double * Aloop_c = (double *) malloc(sizeof(double)*(NRow*cBS));
        double * Sloop_c = (double *) malloc(sizeof(double)*cBS*Npix);
        double * Sloop = (double *) malloc(sizeof(double)*BS*Npix);
        int * RandPerm = (int *) malloc(sizeof(int)*NCol);


        // RANDOM PERMUTATION

        for (int i = 0; i < NCol; i++) RandPerm[i] = (int) RandInd_data(i);

        //

        for (int j=0;j<Npix;j++){
          for (int i=0;i<NRow;i++) Xloop[j + i*Npix] = X_data(i,j,batch);
          for (int i=BS;i<NCol;i++) Sloop_c[j + (i-BS)*Npix] = Sin_data(RandPerm[i],j,batch);
        }

        for (int i=0;i<NRow;i++)
        {
          for (int k=0;k<NCol;k++)
          {
            if (k < BS) Aloop[k + i*BS] = Ain_data(i,RandPerm[k]);
            if (k > BS-1) Aloop_c[k-BS + i*cBS] = Ain_data(i,RandPerm[k]);
          }
        }

        // Compute the residual

        MM_multiply(Aloop_c,Sloop_c,ASloop,NRow,cBS,Npix);
        for (int j=0;j<Npix;j++) for (int i=0;i<NRow;i++) Xloop[j + i*Npix] = Xloop[j + i*Npix] - ASloop[j + i*Npix];

        // Update S

        L_CG2(Aloop, Xloop, Sloop, NRow, BS, Npix, CG_NIter);

        for (int j=0;j<Npix;j++) for (int i=0;i<BS;i++) A_out_data(RandPerm[i],j,batch) = Sloop[j + i*Npix];

        // Freeing what needs to be freed

        free(Xloop);
        free(Aloop);
        free(RandPerm);
        free(ASloop);
        free(Sloop);
        free(Sloop_c);
        free(Aloop_c);

      }

      return A_out;

    }

    //********************************************************************************************************
    //************************************** bbGMCA - Compute Residual ****************************
    //********************************************************************************************************

    np::ndarray bGMCA_UpdateAS_Batches_omp(np::ndarray &X,np::ndarray &Ain,np::ndarray &Sin,np::ndarray &ThrdIn,np::ndarray &BlockSize,np::ndarray &RandInd){

      NumPyArrayData<double> X_data(X);
      NumPyArrayData<double> Ain_data(Ain);
      NumPyArrayData<double> Sin_data(Sin);
      NumPyArrayData<double> BS_data(BlockSize);
      NumPyArrayData<double> RandInd_data(RandInd);
      NumPyArrayData<double> Thrd_data(ThrdIn);

      np::ndarray A_out = np::zeros(bp::make_tuple(NRow+3,NCol,NBlock), np::dtype::get_builtin<double>());
      NumPyArrayData<double> A_out_data(A_out);

      // Defining outer-loop variables

      int batch;
      int nrow = NRow;
      int ncol = NCol;
      int npix = Npix;
      int cgniter = CG_NIter;
      int niter = NIter;
      int maxts = Max_ts;
      int mints = Min_ts;
      bool l1 = L1;
      int usep = UseP;


      #pragma omp parallel for shared(X_data,Thrd_data,RandInd_data, Ain_data,Sin_data, A_out_data, batch,nrow,ncol,npix,cgniter,niter,maxts,mints,l1,usep,BS_data)

      for (batch=0;batch<NBlock;batch++)
      {
        // Allocating what needs to be allocated

        int BS = (int) BS_data(0);
        int cBS = NCol - BS;
        double * Xloop = (double *) malloc(sizeof(double)*NRow*Npix);
        double * ASloop = (double *) malloc(sizeof(double)*NRow*Npix);
        double * Aloop = (double *) malloc(sizeof(double)*(NRow*BS));
        double * Aloop_c = (double *) malloc(sizeof(double)*(NRow*cBS));
        double * Sloop_c = (double *) malloc(sizeof(double)*cBS*Npix);
        double * Sloop = (double *) malloc(sizeof(double)*BS*Npix);
        double * Aloop_in = (double *) malloc(sizeof(double)*NCol*NRow);
        double * Aloop_out = (double *) malloc(sizeof(double)*NCol*NRow);
        int * RandPerm = (int *) malloc(sizeof(int)*NCol);


        // RANDOM PERMUTATION

        for (int i = 0; i < NCol; i++) RandPerm[i] = (int) RandInd_data(i);

        //

        for (int j=0;j<Npix;j++){
          for (int i=0;i<NRow;i++) Xloop[j + i*Npix] = X_data(i,j,batch);
          for (int i=BS;i<NCol;i++) Sloop_c[j + (i-BS)*Npix] = Sin_data(RandPerm[i],j,batch);
        }

        for (int i=0;i<NRow;i++)
        {
          for (int k=0;k<NCol;k++)
          {
            Aloop_in[k + i*NCol] = Ain_data(i,k);
            Aloop_out[k + i*NCol] = Ain_data(i,k);
            if (k < BS) Aloop[k + i*BS] = Ain_data(i,RandPerm[k]);
            if (k > BS-1) Aloop_c[k-BS + i*cBS] = Ain_data(i,RandPerm[k]);
          }
        }

        // Compute the residual

        MM_multiply(Aloop_c,Sloop_c,ASloop,NRow,cBS,Npix);
        for (int j=0;j<Npix;j++) for (int i=0;i<NRow;i++) Xloop[j + i*Npix] = Xloop[j + i*Npix] - ASloop[j + i*Npix];

        // Update S

        L_CG2(Aloop, Xloop, Sloop, NRow, BS, Npix, CG_NIter);

        // Thresholding

        // Thresholding

        double * vloop = (double *) malloc(sizeof(double)*Npix);
        double * wloop = (double *) malloc(sizeof(double)*Npix);
        double val_max = 0;

        for (int i=0;i<BS;i++){

            for (int k=0;k<Npix;k++)  vloop[k] = Sloop[k + i*Npix];

            A_out_data(NRow,RandPerm[i],batch) = mad2(vloop,Npix, wloop); // Value of the mad

            val_max = std::abs(Sloop[i*Npix]);

            // Copying V

            for (int k=0;k<Npix;k++) if (std::abs(Sloop[k + i*Npix]) > val_max) val_max = std::abs(Sloop[k + i*Npix]);
            A_out_data(NRow+1,RandPerm[i],batch) = val_max; // get the max

           // thresholding

           double thrd = Thrd_data(RandPerm[i]);

           for (int k=0;k<Npix;k++)
           {
             if (std::abs(Sloop[k + i*Npix]) < thrd) Sloop[k + i*Npix] = 0;

             if (std::abs(Sloop[k + i*Npix]) > thrd)
             {
               if (L1 == true)
               {
                   if (Sloop[k + i*Npix] < 0)
                   {
                       Sloop[k + i*Npix] += thrd;
                   }
                   if (Sloop[k + i*Npix] > 0)
                   {
                       Sloop[k + i*Npix]-= thrd;
                   }
               }
             }
           }
        }

        free(vloop);
        free(wloop);

        // Update A

        // Checking which sources are non-negative

        int NNz = 0;
        int * NNz_loop = (int *) malloc(sizeof(int)*BS);

        for (int i=0;i<BS;i++){
            double L2s = 0;
            for (int k=0;k<Npix;k++){
              L2s += Sloop[k + i*Npix]*Sloop[k + i*Npix];
            }
            if (L2s > 1e-12){
              NNz_loop[NNz] = i;
              NNz += 1;
            }
          }

        if (NNz > 1){

          // Update the sources that need to be updated

          double * Sloop_small = (double *) malloc(sizeof(double)*NNz*Npix);
          double * Aloop_small = (double *) malloc(sizeof(double)*NNz*NRow);
          double * Aloop_big = (double *) malloc(sizeof(double)*NCol*NRow);

          for (int k=0;k<NNz;k++){
            for (int i=0;i<NRow;i++) Aloop_small[k + i*NNz] = Aloop[NNz_loop[k] + i*NNz];
            for (int i=0;i<Npix;i++) Sloop_small[i + k*Npix] = Sloop[i + NNz_loop[k]*Npix] ;
          }

          // Updating the mixing matrix

          R_CG2(Aloop_small, Xloop, Sloop_small, NRow, NNz, Npix, CG_NIter);

          // Projection onto the oblique ensemble

          for (int k=0;k<NCol*NRow;k++) Aloop_big[k] = Aloop_in[k];

          for (int k=0;k<NNz;k++)
          {
            double L2n = 0;
            for (int i=0;i<NRow;i++) L2n += Aloop_small[k + i*NNz]*Aloop_small[k + i*NNz];
            for (int i=0;i<NRow;i++) Aloop_big[RandPerm[NNz_loop[k]] + i*NCol]  = Aloop_small[k + i*NNz]/(std::sqrt(L2n) + 1e-12) ;
          }


          free(Aloop_small);
          free(Sloop_small);

          // Correct for permutations

          CorrectPERM(Aloop_in, Aloop_big,Aloop_out);

          free(Aloop_big);

          // Aggregation weights

          for (int k=0;k<BS;k++){
            double L2s = 0;
            for (int i=0;i<Npix;i++) L2s += std::abs(Sloop[i + k*Npix]);
            A_out_data(NRow+2,RandPerm[k],batch) = L2s*L2s;
          }

          for (int i=0;i<NRow;i++) for (int k=0;k<NCol;k++) A_out_data(i,k,batch) = Aloop_out[k + i*NCol];
        }

        // If there's no source to be updated
        if (NNz < 2){
          for (int k=0;k<NCol;k++)
          {
            for (int i=0;i<NRow;i++) A_out_data(i,k,batch) = Aloop_in[k + i*NCol]; // Put back the initial mixing matrix
          }
        }

        // Freeing what needs to be freed

        free(Xloop);
        free(Aloop);
        free(Aloop_in);
        free(Aloop_out);
        free(RandPerm);
        free(ASloop);
        free(Sloop);
        free(Sloop_c);
        free(Aloop_c);

      }

      return A_out;

    }



    //********************************************************************************************************
    //************************************** bGMCA on batches - OMP - SINGLE STEP ****************************
    //********************************************************************************************************

    np::ndarray GMCA_GetS_Batches_omp(np::ndarray &X,np::ndarray &Ain,np::ndarray &Thresholds){

      NumPyArrayData<double> X_data(X);
      NumPyArrayData<double> Ain_data(Ain);
      NumPyArrayData<double> Thrd_data(Thresholds);

      np::ndarray S_out = np::zeros(bp::make_tuple(NCol,Npix,NBlock), np::dtype::get_builtin<double>());
      NumPyArrayData<double> S_out_data(S_out);

      // Defining outer-loop variables

      int batch;
      int nrow = NRow;
      int ncol = NCol;
      int npix = Npix;
      int cgniter = CG_NIter;
      int niter = NIter;
      int maxts = Max_ts;
      int mints = Min_ts;
      bool l1 = L1;
      int usep = UseP;

      #pragma omp parallel for shared(X_data, Ain_data,Thrd_data, S_out_data, batch,nrow,ncol,npix,cgniter,niter,maxts,mints,l1,usep)

      for (batch=0;batch<NBlock;batch++)
      {
        // Allocating what needs to be allocated

        double * Xloop = (double *) malloc(sizeof(double)*NRow*Npix);
        double * Aloop = (double *) malloc(sizeof(double)*(NRow*NCol));
        double * Sloop = (double *) malloc(sizeof(double)*NCol*Npix);

        //

        for (int j=0;j<Npix;j++) for (int i=0;i<NRow;i++) Xloop[j + i*Npix] = X_data(i,j,batch);
        for (int i=0;i<NRow;i++) for (int k=0;k<NCol;k++) Aloop[k + i*NCol] = Ain_data(i,k); // should be made only once

        // Updating the sources

        L_CG2(Aloop, Xloop, Sloop, NRow, NCol, Npix, CG_NIter);

        // Thresholding

        double * vloop = (double *) malloc(sizeof(double)*Npix);

        for (int i=0;i<NCol;i++){

           // thresholding

           double thrd = Thrd_data(i);

           for (int k=0;k<Npix;k++)
           {
             if (std::abs(Sloop[k + i*Npix]) < thrd) Sloop[k + i*Npix] = 0;

             if (std::abs(Sloop[k + i*Npix]) > thrd)
             {
               if (L1 == true)
               {
                   if (Sloop[k + i*Npix] < 0)
                   {
                       Sloop[k + i*Npix] += thrd;
                   }
                   if (Sloop[k + i*Npix] > 0)
                   {
                       Sloop[k + i*Npix]-= thrd;
                   }
               }
             }
           }
        }

        for (int j=0;j<Npix;j++) for (int i=0;i<NCol;i++) S_out_data(i,j,batch) = Sloop[j + i*Npix];



        // Freeing what needs to be freed

        free(Xloop);
        free(Aloop);
        free(Sloop);
      }

      return S_out;

    }

    //********************************************************************************************************
    //************************************** GMCA on batches - OMP *******************************************
    //********************************************************************************************************

    np::ndarray GMCA_Batches_omp(np::ndarray &X,np::ndarray &Ain){

      NumPyArrayData<double> X_data(X);
      NumPyArrayData<double> Ain_data(Ain);

      np::ndarray A_out = np::zeros(bp::make_tuple(NRow,NCol,NBlock), np::dtype::get_builtin<double>());
      NumPyArrayData<double> A_out_data(A_out);

      // Defining outer-loop variables

      int batch;
      int nrow = NRow;
      int ncol = NCol;
      int npix = Npix;
      int cgniter = CG_NIter;
      int niter = NIter;
      int maxts = Max_ts;
      int mints = Min_ts;
      bool l1 = L1;
      int usep = UseP;

      #pragma omp parallel for shared(X_data, Ain_data, A_out_data, batch,nrow,ncol,npix,cgniter,niter,maxts,mints,l1,usep)

      for (batch=0;batch<NBlock;batch++)
      {
        // Allocating what needs to be allocated

        double * Xloop = (double *) malloc(sizeof(double)*NRow*Npix);
        double * Aloop = (double *) malloc(sizeof(double)*(NRow*NCol));
        double * Sloop = (double *) malloc(sizeof(double)*NCol*Npix);
        double * OldA = (double *) malloc(sizeof(double)*NRow*NCol);

        double * t_A = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_S = (double *) malloc(sizeof(double)*NCol*Npix);
        double * t_NcNc = (double *) malloc(sizeof(double)*NCol*NCol);

        double * t_s2 = (double *) malloc(sizeof(double)*Npix);
        double * t_s = (double *) malloc(sizeof(double)*Npix);

        double * t_Rs = (double *) malloc(sizeof(double)*NCol*Npix);
        double * t_Ds = (double *) malloc(sizeof(double)*NCol*Npix);
        double * t_Bs = (double *) malloc(sizeof(double)*NCol*Npix);
        double * t_Qs = (double *) malloc(sizeof(double)*NCol*Npix);

        double * t_Ra = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_Da = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_Ba = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_Qa = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_Za = (double *) malloc(sizeof(double)*NRow*NCol);

        for (int i=0;i<NRow;i++)
        {
          for (int j=0;j<Npix;j++) Xloop[j + i*Npix] = X_data(i,j,batch);
          for (int k=0;k<NCol;k++) Aloop[k + i*NCol] = Ain_data(i,k); // should be made only once
        }

        // Running GMCA for a single batch

        GMCA_BASIC(Xloop,Aloop,Sloop,OldA,nrow,ncol,npix,cgniter,niter,maxts,mints,l1,t_A,t_S,t_NcNc,t_Rs,t_Ds,t_Bs,t_Qs,t_Ra,t_Da,t_Ba,t_Qa,t_Za,t_s,t_s2);

        for (int i=0;i<NRow;i++)
        {
          for (int k=0;k<NCol;k++)  A_out_data(i,k,batch) = Aloop[k + i*NCol] ;
        }

        // Freeing what needs to be freed

        free(Xloop);
        free(Aloop);
        free(Sloop);
        free(OldA);

        free(t_A);
        free(t_S);
        free(t_NcNc);

        free(t_s);
        free(t_s2);

        free(t_Rs);
        free(t_Ds);
        free(t_Bs);
        free(t_Qs);

        free(t_Ra);
        free(t_Da);
        free(t_Ba);
        free(t_Qa);
        free(t_Za);

      }

      return A_out;

    }

    //********************************************************************************************************
    //************************************** Update S on batches - OMP ***************************************
    //********************************************************************************************************

    np::ndarray UpdateS_Batches_omp(np::ndarray &X,np::ndarray &Ain){

      NumPyArrayData<double> X_data(X);
      NumPyArrayData<double> Ain_data(Ain);

      np::ndarray S_out = np::zeros(bp::make_tuple(NCol,Npix,NBlock), np::dtype::get_builtin<double>());
      NumPyArrayData<double> S_out_data(S_out);

      // Defining outer-loop variables

      int batch;
      //int nrow = NRow;
      //int ncol = NCol;
      //int npix = Npix;
      //int cgniter = CG_NIter;
      //int niter = NIter;
      //int maxts = Max_ts;
      //int mints = Min_ts;
      //bool l1 = L1;
      //int usep = UseP;

      #pragma omp parallel for shared(X_data, Ain_data, S_out_data, batch)

      for (batch=0;batch<NBlock;batch++)
      {
        // Allocating what needs to be allocated

        double * Xloop = (double *) malloc(sizeof(double)*NRow*Npix);
        double * Aloop = (double *) malloc(sizeof(double)*NRow*NCol);
        double * Sloop = (double *) malloc(sizeof(double)*NCol*Npix);

        for (int i=0;i<NRow;i++)
        {
          for (int j=0;j<Npix;j++) Xloop[j + i*Npix] = X_data(i,j,batch);
          for (int k=0;k<NCol;k++) Aloop[k + i*NCol] = Ain_data(i,k); // should be made only once
        }

        // Running LCG for a single batch

        L_CG2(Aloop, Xloop, Sloop, NRow, NCol, Npix, CG_NIter);


        for (int i=0;i<NCol;i++)
        {
          for (int k=0;k<Npix;k++)  S_out_data(i,k,batch) = Sloop[k + i*Npix] ;
        }

        // Freeing what needs to be freed

        free(Xloop);
        free(Aloop);
        free(Sloop);

      }

      return S_out;

    }

    //********************************************************************************************************
    //************************************** Update A on batches - OMP ***************************************
    //********************************************************************************************************

    np::ndarray UpdateA_Batches_omp(np::ndarray &X,np::ndarray &Sin){

      NumPyArrayData<double> X_data(X);
      NumPyArrayData<double> Sin_data(Sin);

      np::ndarray A_out = np::zeros(bp::make_tuple(NRow,NCol,NBlock), np::dtype::get_builtin<double>());
      NumPyArrayData<double> A_out_data(A_out);

      // Defining outer-loop variables

      int batch;

      #pragma omp parallel for shared(X_data, Sin_data, A_out_data, batch)

      for (batch=0;batch<NBlock;batch++)
      {
        // Allocating what needs to be allocated

        double * Xloop = (double *) malloc(sizeof(double)*NRow*Npix);
        double * Aloop = (double *) malloc(sizeof(double)*NRow*NCol);
        double * Sloop = (double *) malloc(sizeof(double)*NCol*Npix);

        for (int j=0;j<Npix;j++)
        {
          for (int i=0;i<NRow;i++) Xloop[j + i*Npix] = X_data(i,j,batch);
          for (int k=0;k<NCol;k++) Sloop[j + k*Npix] = Sin_data(k,j,batch); // should be made only once
        }

        // Running LCG for a single batch

        R_CG2(Aloop, Xloop, Sloop, NRow, NCol, Npix, CG_NIter);


        for (int i=0;i<NRow;i++)
        {
          for (int k=0;k<NCol;k++)  A_out_data(i,k,batch) = Aloop[k + i*NCol] ;
        }

        // Freeing what needs to be freed

        free(Xloop);
        free(Aloop);
        free(Sloop);

      }

      return A_out;

    }

    //********************************************************************************************************
    //************************************* Correct permutations on batches - OMP ****************************
    //********************************************************************************************************

    np::ndarray CorrectPerm_Batches_omp(np::ndarray &Xref,np::ndarray &X){

      NumPyArrayData<double> X_data(X);
      NumPyArrayData<double> Xref_data(Xref);

      np::ndarray A_out = np::zeros(bp::make_tuple(NRow,NCol,NBlock), np::dtype::get_builtin<double>());
      NumPyArrayData<double> A_out_data(A_out);

      // Defining outer-loop variables

      int batch;

      #pragma omp parallel for shared(X_data, Xref_data, A_out_data, batch)

      for (batch=0;batch<NBlock;batch++)
      {
        // Allocating what needs to be allocated

        double * Xloop = (double *) malloc(sizeof(double)*NRow*NCol);
        double * Aloop = (double *) malloc(sizeof(double)*NRow*NCol);
        double * Xrefloop = (double *) malloc(sizeof(double)*NRow*NCol);

        for (int j=0;j<NCol;j++)
        {
          for (int i=0;i<NRow;i++){
            Xloop[j + i*NCol] = X_data(i,j,batch);
            Xrefloop[j + i*NCol] = Xref_data(i,j);
          }
        }

        // Running LCG for a single batch

        CorrectPERM(Xrefloop, Xloop,Aloop);


        for (int i=0;i<NRow;i++)
        {
          for (int k=0;k<NCol;k++)  A_out_data(i,k,batch) = Aloop[k + i*NCol] ;
        }

        // Freeing what needs to be freed

        free(Xloop);
        free(Aloop);
        free(Xrefloop);

      }

      return A_out;

    }

    //********************************************************************************************************
    //************************************** AMCA on batches - OMP *******************************************
    //********************************************************************************************************

    np::ndarray AMCA_Batches_omp(np::ndarray &X,np::ndarray &Ain, double alpha){

      NumPyArrayData<double> X_data(X);
      NumPyArrayData<double> Ain_data(Ain);

      np::ndarray A_out = np::zeros(bp::make_tuple(NRow,NCol,NBlock), np::dtype::get_builtin<double>());
      NumPyArrayData<double> A_out_data(A_out);

      // Defining outer-loop variables

      int batch;
      int nrow = NRow;
      int ncol = NCol;
      int npix = Npix;
      int cgniter = CG_NIter;
      int niter = NIter;
      int maxts = Max_ts;
      int mints = Min_ts;
      bool l1 = L1;
      int usep = UseP;

      #pragma omp parallel for shared(X_data, Ain_data, A_out_data, batch,nrow,ncol,npix,cgniter,niter,maxts,mints,l1,usep,alpha)

      for (batch=0;batch<NBlock;batch++)
      {
        // Allocating what needs to be allocated

        double * Xloop = (double *) malloc(sizeof(double)*NRow*Npix);
        double * Aloop = (double *) malloc(sizeof(double)*(NRow*NCol));
        double * Sloop = (double *) malloc(sizeof(double)*NCol*Npix);
        double * OldA = (double *) malloc(sizeof(double)*NRow*NCol);

        double * t_A = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_S = (double *) malloc(sizeof(double)*NCol*Npix);
        double * t_NcNc = (double *) malloc(sizeof(double)*NCol*NCol);
        double * Weights = (double *) malloc(sizeof(double)*Npix);

        double * t_s2 = (double *) malloc(sizeof(double)*Npix);
        double * t_s = (double *) malloc(sizeof(double)*Npix);

        double * t_Rs = (double *) malloc(sizeof(double)*NCol*Npix);
        double * t_Ds = (double *) malloc(sizeof(double)*NCol*Npix);
        double * t_Bs = (double *) malloc(sizeof(double)*NCol*Npix);
        double * t_Qs = (double *) malloc(sizeof(double)*NCol*Npix);

        double * t_Ra = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_Da = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_Ba = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_Qa = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_Za = (double *) malloc(sizeof(double)*NRow*NCol);

        for (int i=0;i<NRow;i++)
        {
          for (int j=0;j<Npix;j++) Xloop[j + i*Npix] = X_data(i,j,batch);
          for (int k=0;k<NCol;k++) Aloop[k + i*NCol] = Ain_data(i,k); // should be made only once
        }

        // Running GMCA for a single batch

        AMCA_BASIC(Xloop, Aloop, Sloop, OldA, Weights,alpha, NRow, NCol, Npix, CG_NIter, NIter,Max_ts, Min_ts, L1, t_A, t_S, t_NcNc,t_Rs,t_Ds,t_Bs,t_Qs,t_Ra,t_Da,t_Ba,t_Qa,t_Za,t_s, t_s2);

        for (int i=0;i<NRow;i++)
        {
          for (int k=0;k<NCol;k++)  A_out_data(i,k,batch) = Aloop[k + i*NCol] ;
        }

        // Freeing what needs to be freed

        free(Xloop);
        free(Aloop);
        free(Sloop);
        free(OldA);

        free(t_A);
        free(t_S);
        free(t_NcNc);

        free(t_s);
        free(t_s2);

        free(t_Rs);
        free(t_Ds);
        free(t_Bs);
        free(t_Qs);

        free(t_Ra);
        free(t_Da);
        free(t_Ba);
        free(t_Qa);
        free(t_Za);

      }

      return A_out;

    }

    //********************************************************************************************************
    //******************************************* PALM IMPLEMENTATION ****************************************
    //********************************************************************************************************

    np::ndarray PALM_Basic_OMPs(np::ndarray &X,np::ndarray &Ain,np::ndarray &Sin){  // Parallelization over the sources

      NumPyArrayData<double> X_data(X);
      NumPyArrayData<double> Ain_data(Ain);
      NumPyArrayData<double> S_data(Sin);

      int i,j;
      int Nbin = 10;
      int Bpix = Npix/Nbin;

      np::ndarray Out = np::zeros(bp::make_tuple(NRow,Npix), np::dtype::get_builtin<double>());
      NumPyArrayData<double> Out_data(Out);

      //for (i=0;i<;i++) // MAIN LOOP
      //{

        #pragma omp parallel for shared(X_data, Ain_data, S_data,Out_data,Bpix,Nbin)
        for (j=0;j<Nbin;j++) // Loop over the sources
        {

          double * A = (double *) malloc(sizeof(double)*NRow*NCol);
          double * X = (double *) malloc(sizeof(double)*NRow*Npix);
          double * S = (double *) malloc(sizeof(double)*NCol*Npix);
          double * Resi = (double *) malloc(sizeof(double)*NRow*Npix);

          int xstart = j*Bpix;
          int xend = xstart + Bpix;

          for (int k=xstart;k<xend;k++)
          {
            for (int l=0;l<NRow;l++) X[k+l*Npix] = X_data(l,k); // We should use batches/blocks otherwise
            for (int l=0;l<NCol;l++) S[k+l*Npix] = S_data(l,k); // We should use batches/blocks otherwise
          }

          for (int k=0;k<NCol;k++)
          {
            for (int l=0;l<NRow;l++) A[k+l*NCol] = Ain_data(l,k); // We should use batches/blocks otherwise
          }

          UpdateResidual(X,A,S,Resi);

          for (int k=xstart;k<xend;k++)
          {
            for (int l=0;l<NRow;l++) Out_data(l,k) = Resi[k+l*Npix]; // We should use batches/blocks otherwise
          }

          free(X);
          free(A);
          free(S);
          free(Resi);
        }


      //} // MAIN LOOP

      return Out;

    }

    np::ndarray PALM_Basic_Batches(np::ndarray &X,np::ndarray &Ain,np::ndarray &Sin,np::ndarray &NormalizedCovMat){  // Parallelization over the sources

      NumPyArrayData<double> X_data(X);
      NumPyArrayData<double> Ain_data(Ain);
      NumPyArrayData<double> S_data(Sin);
      NumPyArrayData<double> NCVM_data(NormalizedCovMat);

      np::ndarray Out = np::zeros(bp::make_tuple(NRow,NCol,NBlock), np::dtype::get_builtin<double>());
      NumPyArrayData<double> Out_data(Out);

      // Defining outer-loop variables

      int batch;

      #pragma omp parallel for shared(X_data, Ain_data,S_data,NCVM_data,Out_data, batch)
      for (batch=0;batch<NBlock;batch++)
        {

          double * A = (double *) malloc(sizeof(double)*NRow*NCol);
          double * X = (double *) malloc(sizeof(double)*NRow*Npix);
          double * S = (double *) malloc(sizeof(double)*NCol*Npix);
          double * iSigma = (double *) malloc(sizeof(double)*NRow);

          for (int k=0;k<Npix;k++)
          {
            for (int l=0;l<NRow;l++) X[k+l*Npix] = X_data(l,k,batch); // We should use batches/blocks otherwise
            for (int l=0;l<NCol;l++) S[k+l*Npix] = S_data(l,k,batch); // We should use batches/blocks otherwise
          }

          for (int k=0;k<NCol;k++)
          {
            for (int l=0;l<NRow;l++) A[k+l*NCol] = Ain_data(l,k,batch); // We should use batches/blocks otherwise
          }

          for (int k=0;k<NRow;k++) iSigma[k] = NCVM_data(k,batch); // Just the diagonal elements

          PALM_MAIN(X,A,S,iSigma);

          for (int k=0;k<NCol;k++)
          {
            for (int l=0;l<NRow;l++) for (int k=0;k<NCol;k++) Out_data(l,k,batch) = A[k+l*NCol]; // We should use batches/blocks otherwise
          }

          free(X);
          free(A);
          free(S);
        }


      //} // MAIN LOOP

      return Out;

    }

    np::ndarray PALM_Basic_Basic(np::ndarray &X,np::ndarray &Ain,np::ndarray &Sin,np::ndarray &NormalizedCovMat){  // Parallelization over the sources

      NumPyArrayData<double> X_data(X);
      NumPyArrayData<double> Ain_data(Ain);
      NumPyArrayData<double> S_data(Sin);
      NumPyArrayData<double> NCVM_data(NormalizedCovMat);

      np::ndarray Out = np::zeros(bp::make_tuple(NRow,NCol,NBlock), np::dtype::get_builtin<double>());
      NumPyArrayData<double> Out_data(Out);

      // Defining outer-loop variables

      int batch = 0;

      for (batch=0;batch<NBlock;batch++)
        {

          double * A = (double *) malloc(sizeof(double)*NRow*NCol);
          double * X = (double *) malloc(sizeof(double)*NRow*Npix);
          double * S = (double *) malloc(sizeof(double)*NCol*Npix);
          double * iSigma = (double *) malloc(sizeof(double)*NRow);

          for (int k=0;k<Npix;k++)
          {
            for (int l=0;l<NRow;l++) X[k+l*Npix] = X_data(l,k,batch); // We should use batches/blocks otherwise
            for (int l=0;l<NCol;l++) S[k+l*Npix] = S_data(l,k,batch); // We should use batches/blocks otherwise
          }

          for (int k=0;k<NCol;k++)
          {
            for (int l=0;l<NRow;l++) A[k+l*NCol] = Ain_data(l,k); // We should use batches/blocks otherwise
          }

          for (int k=0;k<NRow;k++) iSigma[k] = NCVM_data(k,batch); // Just the diagonal elements

          PALM_MAIN(X,A,S,iSigma);

          for (int k=0;k<NCol;k++)
          {
            for (int l=0;l<NRow;l++) for (int k=0;k<NCol;k++) Out_data(l,k,batch) = A[k+l*NCol]; // We should use batches/blocks otherwise
          }

          free(X);
          free(A);
          free(S);
        }


      //} // MAIN LOOP

      return Out;

    }

    np::ndarray PowerMethod_Batches(np::ndarray &Ain,np::ndarray &Sin){  // Parallelization over the sources

      NumPyArrayData<double> Ain_data(Ain);
      NumPyArrayData<double> S_data(Sin);

      np::ndarray Out = np::zeros(bp::make_tuple(NBlock), np::dtype::get_builtin<double>());
      NumPyArrayData<double> Out_data(Out);

      // Defining outer-loop variables

      int batch;

      #pragma omp parallel for shared(Ain_data,S_data,Out_data, batch)
      for (batch=0;batch<NBlock;batch++)
        {

          double * A = (double *) malloc(sizeof(double)*NRow*NCol);
          double * tS = (double *) malloc(sizeof(double)*NCol*Npix);

          for (int k=0;k<Npix;k++)
          {
            for (int l=0;l<NCol;l++) tS[l+k*NCol] = S_data(l,k,batch); // We should use batches/blocks otherwise
          }

          for (int k=0;k<NCol;k++)
          {
            for (int l=0;l<NRow;l++) A[k+l*NCol] = Ain_data(l,k,batch); // We should use batches/blocks otherwise
          }

          double L = PowerMethod2(tS, Npix, NCol);

          Out_data(batch) = L; // We should use batches/blocks otherwise

          free(A);
          free(tS);
        }


      //} // MAIN LOOP

      return Out;

    }

    //********************************************************************************************************
    //******************************************* TEST FUNCTIONS  ********************************************
    //********************************************************************************************************

    np::ndarray GMCA_Basic(np::ndarray &X,np::ndarray &Ain,np::ndarray &Sin){

      NumPyArrayData<double> X_data(X);
      NumPyArrayData<double> Ain_data(Ain);
      NumPyArrayData<double> S_data(Sin);
      int npix = Npix;
      int nrow = NRow;
      int ncol = NCol;
      int i,j,k;

      //np::ndarray Out = np::zeros(bp::make_tuple(nrow,ncol), np::dtype::get_builtin<double>());
      np::ndarray Out = np::zeros(bp::make_tuple(nrow,ncol), np::dtype::get_builtin<double>());
      NumPyArrayData<double> Out_data(Out);

      // Defining outer-loop variables

        // Allocating what needs to be allocated

        double * Xloop = (double *) malloc(sizeof(double)*nrow*npix);
        double * Aloop = (double *) malloc(sizeof(double)*(nrow*ncol));
        double * Sloop = (double *) malloc(sizeof(double)*NCol*Npix);
        double * OldA = (double *) malloc(sizeof(double)*NRow*NCol);

        double * t_A = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_S = (double *) malloc(sizeof(double)*NCol*Npix);
        double * t_S2 = (double *) malloc(sizeof(double)*NCol*Npix);
        double * t_NcNc = (double *) malloc(sizeof(double)*NCol*NCol);

        double * t_s2 = (double *) malloc(sizeof(double)*Npix);
        double * t_s = (double *) malloc(sizeof(double)*Npix);

        double * t_Rs = (double *) malloc(sizeof(double)*NCol*Npix);
        double * t_Ds = (double *) malloc(sizeof(double)*NCol*Npix);
        double * t_Bs = (double *) malloc(sizeof(double)*NCol*Npix);
        double * t_Qs = (double *) malloc(sizeof(double)*NCol*Npix);

        double * t_Ra = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_Da = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_Ba = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_Qa = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_Za = (double *) malloc(sizeof(double)*NRow*NCol);

        for (int i=0;i<NRow;i++)
        {
          for (int j=0;j<Npix;j++) Xloop[j + i*Npix] = X_data(i,j);
          for (int k=0;k<NCol;k++) Aloop[k + i*NCol] = Ain_data(i,k);
        }

        for (int i=0;i<NCol;i++)
        {
          for (int j=0;j<Npix;j++) Sloop[j + i*Npix] = S_data(i,j);
        }

        GMCA_BASIC(Xloop, Aloop, Sloop, OldA, NRow, NCol, Npix, CG_NIter, NIter,Max_ts, Min_ts, L1, t_A, t_S, t_NcNc,t_Rs,t_Ds,t_Bs,t_Qs,t_Ra,t_Da,t_Ba,t_Qa,t_Za,t_s, t_s2);

        for (int i=0;i<NRow;i++)
        {
          for (int k=0;k<NCol;k++)  Out_data(i,k) = Aloop[k + i*NCol];
        }

        free(Xloop);
        free(Aloop);
        free(Sloop);
        free(OldA);

        free(t_A);
        free(t_S);
        free(t_S2);
        free(t_NcNc);

        free(t_s);
        free(t_s2);

        free(t_Rs);
        free(t_Ds);
        free(t_Bs);
        free(t_Qs);

        free(t_Ra);
        free(t_Da);
        free(t_Ba);
        free(t_Qa);
        free(t_Za);

      return Out;
    }


    //********************************************************************************************************
    //******************************************* TEST FUNCTIONS  ********************************************
    //********************************************************************************************************

    np::ndarray AMCA_Basic(np::ndarray &X,np::ndarray &Ain,double alpha){

      NumPyArrayData<double> X_data(X);
      NumPyArrayData<double> Ain_data(Ain);
      int npix = Npix;
      int nrow = NRow;
      int ncol = NCol;
      int i,j,k;
      bool L1 = false;

      //np::ndarray Out = np::zeros(bp::make_tuple(nrow,ncol), np::dtype::get_builtin<double>());
      np::ndarray Out = np::zeros(bp::make_tuple(nrow,ncol), np::dtype::get_builtin<double>());
      NumPyArrayData<double> Out_data(Out);

      // Defining outer-loop variables

        // Allocating what needs to be allocated

        double * Xloop = (double *) malloc(sizeof(double)*nrow*npix);
        double * Aloop = (double *) malloc(sizeof(double)*(nrow*ncol));
        double * Sloop = (double *) malloc(sizeof(double)*NCol*Npix);
        double * OldA = (double *) malloc(sizeof(double)*NRow*NCol);

        double * t_A = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_S = (double *) malloc(sizeof(double)*NCol*Npix);
        double * t_S2 = (double *) malloc(sizeof(double)*NCol*Npix);
        double * t_NcNc = (double *) malloc(sizeof(double)*NCol*NCol);

        double * t_s2 = (double *) malloc(sizeof(double)*Npix);
        double * t_s = (double *) malloc(sizeof(double)*Npix);
        double * Weights = (double *) malloc(sizeof(double)*Npix);

        double * t_Rs = (double *) malloc(sizeof(double)*NCol*Npix);
        double * t_Ds = (double *) malloc(sizeof(double)*NCol*Npix);
        double * t_Bs = (double *) malloc(sizeof(double)*NCol*Npix);
        double * t_Qs = (double *) malloc(sizeof(double)*NCol*Npix);

        double * t_Ra = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_Da = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_Ba = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_Qa = (double *) malloc(sizeof(double)*NRow*NCol);
        double * t_Za = (double *) malloc(sizeof(double)*NRow*NCol);

        for (int i=0;i<NRow;i++)
        {
          for (int j=0;j<Npix;j++) Xloop[j + i*Npix] = X_data(i,j);
          for (int k=0;k<NCol;k++) Aloop[k + i*NCol] = Ain_data(i,k);
        }

        AMCA_BASIC(Xloop, Aloop, Sloop, OldA, Weights,alpha, NRow, NCol, Npix, CG_NIter, NIter,Max_ts, Min_ts, L1, t_A, t_S, t_NcNc,t_Rs,t_Ds,t_Bs,t_Qs,t_Ra,t_Da,t_Ba,t_Qa,t_Za,t_s, t_s2);

        for (int i=0;i<NRow;i++)
        {
          for (int k=0;k<NCol;k++)  Out_data(i,k) = Aloop[k + i*NCol];
        }

        free(Xloop);
        free(Aloop);
        free(Sloop);
        free(OldA);

        free(t_A);
        free(t_S);
        free(t_S2);
        free(t_NcNc);

        free(t_s);
        free(t_s2);
        free(Weights);

        free(t_Rs);
        free(t_Ds);
        free(t_Bs);
        free(t_Qs);

        free(t_Ra);
        free(t_Da);
        free(t_Ba);
        free(t_Qa);
        free(t_Za);

      return Out;
    }

private:

    int NRow, NCol, NBlock, Npix, NFixed;
    int CG_NIter;  //
    int NIter;  // Number of iterations
    double Max_ts,Min_ts;
    bool L1;
    int UseP;

};

#endif // MATRIX_UTILS_OMP_H
