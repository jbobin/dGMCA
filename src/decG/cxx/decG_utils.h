/*
 * This software is a computer program whose purpose is to apply mutli-
 * resolution signal processing algorithms on spherical 3D data.
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

#ifndef DECG_UTILS_H
#define DECG_UTILS_H

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "omp.h"
#include "NumPyArrayData.h"

namespace bp = boost::python;
namespace np = boost::python::numpy;

class PLK
{

public:
    PLK();
    ~PLK();

       double PowerMethod(double* AtA,double* AtAx, double* x, int nr, int nc);
       void transpose(double* A, double* tA, int nr, int nc);
       void MM_multiply(double* A, double* B, double *AB, int nr, int nc, int ncB);
       void MV_multiply(double* A, double* b, double *Ab, int nr, int nc);
       void Basic_CG(double* Z, double* X, double* S, int nc, int niter_max);
       void Mcopy(double* A, double* cA, int nr, int nc);
       void R_CG_cpx(double* A, double* Xr, double* Xi, double* Sr, double* Si, int nr, int nc, int npix,int niter_max);
       void R_CG2(double* A, double* X, double* S, int nr, int nc, int npix,int niter_max);

       //####################################################
       // CONJUGATE GRADIENT
       //####################################################

       np::ndarray Left_CG_numpy(np::ndarray X_In,np::ndarray M,np::ndarray mixmat,np::ndarray epsilon,np::ndarray niter,np::ndarray npar,np::ndarray nfreq){

        NumPyArrayData<double> X_data(X_In);
        NumPyArrayData<double> B_data(M);
        NumPyArrayData<double> Epsi(epsilon);
        NumPyArrayData<double> Npar(npar);
        NumPyArrayData<double> n_iter(niter);
        NumPyArrayData<double> Nfreq(nfreq);
        NumPyArrayData<double> MixMat(mixmat);


        double eps = Epsi(0);
        long NSources = Npar(0);  // Number of sources
        long NFreq = Nfreq(0);  // Number of observations
        long Niter = n_iter(0);

        np::ndarray Out = np::zeros(bp::make_tuple(NSources), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);

        double *pt_Ht = (double *) malloc(sizeof(double)*NFreq*NSources);  // This is the mixing matrix
        double *pt_D = (double *) malloc(sizeof(double)*NFreq); // This is the datum
        double *pt_Pout = (double *) malloc(sizeof(double)*NSources);  // Output sources
        double *pt_P = (double *) malloc(sizeof(double)*NSources); // These are the sources
        double Lip = 0;  // Lipschitz ct

        // Multiplying by Ht

        for (int y=0; y < NFreq; y++) {
            pt_D[y] = X_data(y);   // Store the data
            for (int z=0;z<NSources;z++){
                pt_Ht[z*NFreq+y] = B_data(y)*MixMat(y,z);  // Already the transpose of A and apply the beam
            }
        }

        // Apply the matrix to the data

        MV_multiply(pt_Ht, pt_D, pt_P, NSources, NFreq);

        // Computing HtH, its norm and add epsilon*L*eye

        double *pt_HtH = (double *) malloc(sizeof(double)*NSources*NSources);
        double *pt_H = (double *) malloc(sizeof(double)*NSources* NFreq);
        double *pt_HtHx = (double *) malloc(sizeof(double)*NSources);
        double *pt_x = (double *) malloc(sizeof(double)*NSources);

        transpose(pt_Ht, pt_H, NSources, NFreq);
        MM_multiply(pt_Ht, pt_H, pt_HtH, NSources, NFreq,NSources);

        Lip = PowerMethod(pt_HtH,pt_HtHx,pt_x,NSources,NSources);

        for (int z=0;z<NSources;z++){
               pt_HtH[z*NSources+z] += eps*(Lip + 1e-24);  // Adding epsilon*Lip, we should rather use the max
        }

        // Apply the inverse applied to pt_D using CG

        Basic_CG(pt_HtH, pt_P, pt_Pout, NSources, Niter);

        //

        for (int y=0; y < NSources; y++) Out_data(y) = pt_Pout[y];

        free(pt_D);
        free(pt_Pout);
        free(pt_H);
        free(pt_Ht);
        free(pt_HtH);
        free(pt_HtHx);
        free(pt_x);
        free(pt_P);

        return Out;

        }


       //####################################################
       // APPLY (Ht H + epsilon*L*eye(3))-1 Ht
       //####################################################

       np::ndarray applyHt_PInv_numpy(np::ndarray X_In,np::ndarray M,np::ndarray mixmat,np::ndarray epsilon,np::ndarray npar,np::ndarray nfreq,np::ndarray nx){

        NumPyArrayData<double> X_data(X_In);
        NumPyArrayData<double> B_data(M);
        NumPyArrayData<double> Epsi(epsilon);
        NumPyArrayData<double> Npar(npar);
        NumPyArrayData<double> n_x(nx);
        NumPyArrayData<double> Nfreq(nfreq);
        NumPyArrayData<double> MixMat(mixmat);


        double eps = Epsi(0);
        long NSources = Npar(0);  // Number of sources
        long NFreq = Nfreq(0);  // Number of observations
        long Nx = n_x(0);  // Number of elements in k-space

        np::ndarray Out = np::zeros(bp::make_tuple(NSources,Nx), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);

        //

        int x=0;

        #pragma omp parallel for shared(x, X_data, B_data, Out_data, MixMat,eps, NSources,NFreq)
        for (x=0; x < Nx; x++) {

            // Check B_data first

            double val = 0;
            for (int y=0; y < NFreq; y++) {
                val += B_data(y,x);
            }

            if (val > 0){ // If there's enough observations

                double *pt_Ht = (double *) malloc(sizeof(double)*NFreq*NSources);  // This is the mixing matrix
                double *pt_D = (double *) malloc(sizeof(double)*NFreq); // This is the datum
                double *pt_Pout = (double *) malloc(sizeof(double)*NSources);  // Output sources
                double *pt_P = (double *) malloc(sizeof(double)*NSources); // These are the sources
                double Lip = 0;  // Lipschitz ct

                // Multiplying by Ht

                for (int y=0; y < NFreq; y++) {
                    pt_D[y] = X_data(y,x);   // Store the data
                    for (int z=0;z<NSources;z++){
                        pt_Ht[z*NFreq+y] = B_data(y,x)*MixMat(y,z);  // Already the transpose of A and apply the beam
                    }
                }

                // Apply the matrix to the data

                MV_multiply(pt_Ht, pt_D, pt_P, NSources, NFreq);

                // Computing HtH, its norm and add epsilon*L*eye

                double *pt_HtH = (double *) malloc(sizeof(double)*NSources*NSources);
                double *pt_H = (double *) malloc(sizeof(double)*NSources* NFreq);
                double *pt_HtHx = (double *) malloc(sizeof(double)*NSources);
                double *pt_x = (double *) malloc(sizeof(double)*NSources);

                transpose(pt_Ht, pt_H, NSources, NFreq);
                MM_multiply(pt_Ht, pt_H, pt_HtH, NSources, NFreq,NSources);

                Lip = PowerMethod(pt_HtH,pt_HtHx,pt_x,NSources,NSources);

                for (int z=0;z<NSources;z++){
                       pt_HtH[z*NSources+z] += eps*Lip;  // Adding epsilon*Lip, we should rather use the max
                }

                // Apply the inverse applied to pt_D using CG

                Basic_CG(pt_HtH, pt_P, pt_Pout, NSources, NSources);

                //

                for (int y=0; y < NSources; y++) Out_data(y,x) = pt_Pout[y];

                free(pt_D);
                free(pt_Pout);
                free(pt_H);
                free(pt_Ht);
                free(pt_HtH);
                free(pt_HtHx);
                free(pt_x);
                free(pt_P);
            }
            else{for (int y=0; y < NSources; y++) Out_data(y,x) = 0;}
        }

        return Out;

        }

        //####################################################
        // Update S with a complex beam
        //####################################################

        np::ndarray UpdateS_Cpx_numpy(np::ndarray X_In_r,np::ndarray X_In_i,np::ndarray M_r,np::ndarray M_i,np::ndarray mixmat,np::ndarray epsilon,np::ndarray npar,np::ndarray nfreq,np::ndarray nx){

         NumPyArrayData<double> X_data_r(X_In_r);
         NumPyArrayData<double> B_data_r(M_r);
         NumPyArrayData<double> X_data_i(X_In_i);
         NumPyArrayData<double> B_data_i(M_i);
         NumPyArrayData<double> Epsi(epsilon);
         NumPyArrayData<double> Npar(npar);
         NumPyArrayData<double> n_x(nx);
         NumPyArrayData<double> Nfreq(nfreq);
         NumPyArrayData<double> MixMat(mixmat);


         double eps = Epsi(0);
         long NSources = Npar(0);  // Number of sources
         long NFreq = Nfreq(0);  // Number of observations
         long Nx = n_x(0);  // Number of elements in k-space

         np::ndarray Out = np::zeros(bp::make_tuple(NSources,Nx,2), np::dtype::get_builtin<double>());
         NumPyArrayData<double> Out_data(Out);

         //

         int x=0;

         #pragma omp parallel for shared(x, X_data_r, B_data_r, X_data_i, B_data_i, Out_data, MixMat,eps, NSources,NFreq)
         for (x=0; x < Nx; x++) {

             // Check B_data first

             double val = 1e-24;
             for (int y=0; y < NFreq; y++) {
                 val += B_data_r(y,x)*B_data_r(y,x) + B_data_i(y,x)*B_data_i(y,x);
             }

             if (val > 1e-15){ // If there's enough observations

                 double *pt_Ht_r = (double *) malloc(sizeof(double)*NFreq*NSources);  // This is the mixing matrix
                 double *pt_Ht_i = (double *) malloc(sizeof(double)*NFreq*NSources);
                 double *pt_Ht = (double *) malloc(sizeof(double)*NFreq*NSources);
                 double *pt_D_r = (double *) malloc(sizeof(double)*NFreq); // This is the datum
                 double *pt_D_i = (double *) malloc(sizeof(double)*NFreq);
                 double *pt_Pout_r = (double *) malloc(sizeof(double)*NSources);  // Output sources
                 double *pt_Pout_i = (double *) malloc(sizeof(double)*NSources);
                 double *pt_P_rr = (double *) malloc(sizeof(double)*NSources); // These are the sources
                 double *pt_P_ii = (double *) malloc(sizeof(double)*NSources);
                 double *pt_P_ri = (double *) malloc(sizeof(double)*NSources);
                 double *pt_P_ir = (double *) malloc(sizeof(double)*NSources);
                 double Lip = 1e-24;  // Lipschitz ct

                 // Multiplying by Ht

                 for (int y=0; y < NFreq; y++) {
                     pt_D_r[y] = X_data_r(y,x);   // Store the data
                     pt_D_i[y] = X_data_i(y,x);
                     for (int z=0;z<NSources;z++){
                         pt_Ht_r[z*NFreq+y] = B_data_r(y,x)*MixMat(y,z);  // Already the transpose of A and apply the beam
                         pt_Ht_i[z*NFreq+y] = B_data_i(y,x)*MixMat(y,z);
                         pt_Ht[z*NFreq+y] = std::sqrt(B_data_i(y,x)*B_data_i(y,x) + B_data_r(y,x)*B_data_r(y,x))*MixMat(y,z);
                     }
                 }

                 // Apply the matrix to the data

                 MV_multiply(pt_Ht_r, pt_D_r, pt_P_rr, NSources, NFreq);
                 MV_multiply(pt_Ht_i, pt_D_i, pt_P_ii, NSources, NFreq);

                 MV_multiply(pt_Ht_r, pt_D_i, pt_P_ri, NSources, NFreq);
                 MV_multiply(pt_Ht_i, pt_D_r, pt_P_ir, NSources, NFreq);

                 for (int z=0;z<NSources;z++){
                   pt_P_rr[z] = pt_P_rr[z] + pt_P_ii[z];
                   pt_P_ii[z] = pt_P_ri[z] - pt_P_ir[z];
                 }

                 // Computing HtH, its norm and add epsilon*L*eye

                 double *pt_HtH = (double *) malloc(sizeof(double)*NSources*NSources);
                 double *pt_H = (double *) malloc(sizeof(double)*NSources* NFreq);
                 double *pt_HtHx = (double *) malloc(sizeof(double)*NSources);
                 double *pt_x = (double *) malloc(sizeof(double)*NSources);

                 transpose(pt_Ht, pt_H, NSources, NFreq);
                 MM_multiply(pt_Ht, pt_H, pt_HtH, NSources, NFreq,NSources);

                Lip = PowerMethod(pt_HtH,pt_HtHx,pt_x,NSources,NSources);

                 for (int z=0;z<NSources;z++){
                        pt_HtH[z*NSources+z] += eps*(Lip + 1e-24);//*Lip;  // Adding epsilon*Lip, we should rather use the max
                 }

                 // Apply the inverse applied to pt_D using CG

                 Basic_CG(pt_HtH, pt_P_rr, pt_Pout_r, NSources, NSources+10);
                 Basic_CG(pt_HtH, pt_P_ii, pt_Pout_i, NSources, NSources+10);

                 //

                 for (int y=0; y < NSources; y++) {
                   Out_data(y,x,0) = pt_Pout_r[y]; // Real part
                   Out_data(y,x,1) = pt_Pout_i[y]; // Imaginary part
                 }

                 free(pt_D_r);
                 free(pt_D_i);
                 free(pt_Pout_r);
                 free(pt_Pout_i);
                 free(pt_H);
                 free(pt_Ht);
                 free(pt_HtH);
                 free(pt_HtHx);
                 free(pt_x);
                 free(pt_P_rr);
                 free(pt_P_ii);
                 free(pt_P_ri);
                 free(pt_P_ir);
             }
             else{
               for (int y=0; y < NSources; y++){
                 Out_data(y,x,0) = 1e-24;
                 Out_data(y,x,1) = 1e-24;
               }
             }

         }

         return Out;

         }

       //####################################################
       // APPLY (Ht H + epsilon*L*eye(3))-1 Ht
       //####################################################

        np::ndarray GradientS_numpy(np::ndarray X_In,np::ndarray M,np::ndarray S,np::ndarray mixmat,np::ndarray npar,np::ndarray nfreq,np::ndarray nx){

        NumPyArrayData<double> X_data(X_In);
        NumPyArrayData<double> B_data(M);
        NumPyArrayData<double> S_data(S);
        NumPyArrayData<double> Npar(npar);
        NumPyArrayData<double> n_x(nx);
        NumPyArrayData<double> Nfreq(nfreq);
        NumPyArrayData<double> MixMat(mixmat);


        long NSources = Npar(0);  // Number of sources
        long NFreq = Nfreq(0);  // Number of observations
        long Nx = n_x(0);  // Number of elements in k-space

        np::ndarray Out = np::zeros(bp::make_tuple(NSources+1,Nx), np::dtype::get_builtin<double>()); // The last row contains the Lipschitz ct per Fourier coeff
        NumPyArrayData<double> Out_data(Out);

        int x=0;

        #pragma omp parallel for shared(x, X_data, B_data, Out_data, MixMat, S_data, NSources,NFreq)
        for (x=0; x < Nx; x++) {

            // Check B_data first

            double val = 0;
            for (int y=0; y < NFreq; y++) {
                val += B_data(y,x);
            }

            if (val > 0){ // If there's enough observations

                double *pt_Ap = (double *) malloc(sizeof(double)*NFreq*NSources);  // This is the weighted mixing matrix
                double *pt_Ap_t = (double *) malloc(sizeof(double)*NFreq*NSources);  // Its transpose
                double *pt_AtAp = (double *) malloc(sizeof(double)*NSources*NSources);
                double *pt_D = (double *) malloc(sizeof(double)*NFreq); // This is the datum
                double *pt_S = (double *) malloc(sizeof(double)*NSources);
                double *pt_AtAS = (double *) malloc(sizeof(double)*NSources); // These are the sources
                double *pt_AtD = (double *) malloc(sizeof(double)*NSources);
                double Lip = 0;  // Lipschitz ct

                // Multiplying by Ht

                for (int y=0; y < NFreq; y++) {
                    pt_D[y] = X_data(y,x);   // Store the data
                    for (int z=0;z<NSources;z++){
                        pt_Ap_t[z*NFreq + y] = B_data(y,x)*MixMat(y,z);
                        pt_Ap[z + y*NSources] = B_data(y,x)*MixMat(y,z);
                    }
                }

                for (int z=0;z<NSources;z++) pt_S[z] = S_data(z,x);

                // Compute AtA
                MM_multiply(pt_Ap_t, pt_Ap, pt_AtAp, NSources, NFreq,NSources);

                // Compute At X
                MV_multiply(pt_Ap_t, pt_D, pt_AtD, NSources, NFreq);

                // Compute AtA S
                MV_multiply(pt_AtAp, pt_S, pt_AtAS, NSources, NSources);

                // Output the results

                for (int y=0; y < NSources; y++) Out_data(y,x) = 2.*pt_AtAS[y] - 2.*pt_AtD[y];

                // Spectral norm of AtA

                Lip = PowerMethod(pt_AtAp,pt_AtAS,pt_S,NSources,NSources);

                Out_data(NSources,x) = Lip;

                free(pt_D);
                free(pt_S);
                free(pt_Ap);
                free(pt_AtAp);
                free(pt_AtD);
                free(pt_AtAS);
                free(pt_Ap_t);
            }
            else{for (int y=0; y < NSources+1; y++) Out_data(y,x) = 0;}
        }

        return Out;

        }

       //####################################################
       // Updating A
       //####################################################

       np::ndarray UpdateA_numpy(np::ndarray Xr,np::ndarray Xi,np::ndarray Sr,np::ndarray Si,np::ndarray M,np::ndarray npar,np::ndarray nfreq,np::ndarray nx){

        NumPyArrayData<double> Xr_data(Xr);
        NumPyArrayData<double> Xi_data(Xi);
        NumPyArrayData<double> Sr_data(Sr);
        NumPyArrayData<double> Si_data(Si);
        NumPyArrayData<double> B_data(M);
        NumPyArrayData<double> Npar(npar);
        NumPyArrayData<double> n_x(nx);
        NumPyArrayData<double> Nfreq(nfreq);

        long NCol = Npar(0);  // Number of sources
        long NRow = Nfreq(0);  // Number of observations
        long Npix = n_x(0);  // Number of elements in k-space

        np::ndarray A_out = np::zeros(bp::make_tuple(NRow,NCol), np::dtype::get_builtin<double>());
        NumPyArrayData<double> A_out_data(A_out);

        // Allocating what needs to be allocated

        double * Xloop = (double *) malloc(sizeof(double)*Npix);
        double * Xloop_i = (double *) malloc(sizeof(double)*Npix);
        double * Aloop = (double *) malloc(sizeof(double)*NCol);
        double * Sloop = (double *) malloc(sizeof(double)*NCol*Npix);
        double * Sloop_i = (double *) malloc(sizeof(double)*NCol*Npix);

        for (int i=0;i<NRow;i++){

            for (int j=0;j<Npix;j++)
            {
              Xloop[j] = Xr_data(i,j);
              Xloop_i[j] = Xi_data(i,j);
              for (int k=0;k<NCol;k++){
                Sloop[j + k*Npix] = B_data(i,j)*Sr_data(k,j);
                Sloop_i[j + k*Npix] = B_data(i,j)*Si_data(k,j);
              }
            }

            // Running LCG for a single batch

            //R_CG2(Aloop, Xloop, Sloop, 1, NCol, Npix, 100);
            R_CG_cpx(Aloop, Xloop, Xloop_i, Sloop, Sloop_i, 1, NCol, Npix, 100);

            for (int k=0;k<NCol;k++)  A_out_data(i,k) = Aloop[k] ;

        }

        free(Xloop);
        free(Xloop_i);
        free(Aloop);
        free(Sloop);
        free(Sloop_i);

      return A_out;

    }

    //####################################################
    // Updating A
    //####################################################

    np::ndarray UpdateA_Cpx_numpy(np::ndarray Xr,np::ndarray Xi,np::ndarray Sr,np::ndarray Si,np::ndarray Mr,np::ndarray Mi,np::ndarray npar,np::ndarray nfreq,np::ndarray nx){

     NumPyArrayData<double> Xr_data(Xr);
     NumPyArrayData<double> Xi_data(Xi);
     NumPyArrayData<double> Sr_data(Sr);
     NumPyArrayData<double> Si_data(Si);
     NumPyArrayData<double> Br_data(Mr);
     NumPyArrayData<double> Bi_data(Mi);
     NumPyArrayData<double> Npar(npar);
     NumPyArrayData<double> n_x(nx);
     NumPyArrayData<double> Nfreq(nfreq);

     long NCol = Npar(0);  // Number of sources
     long NRow = Nfreq(0);  // Number of observations
     long Npix = n_x(0);  // Number of elements in k-space

     np::ndarray A_out = np::zeros(bp::make_tuple(NRow,NCol), np::dtype::get_builtin<double>());
     NumPyArrayData<double> A_out_data(A_out);

     // Allocating what needs to be allocated

     double * Xloop = (double *) malloc(sizeof(double)*Npix);
     double * Xloop_i = (double *) malloc(sizeof(double)*Npix);
     double * Aloop = (double *) malloc(sizeof(double)*NCol);
     double * Sloop = (double *) malloc(sizeof(double)*NCol*Npix);
     double * Sloop_i = (double *) malloc(sizeof(double)*NCol*Npix);

     for (int i=0;i<NRow;i++){

         for (int j=0;j<Npix;j++)
         {
           Xloop[j] = Xr_data(i,j);
           Xloop_i[j] = Xi_data(i,j);
           for (int k=0;k<NCol;k++){
             Sloop[j + k*Npix] = Br_data(i,j)*Sr_data(k,j) - Bi_data(i,j)*Si_data(k,j);
             Sloop_i[j + k*Npix] = Br_data(i,j)*Si_data(k,j) + Bi_data(i,j)*Sr_data(k,j);
           }
         }

         // Running LCG for a single batch

         //R_CG2(Aloop, Xloop, Sloop, 1, NCol, Npix, 100);
         R_CG_cpx(Aloop, Xloop, Xloop_i, Sloop, Sloop_i, 1, NCol, Npix, 100);

         for (int k=0;k<NCol;k++)  A_out_data(i,k) = Aloop[k] ;

     }

     free(Xloop);
     free(Xloop_i);
     free(Aloop);
     free(Sloop);
     free(Sloop_i);

   return A_out;

  }


    //####################################################
    // Updating A - Batches
    //####################################################

       np::ndarray UpdateA_batches_numpy(np::ndarray Xr,np::ndarray Xi,np::ndarray Sr,np::ndarray Si,np::ndarray M,np::ndarray npar,np::ndarray nfreq,np::ndarray nx,np::ndarray nb){

        NumPyArrayData<double> Xr_data(Xr);
        NumPyArrayData<double> Xi_data(Xi);
        NumPyArrayData<double> Sr_data(Sr);
        NumPyArrayData<double> Si_data(Si);
        NumPyArrayData<double> B_data(M);
        NumPyArrayData<double> Npar(npar);
        NumPyArrayData<double> n_x(nx);
        NumPyArrayData<double> Nfreq(nfreq);
        NumPyArrayData<double> Nb(nb);

        long NCol = Npar(0);  // Number of sources
        long NRow = Nfreq(0);  // Number of observations
        long Npix = n_x(0);  // Number of elements in k-space
        long Nbatches = Nb(0);

        np::ndarray A_out = np::zeros(bp::make_tuple(NRow,NCol,Nbatches), np::dtype::get_builtin<double>());
        NumPyArrayData<double> A_out_data(A_out);

        // Allocating what needs to be allocated

        int batch = 0;

        #pragma omp parallel for shared(batch, Xr_data, Xi_data, Sr_data, Si_data, B_data, A_out_data, NCol,Npix,NRow)
        for (batch =0;batch<Nbatches;batch++){ // loop on batches

            double * Xloop = (double *) malloc(sizeof(double)*Npix);
            double * Xloop_i = (double *) malloc(sizeof(double)*Npix);
            double * Aloop = (double *) malloc(sizeof(double)*NCol);
            double * Sloop = (double *) malloc(sizeof(double)*NCol*Npix);
            double * Sloop_i = (double *) malloc(sizeof(double)*NCol*Npix);

            for (int i=0;i<NRow;i++){

                for (int j=0;j<Npix;j++)
                {
                  Xloop[j] = Xr_data(i,j,batch);
                  Xloop_i[j] = Xi_data(i,j,batch);
                  for (int k=0;k<NCol;k++){
                    Sloop[j + k*Npix] = B_data(i,j,batch)*Sr_data(k,j,batch);
                    Sloop_i[j + k*Npix] = B_data(i,j,batch)*Si_data(k,j,batch);
                  }
                }

                // Running LCG for a single batch

                //R_CG2(Aloop, Xloop, Sloop, 1, NCol, Npix, 100);
                R_CG_cpx(Aloop, Xloop, Xloop_i, Sloop, Sloop_i, 1, NCol, Npix, 100);

                for (int k=0;k<NCol;k++)  A_out_data(i,k,batch) = Aloop[k] ;

            }

            free(Xloop);
            free(Xloop_i);
            free(Aloop);
            free(Sloop);
            free(Sloop_i);

        }

      return A_out;

    }

    //####################################################
    // Updating A - Batches
    //####################################################

       np::ndarray UpdateA_Cpx_batches_numpy(np::ndarray Xr,np::ndarray Xi,np::ndarray Sr,np::ndarray Si,np::ndarray Mr,np::ndarray Mi,np::ndarray npar,np::ndarray nfreq,np::ndarray nx,np::ndarray nb){

        NumPyArrayData<double> Xr_data(Xr);
        NumPyArrayData<double> Xi_data(Xi);
        NumPyArrayData<double> Sr_data(Sr);
        NumPyArrayData<double> Si_data(Si);
        NumPyArrayData<double> Br_data(Mr);
        NumPyArrayData<double> Bi_data(Mi);
        NumPyArrayData<double> Npar(npar);
        NumPyArrayData<double> n_x(nx);
        NumPyArrayData<double> Nfreq(nfreq);
        NumPyArrayData<double> Nb(nb);

        long NCol = Npar(0);  // Number of sources
        long NRow = Nfreq(0);  // Number of observations
        long Npix = n_x(0);  // Number of elements in k-space
        long Nbatches = Nb(0);

        np::ndarray A_out = np::zeros(bp::make_tuple(NRow,NCol,Nbatches), np::dtype::get_builtin<double>());
        NumPyArrayData<double> A_out_data(A_out);

        // Allocating what needs to be allocated

        int batch = 0;

        #pragma omp parallel for shared(batch, Xr_data, Xi_data, Sr_data, Si_data, Br_data, Bi_data, A_out_data, NCol,Npix,NRow)
        for (batch =0;batch<Nbatches;batch++){ // loop on batches

            double * Xloop = (double *) malloc(sizeof(double)*Npix);
            double * Xloop_i = (double *) malloc(sizeof(double)*Npix);
            double * Aloop = (double *) malloc(sizeof(double)*NCol);
            double * Sloop = (double *) malloc(sizeof(double)*NCol*Npix);
            double * Sloop_i = (double *) malloc(sizeof(double)*NCol*Npix);

            for (int i=0;i<NRow;i++){

                for (int j=0;j<Npix;j++)
                {
                  Xloop[j] = Xr_data(i,j,batch);
                  Xloop_i[j] = Xi_data(i,j,batch);
                  for (int k=0;k<NCol;k++){
                    Sloop[j + k*Npix] = Br_data(i,j)*Sr_data(k,j) - Bi_data(i,j)*Si_data(k,j);
                    Sloop_i[j + k*Npix] = Br_data(i,j)*Si_data(k,j) + Bi_data(i,j)*Sr_data(k,j);
                  }
                }

                // Running LCG for a single batch

                //R_CG2(Aloop, Xloop, Sloop, 1, NCol, Npix, 100);
                R_CG_cpx(Aloop, Xloop, Xloop_i, Sloop, Sloop_i, 1, NCol, Npix, 100);

                for (int k=0;k<NCol;k++)  A_out_data(i,k,batch) = Aloop[k] ;

            }

            free(Xloop);
            free(Xloop_i);
            free(Aloop);
            free(Sloop);
            free(Sloop_i);

        }

      return A_out;

    }

    //####################################################
    // Gradient w.r. to A - Batches
    //####################################################

       np::ndarray GradientA_numpy(np::ndarray Xr,np::ndarray Xi,np::ndarray Sr,np::ndarray Si,np::ndarray A,np::ndarray M,np::ndarray npar,np::ndarray nfreq,np::ndarray nx,np::ndarray nb){

        NumPyArrayData<double> Xr_data(Xr);
        NumPyArrayData<double> Xi_data(Xi);
        NumPyArrayData<double> Sr_data(Sr);
        NumPyArrayData<double> Si_data(Si);
        NumPyArrayData<double> A_data(A);
        NumPyArrayData<double> B_data(M);
        NumPyArrayData<double> Npar(npar);
        NumPyArrayData<double> n_x(nx);
        NumPyArrayData<double> Nfreq(nfreq);
        NumPyArrayData<double> Nb(nb);

        long NCol = Npar(0);  // Number of sources
        long NRow = Nfreq(0);  // Number of observations
        long Npix = n_x(0);  // Number of elements in k-space
        long Nbatches = Nb(0);

        np::ndarray A_out = np::zeros(bp::make_tuple(NRow,NCol+1,Nbatches), np::dtype::get_builtin<double>());
        NumPyArrayData<double> A_out_data(A_out);

        // Allocating what needs to be allocated

        int batch = 0;

        //#pragma omp parallel for shared(batch, Xr_data, Xi_data, Sr_data, Si_data, A_data,B_data, A_out_data, NCol,Npix,NRow)
        for (batch =0;batch<Nbatches;batch++){ // loop on batches

            double * Xloop = (double *) malloc(sizeof(double)*Npix);
            double * Xloop_i = (double *) malloc(sizeof(double)*Npix);
            double * Aloop = (double *) malloc(sizeof(double)*NCol);
            double * Sloop = (double *) malloc(sizeof(double)*NCol*Npix);
            double * Sloop_i = (double *) malloc(sizeof(double)*NCol*Npix);
            double * pt_SSt = (double *) malloc(sizeof(double)*NCol*NCol);
            double * pt_aSSt = (double *) malloc(sizeof(double)*NCol);
            double * pt_XSt = (double *) malloc(sizeof(double)*NCol);

            double Lip = 0;
            double temp = 0;

            for (int i=0;i<NRow;i++){

                for (int j=0;j<Npix;j++)
                {
                  Xloop[j] = Xr_data(i,j,batch);
                  Xloop_i[j] = Xi_data(i,j,batch);
                  for (int k=0;k<NCol;k++){
                    Sloop[j + k*Npix] = B_data(i,j,batch)*Sr_data(k,j,batch);
                    Sloop_i[j + k*Npix] = B_data(i,j,batch)*Si_data(k,j,batch);
                  }
                }

                for (int k=0;k<NCol;k++) Aloop[k] = A_data(i,k);

                // Compute XSt
                for (int k=0;k<NCol;k++) pt_XSt[k]=0;
                for (int k=0;k<NCol;k++) for (int j=0;j<Npix;j++) pt_XSt[k] += Xloop[j]*Sloop[j + k*Npix] + Xloop_i[j]*Sloop_i[j + k*Npix];

                // Compute SSt

                for (int j=0;j<NCol;j++)
                {
                    for (int k=0;k<NCol;k++)
                    {
                        temp = 0;
                        for (int l=0;l<Npix;l++)
                        {
                            temp += Sloop[l*NCol + k]*Sloop[l*NCol + j]+Sloop_i[l*NCol + k]*Sloop_i[l*NCol + j];
                        }
                        pt_SSt[j*NCol + k] = temp;
                    }
                }


                // Compute pt_aSSt

                MV_multiply(pt_SSt, Aloop, pt_aSSt, NCol, NCol);

                for (int k=0;k<NCol;k++)  A_out_data(i,k,batch) = 2.*pt_aSSt[k] - 2.*pt_XSt[k];

                // Spectral norm

                for (int k=0;k<NCol;k++) pt_XSt[k]=0;
                for (int k=0;k<NCol;k++) pt_aSSt[k]=0;
                Lip = PowerMethod(pt_SSt,pt_XSt,pt_aSSt,NCol,NCol);

                A_out_data(i,NCol,batch) = Lip;

            }

            free(Xloop);
            free(Xloop_i);
            free(Aloop);
            free(Sloop);
            free(Sloop_i);
            free(pt_SSt);
            free(pt_aSSt);
            free(pt_XSt);

        }

      return A_out;

    }

private:

    int Nx, NFreq, Nf;  // Not used

};

#endif // PLANCK_UTILS_H
