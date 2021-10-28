"""
Distributed Generalized Morphological Analysis (DGMCA)
Python 2.7 version.



This code is not actually parallelized but it was done to allow an easy preparation of a
future parallelized version.
The code solves the Blind Source Separation (BSS) problem giving as a result the
source and the mixing matrices.

Solving de BSS problem when the images are sparse in another domain.
The parallelization is done only over the columns.
Permutation of the columns is optional via the shufflingOpt parameter (Recomended).
Threshold strategy: using exponential decay and norm_inf
Weight for FM strategy: SNR over the S space

  Usage:
    Ex for J=0:
    Results = DGMCA(X,n=5,mints=3,nmax=100,L0=1,verb=0,Init=0,BlockSize= None,Kmax=1.,AInit=None,tol=1e-6,\
    subBlockSize=500, SCOpt=1,alphaEstOpt=1,alpha_exp=0.5)
    Ex. for J>0:
    Results = DGMCA(X,n=5,mints=3,nmax=100,L0=1,verb=0,Init=0,BlockSize= None,Kmax=1.,AInit=None,tol=1e-6,\
    subBlockSize=500, SCOpt=1,alphaEstOpt=1,alpha_exp=0.5,J=3,WNFactors=WNFactors, normOpt=normOpt)


  Inputs:
    X            : m x t array (input data, each row corresponds to a single observation)
    n            : scalar (number of sources to be estimated)
    mints        : scalar (final value of the k-mad thresholding)
    nmax         : scalar (number of iterations)
    q_f          : scalar (amca parameter to evaluate to correlation between the sources; should be lower than 1)
    L0           : if set to 1, L0 rather than L1 penalization
    verb         : boolean (if set to 1, in verbose mode)
    Init         : scalar (if set to 0: PCA-based initialization, if set to 1: random initialization)
    BlockSize    : scalar (should be lower than number of sources. The number of random sources to update at each iteration)
    Kmax         : scalar (maximal L0 norm of the sources. Being a percentage, it should be between 0 and 1)
    AInit        : if not None, provides an initial value for the mixing matrix
    tol          : scalar (tolerance on the mixing matrix criterion)
    subBlockSize : scalar (size of the subproblem (t_j), must be smaller than t(columns of X).)
    alpha_exp    : scalar (parameter controling the exponential decay of the thresholding strategy)
    alphaEstOpt  : Use online estimation for the alpha_exp thresholding parameter
                    # 0 --> Deactivated
                    # 1 --> Activated
    SCOpt        : Weighting for the partially correlated sources
                    # 0 --> Deactivated
                    # 1 --> Activated
    J            : scalar (Wavelet decomposition level. J=0 means no decomposition at all.)
    normOpt      : Normalize wavelet decomposition levels (for J>0)
                    # 0 --> Deactivated
                    # 1 --> Activated
    WNFactors    : Factors to correct the amplitude of the different levels of the wavelet decomposition (output of the data_input_preparation() function)(for J>0)


  Outputs:
   Results : dict with entries:
        if J = 0:
            sources    : n x t array (estimated sources)
            mixmat     : m x n array (estimated mixing matrix)
        if J > 0:
            sources    : n x t array (estimated sources)
            mixmat     : m x n array (estimated mixing matrix)
            images     :

  Description:
    Computes the sources and the mixing matrix with DGMCA.

  Example:
     S,A = GMCA(X,n=2,mints=0,nmax=500) will perform GMCA assuming that the data are noiseless

  Version
    v1 - November,28 2017 - J.Bobin - CEA Saclay
    v2 - ... 2018 - T.Liaudat - CEA Saclay
    v3 - ... 2019 - J.Bobin - CEA Saclay

"""

import numpy as np
import scipy.linalg as lng
import copy as cp
import time
from utils_dgmca import randperm
from utils_dgmca import mad
from FrechetMean import FrechetMean
from misc_dgmca import *
from starlet import *
import maniutils_py as mp
from utils_dgmca import EvalCriterion

################# DGMCA Main function

def DGMCA(X,n=2,mints=3,nmax=100,q_f=0.1,L0=0,verb=0,Init=0,BlockSize= None,\
    Kmax=0.5,AInit=None,tol=1e-6,subBlockSize=100,alphaEstOpt=1,SCOpt=1,alpha_exp=2.,J=0\
    ,WNFactors=0,normOpt=0,FullyRandom=False,UseStopping=False):

    nX = np.shape(X);
    m = nX[0];t = nX[1]

    if J > 0:
        Xw, line_id, r_id, ccX = reshapeDecData(X,batch_size=subBlockSize,J=J,normOpt=normOpt,WNFactors=WNFactors)
    else:
        line_id = 0
        r_id = 0
        Xw = X

    if BlockSize == None:
        BlockSize = n

    if verb:
        print("Initializing ...")
    if Init == 0:
        R = np.dot(Xw,Xw.T)
        D,V = lng.eig(R)
        A = V[:,0:n]
    if Init == 1:
        A = np.random.randn(m,n)
    if AInit != None:
        A = cp.deepcopy(AInit)

    for r in range(0,n):
        A[:,r] = A[:,r]/lng.norm(A[:,r])

    S = np.dot(A.T,Xw);

    # Call the core algorithm
    S,A = Core_DGMCA(X=Xw,A=A,S=S,n=n,q_f=q_f,BlockSize = BlockSize,tol=tol,kend = mints,nmax=nmax,L0=L0,\
        verb=verb,Kmax=Kmax,subBlockSize=subBlockSize,SCOpt=SCOpt,alphaEstOpt=alphaEstOpt,alpha_exp=alpha_exp,FullyRandom=FullyRandom,UseStopping=UseStopping);

    if J > 0:
        Ra = np.dot(A.T,A)
        Ua,Sa,Va = np.linalg.svd(Ra)
        Sa[Sa < np.max(Sa)*1e-9] = np.max(Sa)*1e-9
        iRa = np.dot(Va.T,np.dot(np.diag(1./Sa),Ua.T))
        piA = np.dot(iRa,A.T)
        ccS = np.dot(piA,ccX)

        S_2Ds = recoverDecData(S,r_id,line_id,J=J,ccX=ccS,deNormOpt=normOpt,WNFactors=WNFactors)

        Results = {"sources":S,"mixmat":A,"images":S_2Ds}

    else:
        Results = {"sources":S,"mixmat":A}


    return Results

################# DGMCA internal code

def Core_DGMCA(X=0,Xbatches=None,n=0,A=0,S=0,kend=3,q_f=0.1,nmax=100,BlockSize = 2,L0=1,verb=0,Kmax=0.5,\
    tol=1e-6, subBlockSize=100, alphaEstOpt=1,alpha_exp=2.,AggMeth='FM',Equal=True,shufflingOpt=False,A0 = None,FullyRandom=False,TwoD = False,UseStopping=True):

#--- Initialization variables

    # Batch related variables
    numBlocks = np.floor(X.shape[1]/(subBlockSize*1.)).astype(int) # Number of blocks
    lastBlockSize = X.shape[1]%subBlockSize # Size of the last block if its not the same size as the other blocks

    # FM related variables
    A_block = np.zeros([A.shape[0],A.shape[1],numBlocks])
    Aold = cp.deepcopy(A)
    perc = Kmax/(nmax*1.) # Kmax should be 1
    A_mean = A
    w_FM = np.zeros([n,numBlocks]) # FM weight of each column of each estimation of A
    K = kend # K from the K-sigma_MAD strategy for the threshold final value

    # Threshold related variables
    thresh_block = np.zeros([n,numBlocks])
    mad_block = np.zeros([n,numBlocks])

    if L0 == 1:
        L1 = False
    else:
        L1 = True

    Go_On = 1
    it = 1

    Output_SAD = []
    Output_CA = []
    Output_SADm = []
    Output_CAm = []

    ## Init

    BSize = subBlockSize
    Nb = numBlocks
    Ntot = np.max([1,np.int(X.shape[1]/BSize)])*BSize

    if Xbatches is None:
        if shufflingOpt: # Shuffling
            print('Shuffling the inputs')
            shuffling_id = randperm(X.shape[1])
            if TwoD:
                print("TwoD")
                Xbatches = np.reshape(X[:,shuffling_id[0:Ntot]],(X.shape[0],Nb,-1)).swapaxes(1,2)
            else:
                Xbatches = np.reshape(X[:,shuffling_id[0:Ntot]],(X.shape[0],BSize,Nb))
        else:
            if TwoD:
                print("TwoD")
                Xbatches = np.reshape(X[:,0:Ntot],(X.shape[0],Nb,-1)).swapaxes(1,2)
                import scipy.io as sio
                sio.savemat("data.mat",mdict={"data":Xbatches})
            else:
                Xbatches = np.reshape(X[:,0:Ntot],(X.shape[0],BSize,Nb))
    else:
        BSize = Xbatches.shape[1]
        Nb = Xbatches.shape[2]

    if verb:
        print("Starting main loop ...")
        print(" ")
        print("  - Final k: ",kend)
        print("  - Maximum number of iterations: ",nmax)
        if L0:
            print("  - Using L0 norm rather than L1")
        print(" ")
        print(" ... processing ...")
        start_time = time.time()
        print('Total size : ',Ntot," - Number of blocks: ",Nb)

    # Initialize the parameters

    #_,MadVal,MaxVal,_ = Update_A_omp(Xbatches,A_mean,threshold,L1=False,FixedCol=None)
    S = np.linalg.inv(A_mean.T@A_mean)@(A_mean.T@X)# Should be made in //
    threshold = np.zeros((n,))
    S_norm_inf = np.zeros((n,))
    mad_S = np.zeros((n,))
    vDeltaA = []

    for r in range(n):
        mad_S[r] = mad(S[r,:])
        S_norm_inf[r] = np.max(abs(S[r,:]))

    # S_norm_inf = np.max(MaxVal,axis=1) # Calculation of the total maxima (max of maxs)
    # mad_S = np.median(MadVal,axis=1) # Naive distributed estimation of mad(S)

#--- Main loop

    while Go_On:

        it += 1

        if it == nmax:
            Go_On = 0

#------ Initialization of the parameters

        for r in range(n):
            threshold[r] = K*mad_S[r] + (S_norm_inf[r] - K*mad_S[r])*np.exp(-1*(it-2)*alpha_exp)

        tA = time.time()

        A_block,MadVal,MaxVal,w_FM = Update_A_omp(Xbatches,A_mean,threshold,L1=L1,FixedCol=None)

        S_norm_inf = np.max(MaxVal,axis=1) # Calculation of the total maxima (max of maxs)
        mad_S = np.median(MadVal,axis=1) # Naive distributed estimation of mad(S)

#------ Fusion of mixing matrices A
        if numBlocks == 1:
            A_mean = A_block[:,:,0]
        else:
# #---------- Fusion method: The Frechet Mean

            if AggMeth == 'FM' :
                if Equal:
                    A_mean = mp.FrechetMean(A_block,np.ones_like(w_FM)/Nb)
                else:
                    A_mean = mp.FrechetMean(A_block,w_FM)
            if AggMeth == 'pFM' :
                if Equal:
                    A_mean = mp.pFrechetMean(A_block,np.ones_like(w_FM)/Nb,Ain=A_mean)
                else:
                    A_mean = mp.pFrechetMean(A_block,w_FM,Ain=A_mean)
            if AggMeth == 'rFM': # Robust
                if Equal:
                    A_mean = mp.FrechetRobustMean(A_block,np.ones_like(w_FM)/Nb)
                else:
                    A_mean = mp.FrechetRobustMean(A_block,w_FM)
            if AggMeth == 'prFM': # Robust
                if Equal:
                    A_mean = mp.pFrechetRobustMean(A_block,np.ones_like(w_FM)/Nb,Ain=A_mean)
                else:
                    A_mean = mp.pFrechetRobustMean(A_block,w_FM,Ain=A_mean)
            if AggMeth == 'Euclidean': # Robust
                if Equal:
                    A_mean = np.mean(A_block,axis=2)
                else:
                    A_mean = np.sum(A_block*np.tile(w_FM.reshape((1,w_FM.shape[0],w_FM.shape[1])),(X.shape[0],1,1)),axis=2)

                A_mean = A_mean/np.maximum(1e-12,np.linalg.norm(A_mean,axis=0))

            DeltaA = abs(1.-np.max(abs(np.sum(Aold*A_mean,axis=0))))
            vDeltaA.append(DeltaA)
            Aold = cp.deepcopy(A_mean)

        if A0 is not None:

            val_sad = []
            val_ca = []

            for q in range(A_block.shape[2]):
                At = A_block[:,:,q]
                val_ca.append(EvalCriterion(A0,A0.T,At,At.T)['ca_med'])
                Aq = CorrectPerm(A0,At)
                val_sad.append(np.sum(A0*Aq,axis=0))

            Output_CA.append(val_ca)
            Output_SAD.append(val_sad)

            Output_CAm.append(EvalCriterion(A0,A0.T,A_mean,At.T)['ca_med'])
            Aq = CorrectPerm(A0,A_mean)
            Output_SADm.append(abs(np.sum(A0[:,0]*Aq[:,0])))

        if FullyRandom:
            shuffling_id = randperm(X.shape[1])
            Xbatches = np.reshape(X[:,shuffling_id[0:Ntot]],(X.shape[0],BSize,Nb)) # Assumes that we can correctly decompose the data (otherwise we can put zeros !)

        tB = time.time()

#------ Stopping criterion
        fstop = DeltaA
        if UseStopping:
            nummax = 10
            if len(vDeltaA) > nummax:
                fstop = np.mean(vDeltaA[::-1][0:nummax])
                if fstop < tol: # average of the last nummax
                    break

#------ End of the block loop
        if verb:
            print("Iteration #",it," - Delta = ",fstop,' - Norm of A',np.linalg.norm(A_mean))
            if np.mod(it,100) == 0:
                print(" ")
                print("Time for a single iteration :",tB-tA)
                print("Remaining time :",(tB-tA)*(nmax-it))

#--- End of the main loop
    if verb:
        elapsed_time = time.time() - start_time
        print("Stopped after ",it," iterations, in ",elapsed_time," seconds")

# Goodbye
    if A0 is not None:
        return A_mean,w_FM,threshold,Output_CA,Output_SAD,Output_CAm,Output_SADm # ADD THE ESTIMATION OF THE SOURCES

    if A0 is None:
        return A_mean,w_FM,threshold # ADD THE ESTIMATION OF THE SOURCES


#########
# bGMCA version
#########

def Core_DbGMCA(X=0,n=0,A=0,S=0,kend=3,q_f=0.1,nmax=100,BlockSize = 2,L0=1,verb=0,Kmax=0.5,\
    tol=1e-6, subBlockSize=100, SCOpt=1,alphaEstOpt=1,alpha_exp=2.,AggMeth='FM',Equal=True):

#--- Import useful modules
    # import numpy as np
    # import scipy.linalg as lng
    # import copy as cp
    # import time

#--- Initialization variables
    shufflingOpt = 1 # Complete shuffle of the columns (pixels)

    perc = Kmax/(nmax*1.) # Kmax should be 1
    Aold = cp.deepcopy(A)

    if SCOpt == 1:
        alpha = 1
        dalpha = (alpha-q_f)/nmax

    K = kend # K from the K-sigma_MAD strategy for the threshold final value

    # Batch related variables
    numBlocks = np.ceil(X.shape[1]/(subBlockSize*1.)).astype(int) # Number of blocks
    lastBlockSize = X.shape[1]%subBlockSize # Size of the last block if its not the same size as the other blocks

    # FM related variables
    A_block = np.zeros([A.shape[0],A.shape[1],numBlocks])
    A_mean = A
    w_FM = np.zeros([n,numBlocks]) # FM weight of each column of each estimation of A

    # Threshold related variables
    thresh_block = np.zeros([n,numBlocks])
    mad_block = np.zeros([n,numBlocks])

    # alpha_r parameter estimation
    GG_maxIt = 50
    GG_warm_start = 0
    if alphaEstOpt == 1:
        GG_warm_start = 1
        A_init = cp.deepcopy(A)

#     if SigmaX is None:
#         SigmaX = np.zeros((X.shape[0],))
#         for r in range(X.shape[0]):
#             SigmaX[r] = mad(X[r,:]) # Could be better ...
# #
    Go_On = 1
    it = 1
#
    if verb:
        print("Starting main loop ...")
        print(" ")
        print("  - Final k: ",kend)
        print("  - Maximum number of iterations: ",nmax)
        if L0:
            print("  - Using L0 norm rather than L1")
        print(" ")
        print(" ... processing ...")
        start_time = time.time()

    ## Init

    original_id = np.arange(X.shape[1])

    if shufflingOpt == 1: # Shuffling
        shuffling_id = randperm(X.shape[1])
        original_id = original_id[shuffling_id] # Keep track of the permutations
        X = X[:,shuffling_id]

    threshold = np.zeros((n,))

    BSize = subBlockSize
    Nb = numBlocks
    Xbatches = np.reshape(X,(X.shape[0],BSize,Nb)) # Assumes that we can correctly decompose the data (otherwise we can put zeros !)

    _,MadVal,MaxVal,_ = Update_A_omp(Xbatches,A_mean,threshold,L1=False,FixedCol=None)
    S = Update_S_omp(Xbatches,A_mean,threshold,BlockSize=BlockSize,L1=False)

    S = 0.*S ## COULD BE CHANGED

    S_norm_inf = np.max(MaxVal,axis=1) # Calculation of the total maxima (max of maxs)
    mad_S = np.median(MadVal,axis=1) # Naive distributed estimation of mad(S)

    perc = 0.5/nmax

#--- Main loop
    while Go_On:

        it += 1

        if it == nmax:
            Go_On = 0

#------ Estimation of the GG parameters
        # if it == GG_maxIt and GG_warm_start==1:
        #     alpha_exp = alpha_thresh_estimation(A_mean, X, alpha_exp,n)
        #     A_mean = A_init # Restart from the initialization point
        #     GG_warm_start = 0
        #     it = 2 # Restart the main loop
        #     if SCOpt == 1:
        #         alpha = 1
        #         dalpha = (alpha-q_f)/nmax

#------ Initializaion of the parameters

        # for r in range(n):
        #     sigma_noise = mad_S[r]
        #     threshold[r] = K*sigma_noise + (S_norm_inf[r] - K*sigma_noise)*np.exp(-1*(it-2)*alpha_exp)

        S = Update_S_omp(Xbatches,A_mean,0.*threshold,BlockSize=BlockSize,L1=False) # Just least-squares

        for r in range(S.shape[0]):
            St = S[r,:]
            tSt = K*mad(St)
            indNZ = np.where(abs(St) - tSt > 0)[0]
            thrd = mad(St[indNZ])
            Kval = np.min([np.floor(np.max([2./S.shape[1],perc*it])*len(indNZ)),S.shape[1]-1.])
            I = abs(St[indNZ]).argsort()[::-1]
            Kval = np.int(np.min([np.max([Kval,5.]),len(I)-1.]))
            IndIX = np.int(indNZ[I[Kval]])
            threshold[r] = abs(St[IndIX])

        A_block,MadVal,MaxVal_old,w_FM = block_Update_A_omp(Xbatches,A_mean,threshold,Sinit=np.reshape(S,(S.shape[0],BSize,Nb)),BlockSize=BlockSize,L1=False,FixedCol=None)



        S_norm_inf = np.max(MaxVal,axis=1) # Calculation of the total maxima (max of maxs)
        mad_S = np.median(MadVal,axis=1) # Naive distributed estimation of mad(S)

#------ Fusion of mixing matrices A
        if numBlocks == 1:

            A_mean = A_block[:,:,0]

        else:

# #---------- Fusion method: The Frechet Mean

            if AggMeth == 'FM' :
                if Equal:
                    A_mean = mp.FrechetMean(A_block,np.ones_like(w_FM)/Nb)
                else:
                    A_mean = mp.FrechetMean(A_block,w_FM)
            if AggMeth == 'rFM': # Robust
                if Equal:
                    A_mean = mp.FrechetRobustMean(A_block,np.ones_like(w_FM)/Nb)
                else:
                    A_mean = mp.FrechetRobustMean(A_block,w_FM)
            if AggMeth == 'Euclidean': # Robust
                if Equal:
                    A_mean = np.mean(A_block,axis=2)
                else:
                    A_mean = np.sum(A_block*np.tile(w_FM.reshape((1,w_FM.shape[0],w_FM.shape[1])),(X.shape[0],1,1)),axis=2)

                A_mean = A_mean/np.maximum(1e-12,np.linalg.norm(A_mean,axis=0))

            DeltaA = abs(1.-np.max(abs(np.sum(Aold*A_mean,axis=0))))
            Aold = cp.deepcopy(A_mean)

#------ AMCA matrix parameter update
        if SCOpt == 1:
            alpha -= dalpha

#------ End of the block loop
        if verb:
            print("Iteration #",it," - Delta = ",DeltaA,' - Norm of A',np.linalg.norm(A_mean))

#--- End of the main loop
    if verb:
        elapsed_time = time.time() - start_time
        print("Stopped after ",it," iterations, in ",elapsed_time," seconds")

    # if BlockSize == S.shape[0]:
    #     #S = Update_S_omp(Xbatches,A_mean,threshold,BlockSize=BlockSize,L1=False)
    #     S = np.linalg.inv(A_mean.T@A_mean)@A_mean.T@X

# Goodbye
    return A_mean,S,w_FM,threshold # ADD THE ESTIMATION OF THE SOURCES

################# Correcting permutations

def CorrectPerm_batches(cA0,cAb):

    for r in range(cAb.shape[2]):
        cAb[:,:,r] = CorrectPerm(cA0,cAb[:,:,r])

    return cAb

def CorrectPerm(cA0,cA):

    import copy as cp
    from numpy import linalg as lng
    A0 = cp.copy(cA0)
    A = cp.copy(cA)

    nX = np.shape(A0)

    for r in range(0,nX[1]):
        A[:,r] = A[:,r]/(1e-24+lng.norm(A[:,r]))
        A0[:,r] = A0[:,r]/(1e-24+lng.norm(A0[:,r]))

    try:
        Diff = abs(np.dot(lng.inv(np.dot(A0.T,A0)),np.dot(A0.T,A)))
    except np.linalg.LinAlgError:
        Diff = abs(np.dot(np.linalg.pinv(A0),A))
        print('WARNING, PSEUDO INVERSE TO CORRECT PERMUTATIONS')

    ind = np.linspace(0,nX[1]-1,nX[1])

    for ns in range(0,nX[1]):
        indix = np.where(Diff[ns,:] == max(Diff[ns,:]))[0]
        ind[ns] = indix[0]

    return A[:,ind.astype(int)]

################# Batch UPDATE A,S OMP
def Update_A_omp(X,Ainit,Thresholds,L1=False,FixedCol=None,NoiseIn=None):

    """
        X should be of size Nobs x Npix x NBatches
    """

    import os
    import sys
    WrapLoc = os.environ['PYWRAP3_LOC']
    sys.path.insert(1,WrapLoc)
    import gmca as gmca_wrap

    ns = np.shape(Ainit)[1]
    nc = np.shape(Ainit)[0]
    if FixedCol is None:
        nf = 0
    else:
        nf = np.size(FixedCol)
    nb = np.shape(X)[2]
    bpix = np.shape(X)[1]

    in_iter = 10 # Might not be larger than ns ?
    out_iter = 1000
    maxts = 7.
    mints = 0.5
    UseP = 1

    if NoiseIn is None:
        out_omp = gmca_wrap.MATRIX_OMP(nc,ns,nf,bpix,nb,in_iter,out_iter,maxts,mints,L1,UseP).GMCA_OneIter_omp(X,Ainit,Thresholds)
    else:
        out_omp = gmca_wrap.MATRIX_OMP(nc,ns,nf,bpix,nb,in_iter,out_iter,maxts,mints,L1,UseP).GMCA_OneIter_omp_TEMP(X,Ainit,Thresholds,NoiseIn)

    Abatches = out_omp[0:nc,:,:]
    MadVal = out_omp[nc,:,:]
    MaxVal = out_omp[nc+1,:,:]
    FMw = out_omp[nc+2,:,:]
    #FMw[FMw > 1e6] = 0

    if NoiseIn is None:
        FMw = np.dot(np.diag(1./(1e-12+np.sum(FMw,axis=1))),FMw)
        FMw[FMw < 1e-12] = 0
    else:
        FMw = np.dot(np.diag(1./(1e-12+np.sum(1./FMw,axis=1))),1./FMw)

    #Abatches = gmca_wrap.MATRIX_OMP(nc,ns,nf,bpix,nb,in_iter,out_iter,maxts,mints,L1,UseP).CorrectPerm_Batches(Ainit,Abatches)

    return Abatches,MadVal,MaxVal,FMw

def block_Update_A_omp(X,Ainit,Thresholds=None,Sinit=None,BlockSize=None,L1=False,FixedCol=None,FullRandom=False):

    """
        X should be of size Nobs x Npix x NBatches
    """

    import os
    import sys
    WrapLoc = os.environ['PYWRAP3_LOC']
    sys.path.insert(1,WrapLoc)
    import gmca as gmca_wrap

    ns = np.shape(Ainit)[1]
    nc = np.shape(Ainit)[0]
    if FixedCol is None:
        nf = 0
    else:
        nf = np.size(FixedCol)
    nb = np.shape(X)[2]
    bpix = np.shape(X)[1]

    in_iter = 10 # Might not be larger than ns ?
    out_iter = 1000
    maxts = 7.
    mints = 0.5
    UseP = 1
    if BlockSize is None:
        BlockSize = ns
    if Sinit is None:
        Sinit = np.zeros((ns,bpix,nb))
    if Thresholds is None:
        Thresholds = np.zeros((ns,))

    Is = np.random.permutation(Ainit.shape[1])

    if BlockSize < ns:
        if FullRandom:
            out_omp = gmca_wrap.MATRIX_OMP(nc,ns,nf,bpix,nb,in_iter,out_iter,maxts,mints,L1,UseP).bGMCA_OneIter_RandBlock_omp(X,Ainit,Sinit,Thresholds,np.array([BlockSize]).astype('double'))
        else:
            out_omp = gmca_wrap.MATRIX_OMP(nc,ns,nf,bpix,nb,in_iter,out_iter,maxts,mints,L1,UseP).bGMCA_OneIter_omp(X,Ainit,Sinit,Thresholds,np.array([BlockSize]).astype('double'),Is.astype('double'))
    else:
        out_omp = gmca_wrap.MATRIX_OMP(nc,ns,nf,bpix,nb,in_iter,out_iter,maxts,mints,L1,UseP).GMCA_OneIter_omp(X,Ainit,Thresholds)

    Abatches = out_omp[0:nc,:,:]
    MadVal = out_omp[nc,:,:]
    MaxVal = out_omp[nc+1,:,:]
    FMw = out_omp[nc+2,:,:]
    #FMw[FMw > 1e6] = 0
    FMw = np.dot(np.diag(1./(1e-12+np.sum(FMw,axis=1))),FMw)
    FMw[FMw < 1e-12] = 0

    #Abatches = gmca_wrap.MATRIX_OMP(nc,ns,nf,bpix,nb,in_iter,out_iter,maxts,mints,L1,UseP).CorrectPerm_Batches(Ainit,Abatches)

    return Abatches,MadVal,MaxVal,FMw

    bGMCA_Residual_Batches_omp

def Update_Residual_omp(X,Ainit,Sinit,Thrd=None,BlockSize=2,Ind=None):

    """
        X should be of size Nobs x Npix x NBatches
    """

    import os
    import sys
    WrapLoc = os.environ['PYWRAP3_LOC']
    sys.path.insert(1,WrapLoc)
    import gmca as gmca_wrap

    ns = np.shape(Ainit)[1]
    nc = np.shape(Ainit)[0]
    nf = 0
    nb = np.shape(X)[2]
    bpix = np.shape(X)[1]

    if Thrd is None:
        Thrd = np.zeros((ns,))

    in_iter = 10 # Might not be larger than ns ?
    out_iter = 1000
    maxts = 7.
    mints = 0.5
    UseP = 1
    L1 = False

    Xout = gmca_wrap.MATRIX_OMP(nc,ns,nf,bpix,nb,in_iter,out_iter,maxts,mints,L1,UseP).bGMCA_UpdateAS_Batches_omp(X,Ainit,Sinit,Thrd,np.array([BlockSize]).astype('double'),Ind.astype('double'))

    return Xout

def Update_S_omp(X,Ainit,Thresholds,BlockSize=None,L1=False):

    """
        X should be of size Nobs x Npix x NBatches
    """

    import os
    import sys
    WrapLoc = os.environ['PYWRAP3_LOC']
    sys.path.insert(1,WrapLoc)
    import gmca as gmca_wrap

    ns = np.shape(Ainit)[1]
    nc = np.shape(Ainit)[0]
    nf = 0
    nb = np.shape(X)[2]
    bpix = np.shape(X)[1]

    in_iter = 10 # Might not be larger than ns ?
    out_iter = 1000
    maxts = 7.
    mints = 0.5
    UseP = 1

    ### DO THAT ALSO WITH A SINGLE A !!!!!!


    Sbatches = gmca_wrap.MATRIX_OMP(nc,ns,nf,bpix,nb,in_iter,out_iter,maxts,mints,L1,UseP).GMCA_GetS_Batches_omp(X,Ainit,Thresholds)

    #Abatches = gmca_wrap.MATRIX_OMP(nc,ns,nf,bpix,nb,in_iter,out_iter,maxts,mints,L1,UseP).CorrectPerm_Batches(Ainit,Abatches)

    return np.reshape(Sbatches,(ns,-1))
