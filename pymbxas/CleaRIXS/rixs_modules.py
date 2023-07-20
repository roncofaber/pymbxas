import sys
import numpy as sp
from scipy import linalg as la
################################
### nocc = number of electrons
### norb = number of orbitals in core-excited state
### chb_xmat = transition matrix in CHB
### gb_xmat = transition matrix in GB
def AbsCalc(ixyz,xi,nocc,norb,chb_xmat):
    AbsTrAmp = []
    AMat = xi[0:nocc,0:nocc]
    AMat = sp.matrix(AMat)
    AInvMat = la.inv(AMat)
    ADet = la.det(AMat)
    APrimeMat = xi[nocc:,0:nocc]
    KMat = sp.matrix(APrimeMat) * AInvMat
    for f in range(nocc,norb): #Loop over FCH state empty orbitals
     CUMUL_CONTRIB = 0
     for h in range(nocc): #Loop over FCH state occupied orbitals
      MB_OVLP = KMat[(f-nocc),h] * ADet
      CH_CONTRIB = (chb_xmat[h]).conjugate() * MB_OVLP
      CUMUL_CONTRIB = CUMUL_CONTRIB - CH_CONTRIB
     FCH_CONTRIB = (chb_xmat[f]).conjugate() * ADet
     CUMUL_CONTRIB = CUMUL_CONTRIB + FCH_CONTRIB
     #AbsTrAmp.append([(ener[f-nocc]/27.2114),(CUMUL_CONTRIB)**2])
     AbsTrAmp.append(CUMUL_CONTRIB)
    return AbsTrAmp

def abs_spec(Absorption, inp_freq, ener, Gamma, GS_tot_en, File):
    broadnd_spec = 0
    Gamma=sp.imag(Gamma)
    #print("lens = "+str([len(Absorption) , len(ener)]))
    ener = ener-GS_tot_en
    #print("Printing ener : "+str(ener))
    #print("Printing inp_freq : "+str(inp_freq))
    for i in range(len(ener)):
     broadnd_spec += abs(Absorption[i])**2 * sp.exp(-( inp_freq - ener[i] )**2/(2*Gamma**2))
    sp.savetxt(File, sp.array([inp_freq , broadnd_spec]).T,fmt="%9.6f")
    return(broadnd_spec)

#RefMats Computes the k-independent reference matrices and determinants
def RefMats(ixyz,n,m,xi,gb_xmat):
    xi_c = sp.matrix(xi[:, n:m]) * sp.matrix(gb_xmat[n:m]).T
    #print("xi")
    #print(xi)
    #print("xi_c")
    #print(xi_c)
    absp_xi_c_ = sp.concatenate((xi[:, 0 : n], xi_c), axis = 1) #(M X N+1) matrix
    #print("absp_xi_c_")
    #print(absp_xi_c_)
    #
    emm_xi = sp.matrix(xi).T   
    emm_xi_c_ = sp.matrix(absp_xi_c_).T #(N+1 X M) matrix
    #
    Dmy1 = xi[0:n+1 , 0:n+1] # (N+1 X N+1) matrix
    Dmy2 = xi[n:m , 0:n+1] # (M-N X N+1) matrix
    eta_mat = sp.matrix(Dmy2) * (la.inv(Dmy1))
    Dmy1_DET = la.det(Dmy1)
    #
    absp_ref_mat = sp.matrix(absp_xi_c_[ : n+1, : n+1]) # Reference matrix for absorption in GB
    #print("absp_ref_mat")
    #print(absp_ref_mat)
    absp_ref_det = la.det(absp_ref_mat)
    absp_ref_inv = la.inv(absp_ref_mat)
    #print("sp.matrix(absp_xi_c_[n : m, :])")
    #print(sp.matrix(absp_xi_c_[n : m, :]))
    #print("absp_ref_inv")
    #print(absp_ref_inv)
    zeta = sp.matrix(absp_xi_c_[n : m, :]) * absp_ref_inv #This is the zeta matrix for absorption
    return emm_xi,emm_xi_c_,eta_mat,zeta,Dmy1_DET,absp_ref_det

#Bk_1MAT is from Eq. 18 and Eq. 19 of PRB 106, 115115 (2022)
#Bk_refMAT is from Eq. 18 and Eq. 19 of PRB 106, 115115 (2022)
#Ck_refMAT is from Eq. 22 of PRB 106, 115115 (2022) 
#Bk_refDET is the determinant of Bk_refMAT
#Ck_refDET is the determinant of Ck_refMAT

#K_DepRefMats computes the k-dependent determinants
def K_DepDets(k,ixyz,n,m,emm_xi,emm_xi_c_,eta_mat,zeta,Dmy1_DET,absp_ref_det):
    Bk_1MAT = sp.concatenate((emm_xi[n:m , 0:n] , emm_xi[n:m , k:k+1]) , axis = 1) # (M-N X N+1) 
    Bk_refMAT = sp.concatenate((emm_xi_c_[:,0:n] , emm_xi_c_[: , k:k+1]) , axis = 1) # (N+1 X N+1)
    Ck_refMAT = sp.concatenate((emm_xi[0:n+1 , 0:n] , emm_xi[0:n+1 , k:k+1]) , axis = 1) # (N+1 X N+1)
    #
    ##### Sherman-Morrison for updating inverse of Ck_refMAT and Bk_refMAT here.
    #
    #
    kappa_mat = sp.matrix(Bk_1MAT) * (la.inv(Ck_refMAT)) #Calculating the k-dependent (M-N X N+1) dimensional \kappa matrix Eq. 24
    gamma_mat = sp.matrix(Bk_1MAT) * (la.inv(Bk_refMAT)) #Calculating the (M-N X N+1) \gamma matrix Eq. 19
    #
    Ck_refDET = eta_mat[k-n , -1] * Dmy1_DET
    Bk_refDET = zeta[k-n , -1] * absp_ref_det
    D2_det = (-1) * kappa_mat[:,-1] * Ck_refDET #For a given k>N, it's (M-N X 1) array of the determinants D2 from Eq. 21. Each entry corresponds to an \alpha>N.
    D1_det = gamma_mat[: , 0:n] * Bk_refDET #For a given k>N, it's (M-N X N) array of the determinants D1 from Eq. 16. The entries are for different \alpha>N and \beta<(N+1)
    return D1_det,D2_det

def BruteForce(nocc,norb,k,alpha,beta,xi,gb_xmat):
    XiLeft = sp.concatenate((xi[0:nocc,0:nocc],xi[k:k+1,0:nocc]),axis = 0)
    xi_c = sp.matrix(xi[:, nocc:norb]) * sp.matrix(gb_xmat[nocc:norb]).T # (MX1) matrix
    AlphaColumnToAdd = sp.concatenate((xi[:nocc,alpha:alpha+1], sp.array([xi[k,alpha:alpha+1]])),axis = 0)
    AlphaColumnToAdd = sp.reshape(AlphaColumnToAdd,(nocc+1)) 
    XiLeft[:,beta]=AlphaColumnToAdd #Replacing the \beta-th column by the \alpha-th column
    BetaTermToAdd = gb_xmat[beta]*xi[:,beta]
    AlphaTermToSubtract = gb_xmat[alpha] * xi[:,alpha]
    xi_c = xi_c + sp.matrix(BetaTermToAdd).T - sp.matrix(AlphaTermToSubtract).T
    Xi_c_ToAdd = sp.concatenate((xi_c[0:nocc],xi_c[k:k+1]), axis = 0) # (N+1 X 1) array
    Yk_alpha_betaMAT = sp.concatenate((XiLeft,Xi_c_ToAdd), axis = 1)
    return Yk_alpha_betaMAT

def EmmCalc(gb_xmat,D1_det,D2_det,f_state,A_f_vc):
    omega_D2 = la.kron(sp.array(gb_xmat[0:n]) , D2_det)
    Q_mat = D1_det + omega_D2 #Dipole Matrix corresponding to emission. This is (Q^f_{\alpha,\beta})^*
    contrib = 0
    for transition in (A_f_vc[f_state]): #loop over transitions constituting the excited state f_state. This is essentially a loop over alpha and beta
     q_value = Q_mat [int(transition[1]-n-1) , int(transition[0]-1)]
     contrib += q_value * complex(transition[2] , (-1 * transition[3]))
    return contrib

def SuggestWin(GS_tot_en, ener, lowE, highE, gridP):
    BottomEn = ener[0] - GS_tot_en - lowE
    TopEn = max(ener) - GS_tot_en - highE
    print("GS_tot_en, BottomEn, TopEn = " +str(GS_tot_en)+" , "+str(BottomEn)+ " , "+str(TopEn))
    gap = (TopEn-BottomEn)/gridP
    w_in = sp.arange(BottomEn, TopEn, gap)
    return(w_in)

def DebugRIXS(full_gs_matrix, inp_freq, CI_Expansion, nocc, norb, ener, GS_tot_en, Absorption,Gamma,core_ind_gs, xi):
    debug_gb_xmat=sp.array(full_gs_matrix)[0,:]
    for debug_win in inp_freq:
     for F in range(len(CI_Expansion)):
      DebugRIXS = 0
      for k in range(nocc,norb):
       DebugAbsorption = Absorption[0][k-nocc]
       DebugDenom = 1/(debug_win+GS_tot_en+Gamma - ener[k-nocc])
       print("DEBUGRIXS: k = "+str(k)+" DebugAbsorption = "+str(DebugAbsorption)+" DebugDenom = "+str(DebugDenom))
       DebugEmission = 0
       for single in CI_Expansion[F][2]:
        if (int(single[0]) < core_ind_gs):
         V_orb = int(single[0])-1
        if (int(single[0]) > core_ind_gs):
         V_orb = int(single[0])-2
        if (int(single[0]) == core_ind_gs):
         continue
        C_orb = int(single[1])-1
        single_contrib = float(single[2]) 
        alpha = C_orb + nocc
        beta = V_orb
        DETy= la.det(BruteForce(nocc,norb,k,alpha,beta,xi,debug_gb_xmat))
        DebugContrib = single_contrib * DETy
        DebugEmission += DebugContrib
        print("DEBUGRIXS: F, k, V_orb, C_orb, single_contrib, DETy, DebugContrib = "+str([F, k, V_orb, C_orb, single_contrib, DETy, DebugContrib]))
       print("DEBUGRIXS: for F, k = "+str([F,k])+" DebugAbsorption= "+str(DebugAbsorption)+" DebugEmission= "+str(DebugEmission)+" DebugDenom= "+str(DebugDenom))
       DebugRIXS += DebugAbsorption * DebugEmission * DebugDenom 
      print("For w_in = "+str(debug_win)+" and F = "+str(F)+"DebugRIXS= "+str(DebugRIXS))

def local_bool(arr):
    Bool_arr =((arr >= sp.roll(arr,  1, 0)) &
              (arr >= sp.roll(arr, -1, 0)) &
              (arr >= sp.roll(arr,  1, 1)) &
              (arr >= sp.roll(arr, -1, 1)))
    return(Bool_arr)

def FindOrbs(E_f, ener, GS_tot_en, in_w, out_w, thres_orb):
    ener = ener - GS_tot_en
    Xorb = sp.where(abs(ener - in_w) < thres_orb)
    Yorb = sp.where(abs(in_w - E_f - out_w) < thres_orb)
    return Xorb, Yorb
     
