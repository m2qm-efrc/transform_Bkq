# ========================================================================================================
# References
# 1. General expressions for Stevens and Racah operator equivalents, Duros et al, 2024, arXiv:2405.08978.
# 2. Quantum theory of angular momentum, D A Varshalovich, A N Moskalev, and V K Khersonskii, 1988. 
# 3. Transformation relations for the conventional Okq and normalised Okq Stevens operator equivalents 
#    with k=1 to 6 and -k⩽q⩽k, C Rudowicz, 1985
# ========================================================================================================

import numpy as np
import scipy as sp
from scipy.special import factorial
from scipy.spatial.transform import Rotation as R
from sympy import Rational
from sympy.physics.wigner import wigner_3j
from external.StevensOperators import StevensOpA
import pyscf

# ========================================================================================================
# Rotation
# ========================================================================================================

def get_D(j, alpha, beta, gamma):
    # Find the Wigner D matrix for spherical harmonics.
    # Convention: external, zyz, gamma-first, same as https://en.wikipedia.org/wiki/Wigner_D-matrix

    D_real = pyscf.symm.Dmatrix.Dmatrix(j, alpha, beta, gamma)
    u = pyscf.symm.sph.sph_pure2real(j)
    D = u @ D_real @ np.transpose( np.conjugate(u) )

    return D

# ========================================================================================================
# Irreducible tensor operators (ITO)
# ========================================================================================================

def factorial_ldb(k):
    return np.longdouble(factorial(k, exact=True))

def get_ak(k, j, convention=""):
    """
    Get the multiplicative factor ak for the reducible tensor operator Tkq according to the Wybourne convention.
    ak = (-1)^(2j-k/2) (2j+1) ((j+k/2)!/((k/2)!(k/2)!(j-k/2)!))^(1/2) (2j-k)!/(2j+k+1)! ((2k)!)^(1/2)
    See get_Tkq, get_reduced_matrix_element, and Ref 1. 
    """
    if convention == "Racah":
        ak = (-1)**k / factorial_ldb(k) * np.sqrt(factorial_ldb(2*k) * factorial_ldb(2*j-k) / factorial_ldb(2*j+k+1))
    elif convention == "Wybourne":
        ak = (-1)**(2*j-k/2) * (2*j+1) * np.sqrt(factorial_ldb(j+k/2)/(factorial_ldb(k/2) * factorial_ldb(k/2) * factorial_ldb(j-k/2))) * factorial_ldb(2*j-k)/factorial_ldb(2*j+k+1) * np.sqrt(factorial_ldb(2*k))
    else:
        # This convention leads to sizable matrix elements of Tkq, which is suitable for numerical anaylysis. 
        # Both Racah and Wybourne conventions lead to rather small matrix elements of Tkq for big k. 
        ak = (-1)**k * np.sqrt(factorial_ldb(2*k)) / factorial_ldb(k)
    return ak

def get_reduced_matrix_element(k, j, ak):
    """
    Get the reduced matrix element <j|| Tk || j> of the reducible tensor operator Tkq.
    <j|| Tk || j> = (-1)^k ak k! ((2j+k+1)!/((2k)!(2j-k)!))^(1/2)
    """
    return (-1)**k * ak * factorial_ldb(k) * np.sqrt(factorial_ldb(int(2*j+k+1))/(factorial_ldb(2*k)*factorial_ldb(int(2*j-k))))
    
def get_an_element_of_Tkq(k, q, j, m1, m2):
    """
    <j m1 | Tkq | j m2 > = (-1)^(j-m1) 3jm(j, k, j, -m1, q, m2) <j|| Tk || j>
    3jm = wigner_3j(j1, j2, j3, m1, m2, m3)
    """
    # threejm = wigner_3j(j, k, j, -m1, q, m2)
    threejm = wigner_3j(j, k, Rational(j), -Rational(m1), q, Rational(m2)).evalf(30)
    ak = get_ak(k, j)
    Tkjj = get_reduced_matrix_element(k, j, ak)
    return (-1)**(j-m1) * threejm * Tkjj

def get_Tkq(k, q, j):
    """
    Get one component Tkq of the irreducible tensor operator Tk.
    <j m1 | Tkq | j m2 > = (-1)^(j-m1) 3jm(j, k, j, -m1, q, m2) <j|| Tk || j>
    3jm = wigner_3j(j1, j2, j3, m1, m2, m3)
    """
    Tkq = []
    ak = get_ak(k, j)
    for m1 in np.arange(-j, j+1):
        sign = (-1)**(j-m1)
        Tkq_row = []
        for m2 in np.arange(-j, j+1):
            # threejm = wigner_3j(j, k, j, -m1, q, m2)
            threejm = wigner_3j(j, k, Rational(j), -Rational(m1), q, Rational(m2)).evalf(30)
            # print(m1, m2, threejm)
            Tkjj = get_reduced_matrix_element(k, j, ak) 
            Tkq_row.append(sign * threejm * Tkjj)
        Tkq.append(Tkq_row)
    Tkq = np.array(Tkq)
    Tkq = Tkq.astype(np.float128)
    return Tkq

def get_Tk(k, j):
    """
    Get the irreducible tensor operator Tk.
    """
    Tk = []
    for q in range(-k, k+1):
        Tkq = get_Tkq(k, q, j)
        Tk.append(Tkq)
    Tk = np.array(Tk)
    return Tk

def get_Bk_ITO_by_projection(Tk, k, j, H):
    """
    Project the operator H on to Tkq, and get the coefficient Bkq as in H = sum_q Bkq Tkq. 
    Bkq = (2k+1)! / ( (k!)^2 ak^2 ) * (2j - k)! / (2j + k + 1)! Tr(Tkq_dagger H)
    """
    Bk = []
    ak = get_ak(k, j)
    for q in range(-k, k+1):
        Tkq_dagger = np.conjugate(np.transpose(Tk[k+q]))
        Bkq = factorial_ldb(2*k+1) / ( (factorial_ldb(k))**2 * ak**2 ) * factorial_ldb(int(2*j - k)) / factorial_ldb(int(2*j + k + 1)) * np.trace(Tkq_dagger @ H)
        Bk.append(Bkq)
    Bk = np.array(Bk)
    return Bk

# ========================================================================================================
# From ITO to ESO. ITO: irreducible tensor operators. ESO: extended Stevens operators.
# ========================================================================================================

def get_A(k, j):
    """
    Find the matrix A that transforms the irreducible tensor operator Tkq into the Stevens operators Okq. 
    Okq = sum_p Tkp Apq
    A is invariant against reference frame rotation, as shown by check_rotatedBkq_ESO. 
    """
    Tk = get_Tk(k, j)
    A = []
    for q in range(-k, k+1):
        Okq = StevensOpA(j, k, q, np.eye(3))
        Aq = get_Bk_ITO_by_projection(Tk, k, j, Okq)
        A.append(Aq)
    A = np.transpose(A)
    return A

# ========================================================================================================
# Check various relations
# ========================================================================================================

def check_rotatedTk(k, j, q):
    alpha = 2*np.pi*np.random.rand()
    beta  =   np.pi*np.random.rand()
    gamma = 2*np.pi*np.random.rand()
    D = get_D(j, alpha, beta, gamma)
    D_dagger = np.conjugate(np.transpose(D))

    Tk = get_Tk(k, j)
    Tkq_new_1 = D @ Tk[k+q] @ D_dagger

    D = get_D(k, alpha, beta, gamma)
    Tk_new = np.einsum('pq,pmn->qmn', D, Tk)
    Tkq_new_2 = Tk_new[k+q]

    if np.all(np.abs(Tkq_new_1 - Tkq_new_2)<1e-9):
        print("Passed")

    return

def check_rotatedBk_ITO(k, j):
    # sum_q Bkq Tkq = sum_pq Bkq_new Tkp Dpq
    #               = sum_p (sum_q Bkq_new Dpq) Tkp
    #               = sum_q (sum_p Bkp_new Dqp) Tkq
    # Bkq = Dqp Bkp_new or
    # Bk = D Bk_new or
    # Bk_new = D_dagger Bk
    
    alpha = 2*np.pi*np.random.rand()
    beta  =   np.pi*np.random.rand()
    gamma = 2*np.pi*np.random.rand()

    D = get_D(k, alpha, beta, gamma)
    D_dagger = np.conjugate(np.transpose(D))
    Tk = get_Tk(k, j)
    Tk_new = np.einsum('pq,pmn->qmn', D, Tk)

    Bk = np.random.rand(2*k+1)
    Bk_new = D_dagger @ Bk

    BkTk = np.einsum('q,qmn->mn', Bk, Tk)
    BkTk_new = np.einsum('q,qmn->mn', Bk_new, Tk_new)

    if np.all(np.abs(BkTk_new - BkTk)<1e-9):
        print("Passed")

    return

def check_projectedBk_ITO(j, H):
    # Decompose H as H = sum_kq Bkq Tkq
    dict_Bk = dict()
    dict_Tk = dict()
    for k in range(0, 2*j+1, 2):
        Tk = get_Tk(k, j)
        Bk = get_Bk_ITO_by_projection(Tk, k, j, H)
        dict_Tk[k] = Tk
        dict_Bk[k] = Bk
        print("k = ", k, ", Bk = ", Bk)

    # Assemble H by H = sum_kq Bkq Tkq
    H_new = np.zeros((2*j+1, 2*j+1))
    for k in range(0, 2*j+1, 2):
        H_part = np.einsum('imn,i->mn', dict_Tk[k], dict_Bk[k])
        H_new = H_new + H_part

    # Check if H == sum_kq Bkq Tkq
    x = np.abs(H_new - H)/np.abs(H)
    x = np.nan_to_num(x, nan=0, posinf=0)
    if np.all(np.nanmax(x) < 1e-6):
        print("Passed")
    else:
        print("Failed")
        print("max(abs(H_new - H)) = ", np.max(np.abs(H_new - H)))
        print("nanmax( abs(H_new - H)/abs(H) ) = ", np.nanmax(x) )

    return

def check_rotatedBkq_ESO(k, j):
    """
    Check if Bkq Okq = Bkq_new Okq_new.
    Bkq_new and Okq_new are for the rotated reference frame. 
    """

    # Construct the Hamitonian using a random set of Bkq in the initial reference frame
    Bk = np.random.rand(2*k+1)
    H = np.zeros((2*j+1, 2*j+1), dtype=np.complex128)
    for q in range(-k, k+1):
        Okq = StevensOpA(j, k, q, np.eye(3))
        H = H + Bk[k+q]*Okq

    # Transform Bkq using a random rotation
    alpha = 2*np.pi*np.random.rand()
    beta  =   np.pi*np.random.rand()
    gamma = 2*np.pi*np.random.rand()

    D = get_D(k, alpha, beta, gamma)
    D_dagger = np.conjugate(np.transpose(D))

    A = get_A(k, j)
    # A_inv = np.linalg.inv(A) # does not support complex256
    A_inv = sp.linalg.inv(A)

    Bk_new = np.einsum('qp,pr,rs,s->q', A_inv, D_dagger, A, Bk)

    # Construct the Hamitonian in the rotated reference frame
    r = R.from_euler('zyz', [gamma, beta, alpha])
    rotmat = r.as_matrix()
    emat_T = np.eye(3)
    emat_new = np.transpose( rotmat @ emat_T )
    H_new = np.zeros((2*j+1, 2*j+1), dtype=np.complex128)
    for q in range(-k, k+1):
        Okq = StevensOpA(j, k, q, emat_new)
        H_new = H_new + Bk_new[k+q]*Okq

    # Check if H == H_new
    x = np.abs(H_new - H)/np.abs(H)
    x = np.nan_to_num(x, nan=0, posinf=0)
    if np.all(np.nanmax(x) < 1e-6):
        print("Passed")
    else:
        print("Failed")
        print("max(abs(H_new - H)) = ", np.max(np.abs(H_new - H)))
        print("nanmax( abs(H_new - H)/abs(H) ) = ", np.nanmax(x) )
    
    return



def test():

    #k=8; j=6
    #ak = get_ak(k, j)
    #print(ak)
    
    #k=2; q=-2; j=2; m1=0; m2=2
    #get_an_element_of_Tkq(k, q, j, m1, m2)
    
    #k=4; q=-2; j=2
    #Tkq = get_Tkq(k, q, j)
    #print(Tkq)
    
    #k=4; j=2
    #Tk = get_Tk(k, j)
    #print(Tk)
    
    #k=4; q=1; j=2
    #check_rotatedTk(k, j, q)
    
    #k=4; j=6
    #check_rotatedBk_ITO(k, j)
    
    # k=2; j=2
    # Tk = get_Tk(k, j)
    # H = np.random.rand(2*j+1, 2*j+1)
    # H = np.transpose(H) + H
    # Bk = get_Bk_ITO_by_projection(Tk, k, j, H)
    # print(Bk)
    # print(Tk)
    
    ### Cannot decompose a random Hermitian matrix, ???
    #j=2
    #H = np.random.rand(2*j+1, 2*j+1)
    #H = np.transpose(H) + H
    #check_projectedBk_ITO(Tk, j, H)
    
    ### Can decompose a matrix which is a superposition of Tkq
    #j=2
    #H = get_Tkq(2, 0, j) + get_Tkq(4, 2, j)
    #check_projectedBk_ITO(j, H)

    #j=8; k=12; q=2
    #Okq = StevensOpA(j, k, q, np.eye(3))
    #check_projectedBk_ITO(j, Okq)

    #k=2; j=2.5
    #A = get_A(k, j)
    #print(A)

    #k=12; j=6; check_rotatedBkq_ESO(k, j)

    return

if __name__ == "__main__":

    test()

