# =================================================================================================
# References
# 1. General expressions for Stevens and Racah operator equivalents, Duros et al, 2024, arXiv:2405.08978.
# 2. Quantum theory of angular momentum, D A Varshalovich, A N Moskalev, and V K Khersonskii, 1988. 
# 3. Transformation relations for the conventional Okq and normalised Okq Stevens operator equivalents 
#    with k=1 to 6 and -k⩽q⩽k, C Rudowicz, 1985
# =================================================================================================

import numpy as np
import scipy as sp
from scipy.special import factorial
from scipy.spatial.transform import Rotation as R
from sympy import Rational
from sympy.physics.wigner import wigner_3j, wigner_d
from StevensOperators import StevensOpA
# import pyscf # Only for the Wigner D matrix during testing

# =================================================================================================
# Irreducible tensor operators (ITO)
# =================================================================================================

def factorial_ldb(k):
    return np.longdouble(factorial(k, exact=True))

def get_ak(k, j, convention="spherical"):
    """
    Get the multiplicative factor ak for the reducible tensor operator Tkq according to the Wybourne convention.
    ak = (-1)^(2j-k/2) (2j+1) ((j+k/2)!/((k/2)!(k/2)!(j-k/2)!))^(1/2) (2j-k)!/(2j+k+1)! ((2k)!)^(1/2)
    See get_Tkq, get_reduced_matrix_element, and Ref 1. 
    """
    if convention == "Racah":
        ak = (-1)**k / factorial_ldb(k) * np.sqrt(factorial_ldb(2*k) * factorial_ldb(2*j-k) / factorial_ldb(2*j+k+1))
    elif convention == "Wybourne":
        ak = (-1)**(2*j-k/2) * (2*j+1) * np.sqrt(factorial_ldb(j+k/2)/(factorial_ldb(k/2) * factorial_ldb(k/2) * factorial_ldb(j-k/2))) * factorial_ldb(2*j-k)/factorial_ldb(2*j+k+1) * np.sqrt(factorial_ldb(2*k))
    elif convention == "noname":
        ak = (-1)**k * 2**(-k/2)
    elif convention == "spherical":
        # This convention leads to sizable matrix elements of Tkq, which is suitable for numerical anaylysis. 
        # Both Racah and Wybourne conventions lead to rather small matrix elements of Tkq for big k. 
        ak = (-1)**k * np.sqrt(factorial_ldb(2*k)) / factorial_ldb(k)
    else:
        # Same as spherical
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

def get_Tkq(k, q, j, convention="spherical"):
    """
    Get one component Tkq of the irreducible tensor operator Tk.
    <j m1 | Tkq | j m2 > = (-1)^(j-m1) 3jm(j, k, j, -m1, q, m2) <j|| Tk || j>
    3jm = wigner_3j(j1, j2, j3, m1, m2, m3)
    """
    Tkq = []
    ak = get_ak(k, j, convention=convention)
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
    Tkq = Tkq.astype(np.float64)
    return Tkq

def get_Tk(k, j, convention="spherical"):
    """
    Get the irreducible tensor operator Tk.
    """
    Tk = []
    for q in range(-k, k+1):
        Tkq = get_Tkq(k, q, j, convention=convention)
        Tk.append(Tkq)
    Tk = np.array(Tk)
    return Tk

def get_Bk_ITO_by_projection(Tk, k, j, H, convention="spherical"):
    """
    Project the operator H on to Tkq, and get the coefficient Bkq as in H = sum_q Bkq Tkq. 
    Bkq = (2k+1)! / ( (k!)^2 ak^2 ) * (2j - k)! / (2j + k + 1)! Tr(Tkq_dagger H)
    """
    Bk = []
    ak = get_ak(k, j, convention=convention)
    for q in range(-k, k+1):
        Tkq_dagger = np.conjugate(np.transpose(Tk[k+q]))
        Bkq = factorial_ldb(2*k+1) / ( (factorial_ldb(k))**2 * ak**2 ) * factorial_ldb(int(2*j - k)) / factorial_ldb(int(2*j + k + 1)) * np.trace(Tkq_dagger @ H)
        Bk.append(Bkq)
    Bk = np.array(Bk)
    return Bk

# =================================================================================================
# From ITO to ESO. ITO: irreducible tensor operators. ESO: extended Stevens operators.
# =================================================================================================

def get_A(k, j, convention="spherical"):
    """
    Find the matrix A that transforms the irreducible tensor operator Tkq into the Stevens operators Okq. 
    Okq = sum_p Tkp Apq
    A is invariant against reference frame rotation, as shown by check_rotatedBkq_ESO. 
    """
    Tk = get_Tk(k, j, convention=convention)
    A = []
    for q in range(-k, k+1):
        Okq = StevensOpA(j, k, q, np.eye(3))
        Aq = get_Bk_ITO_by_projection(Tk, k, j, Okq, convention=convention)
        A.append(Aq)
    A = np.transpose(A)
    return A

# =================================================================================================
# Check various relations
# =================================================================================================

def check_rotatedTk(k, j, q):
    alpha = 2*np.pi*np.random.rand()
    beta  =   np.pi*np.random.rand()
    gamma = 2*np.pi*np.random.rand()
    D = get_D(j, alpha, beta, gamma)
    D_dagger = np.conjugate(np.transpose(D))

    Tk = get_Tk(k, j, convention="spherical")
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
    Tk = get_Tk(k, j, convention="spherical")
    Tk_new = np.einsum('pq,pmn->qmn', D, Tk)

    Bk = np.random.rand(2*k+1)
    Bk_new = D_dagger @ Bk

    BkTk = np.einsum('q,qmn->mn', Bk, Tk)
    BkTk_new = np.einsum('q,qmn->mn', Bk_new, Tk_new)

    if np.all(np.abs(BkTk_new - BkTk)<1e-9):
        print("Passed")

    return

def check_Bqks(k, j, convention="spherical"):
    # Check if sum_q Bqk Tkq == sum_q Bkq Okq.
    
    fin = "Bkqs.dat"

    Bkqs = read_Bkqs(fin)
    unique_ks, Bk_dict_ESO = get_Bk_dict(Bkqs)

    Bqks = transform_Bkqs_to_Bqks(fin, j, convention=convention)
    _, Bk_dict_ITO = get_Bk_dict(Bqks)

    Ok = []
    for q in range(-k, k+1):
        Okq = StevensOpA(j, k, q, np.eye(3))
        Ok.append(Okq)
    BkOk = np.einsum('q,qmn->mn', Bk_dict_ESO[k], Ok)

    Tk = get_Tk(k, j, convention=convention)
    BkTk = np.einsum('q,qmn->mn', Bk_dict_ITO[k], Tk)

    # print(BkOk[0])
    # print(BkTk[0])

    tol = 1e-12
    if np.allclose(np.real(BkOk), np.real(BkTk), atol=tol) and np.allclose(np.imag(BkOk), np.imag(BkTk), atol=tol):
        print("Passed")

    return

def check_projectedBk_ITO(j, H):
    # Decompose H as H = sum_kq Bkq Tkq
    dict_Bk = dict()
    dict_Tk = dict()
    for k in range(0, 2*j+1, 2):
        Tk = get_Tk(k, j, convention="spherical")
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

# =================================================================================================
# Read Bkq
# =================================================================================================

def read_Bkqs(fname):
    """
    Format of Bkqs.dat: k, q, Bkq
    """
    Bkqs = np.loadtxt(fname)
    return Bkqs

def read_Bqks(fname):
    """
    Format of Bqks.dat: k, q, Bkq
    """
    Bqks = np.loadtxt(fname, dtype=np.complex128)
    return Bqks

def get_Bk_dict(Bkqs):

    n = len(Bkqs)

    ks = [int(np.real(Bkqs[i][0])) for i in range(n)]
    qs = [int(np.real(Bkqs[i][1])) for i in range(n)]

    unique_ks = set( list( ks ) )

    Bkqs_dict = dict({})
    for i in range(n):
        Bkqs_dict[(ks[i], qs[i])] = Bkqs[i][2]
    kq_keys = Bkqs_dict.keys()

    Bk_dict = dict({})
    for k in unique_ks:
        Bk_dict[k] = []
        for q in range(-k, k+1):
            if (k, q) in kq_keys:
                Bk_dict[k].append(Bkqs_dict[(k, q)])
            else:
                Bk_dict[k].append(0.0)

    return ( unique_ks, Bk_dict )


# =================================================================================================
# Rotation
# =================================================================================================

def get_D(j, alpha, beta, gamma):
    """
    Find the Wigner D matrix for spherical harmonics.
    Convention: external, zyz, gamma-first, same as https://en.wikipedia.org/wiki/Wigner_D-matrix
    """

    # Option 1: Using the sympy package
    D = wigner_d(j, alpha, beta, gamma)
    D = np.array(D, dtype=np.complex128)

    # Option 2: Using the pyscf package
    # if False:
    #     D_real = pyscf.symm.Dmatrix.Dmatrix(j, alpha, beta, gamma)
    #     u = pyscf.symm.sph.sph_pure2real(j)
    #     D_ = u @ D_real @ np.transpose( np.conjugate(u) )
    #     
    #     # Check if the two options give the same result, which is not always the case.
    #     print( np.allclose(np.real(D), np.real(D_)) )
    #     print( np.allclose(np.imag(D), np.imag(D_)) )

    return D

def transform_Bk_for_one_k(k, j, Bk, alpha, beta, gamma):
    """
    R(alpha, beta, gamma): extrinsic, gamma-first, zyz.
    R(alpha, beta, gamma) rotates the initial reference frame to the target reference frame.
    Unit for angles: deg
    Bk is a vector [Bk(q), q = -k, ..., k]
    k: rank of the crystal field parameters Bk
    j: total angular momentum
    """

    # Convert deg to rad.
    alpha = np.deg2rad(alpha); beta = np.deg2rad(beta); gamma = np.deg2rad(gamma)

    # Rotate Bkq
    D = get_D(k, alpha, beta, gamma)
    D_dagger = np.conjugate(np.transpose(D))

    A = get_A(k, j)
    A_inv = sp.linalg.inv(A)

    Bk_new = np.einsum('qp,pr,rs,s->q', A_inv, D_dagger, A, Bk)
    Bk_new = np.real(Bk_new)

    return Bk_new

def transform_Bk_for_all_ks(unique_ks, Bk_dict, j, alpha, beta, gamma):
    Bkqs = []
    for k in unique_ks:
        Bk = Bk_dict[k]
        Bk_new = transform_Bk_for_one_k(k, j, Bk, alpha, beta, gamma)
        for q in range(-k, k+1):
            Bkqs.append([k, q, Bk_new[k+q]])
    return Bkqs

def transform_Bk_using_emats(unique_ks, Bk_dict, j, emat_in, emat_out):

    # Fist, find the rotation matrix that rotates emat_in to emat_out
    # emat_in emat_in^T = emat_out emat_out^T
    # emat_out^T emat_in emat_in^T = emat_out^T
    # R emat_in^T = emat_out^T, R = emat_out^T emat_in
    rotmat = np.transpose(emat_out) @ emat_in

    # Second, find the corresponding Euler angles 
    r = R.from_matrix(rotmat)
    gamma, beta, alpha = r.as_euler('zyz', degrees=True) # gamma is to be applied first

    # Third, tranform Bkqs using the Euler anles
    Bkqs = transform_Bk_for_all_ks(unique_ks, Bk_dict, j, alpha, beta, gamma)

    return Bkqs

def save_Bkqs(fname, Bkqs):
    with open(fname, "w") as f:
        for i in range(len(Bkqs)):
            f.write("{:3d} {:3d} {:18.10E} {:8.4f}\n".format(*Bkqs[i], Bkqs[i][2]))
    return

def save_Bqks(fname, Bkqs):
    with open(fname, "w") as f:
        for i in range(len(Bkqs)):
            f.write("{:3d} {:3d} {:36.10E} {:16.4f}\n".format(*Bkqs[i], Bkqs[i][2]))
    return

def transform_Bkqs_euler(fin, fout, j, alpha, beta, gamma):
    """
    fin: input file that contains the original Bkqs
    fout: output file that contains the transformed Bkqs
    j: total angular momentum
    R(alpha, beta, gamma): extrinsic, gamma-first, zyz.
    R(alpha, beta, gamma) rotates the initial reference frame to the target reference frame.
    Unit for angles: deg
    """
    Bkqs = read_Bkqs(fin)
    unique_ks, Bk_dict = get_Bk_dict(Bkqs)
    Bkqs = transform_Bk_for_all_ks(unique_ks, Bk_dict, j, alpha, beta, gamma)
    save_Bkqs(fout, Bkqs)
    return

def transform_Bkqs_emat(fin, fout, j, emat_in, emat_out):
    """
    fin: input file that contains the original Bkqs
    fout: output file that contains the transformed Bkqs
    j: total angular momentum
    """
    Bkqs = read_Bkqs(fin)
    unique_ks, Bk_dict = get_Bk_dict(Bkqs)
    Bkqs = transform_Bk_using_emats(unique_ks, Bk_dict, j, emat_in, emat_out)
    save_Bkqs(fout, Bkqs)
    return

def find_rotated_reference_frame(emat_in, alpha, beta, gamma):
    # Rotate the reference frame emat_in by Rz(alpha) Ry(beta) Rz(gamma)

    # Find the rotation matrix
    r = R.from_euler('zyz', [gamma, beta, alpha], degrees=True)
    rotmat = r.as_matrix()

    emat_out = np.transpose( rotmat @ np.transpose(emat_in) )

    print_emat_for_spin_model(emat_out)

    return emat_out

def read_and_print_Bkqs(fin):
    Bkqs = read_Bkqs(fin)
    print_Bkqs_for_spin_model(Bkqs)
    return

def trransform_Bkqs(fin, fout, j, use_emat=False, euler_angles=(0,0,0), emat_in=np.eye(3), emat_out=np.eye(3)):
    """
    Transform the crystal field parameters Bkqs using Euler angles or the basis vectors of the initial and target reference frames.

    fin: input file that contains the original Bkqs
    fout: output file that contains the transformed Bkqs
    j: total angular momentum
    euler_angles = (alpha, beta, gamma)
        Euler angles from the inital reference frame to the target reference frame
        Convention: extrinsic, zyz, gamma-first.
        Unit: deg.
    emat_in = [ex, ey, ez]. The basis vectors for the inital reference frame. Each row is a basis vector.
    emat_out = [ex, ey, ez]. The basis vectors for the target reference frame. Each row is a basis vector.

    Default: Use Euler angles. If use_emat is True, use the basis vectors of the initial and target reference frames.
    """

    if use_emat:
        # =================================================================================================
        # Transformation using the basis vectors of the initial and target reference frames
        # =================================================================================================

        # print("Transforming Bkqs using the basis vectors of the initial and target reference frames.")

        # Transform Bkqs using the basis vectors of the initial and target reference frames
        transform_Bkqs_emat(fin, fout, j, emat_in, emat_out)

    else:
        # =================================================================================================
        # Transformation using Euler angles
        # =================================================================================================

        # print("Transforming Bkqs using Euler angles.")

        # Euler angles from the inital reference frame to the target reference frame. Convention: extrinsic, zyz, gamma-first. Unit: deg
        alpha, beta, gamma = euler_angles
        
        # Transform Bkqs using the Euler angles
        transform_Bkqs_euler(fin, fout, j, alpha, beta, gamma)

    print("The new crystal field parameters are saved in ", fout)

    return

# =================================================================================================
# Print to the screen
# =================================================================================================

def print_emat_for_spin_model(emat):
    print( ( "reference_frame: [ " + 8*"{:13.10f}, " + "{:13.10f} ]" ).format(*emat.flatten()) )
    return

def print_Bkqs_for_spin_model(Bkqs):
    n = len( Bkqs )
    print( ( "ks: [ " + (n-1)*"{:3d}, " + "{:3d} ]" ).format(*Bkqs[:, 0].astype(int)) )
    print( ( "qs: [ " + (n-1)*"{:3d}, " + "{:3d} ]" ).format(*Bkqs[:, 1].astype(int)) )
    print( ( "Bkqs: [ " + (n-1)*"{:18.10E}, " + "{:18.10E} ]" ).format(*Bkqs[:, 2]) )
    return


# =================================================================================================
# Transform the crystal field parameters Bkqs for ESO
# from/into the crystal field parameters Bqks for ITO
# =================================================================================================

def transform_Bkqs_to_Bqks(fin, j, convention="spherical"):
    """
    Bkqs: [k, q, Bkq] for ESO
    Bqks: [k, q, Bqk] for ITO
    """
    # Read Bkqs
    Bkqs = read_Bkqs(fin)

    # Group the crystal field parameters Bkq according to the order k
    unique_ks, Bk_dict = get_Bk_dict(Bkqs)

    # Transform the crystal field parameters Bkqs to Bqks
    Bqks = []
    for k in unique_ks:
        Bk_ESO = Bk_dict[k]
        A = get_A(k, j, convention=convention)
        Bk_ITO = A @ Bk_ESO
        for q in range(-k, k+1):
            Bqks.append([k, q, Bk_ITO[k+q]])

    # Save Bqks in a file
    save_Bqks("Bqks_{:s}.dat".format(convention), Bqks)

    print("The crystal field parameters Bqks are saved in Bqks_{:s}.dat".format(convention))

    return Bqks

def transform_Bqks_to_Bkqs(fin, j, convention="spherical"):
    """
    Bkqs: [k, q, Bkq] for ESO
    Bqks: [k, q, Bqk] for ITO
    """
    # Read Bqks
    Bqks = read_Bqks(fin)

    # Group the crystal field parameters Bkq according to the order k
    unique_ks, Bk_dict = get_Bk_dict(Bqks)

    # Transform the crystal field parameters Bkqs to Bqks
    Bkqs = []
    for k in unique_ks:
        Bk_ITO = Bk_dict[k]
        A = get_A(k, j, convention=convention)
        A_inv = sp.linalg.inv(A)
        Bk_ESO = A_inv @ Bk_ITO
        Bk_ESO = np.real(Bk_ESO)
        for q in range(-k, k+1):
            Bkqs.append([k, q, Bk_ESO[k+q]])

    # Save Bkqs in a file
    save_Bkqs("Bkqs_ITO2ESO.dat", Bkqs)

    print("The crystal field parameters Bkqs are saved in Bkqs_ITO2ESO.dat")

    return Bqks


# =================================================================================================
# Test
# =================================================================================================

if __name__ == "__main__":

    #k=8; j=6
    #ak = get_ak(k, j)
    #print(ak)
    
    #k=2; q=-2; j=2; m1=0; m2=2
    #get_an_element_of_Tkq(k, q, j, m1, m2)
    
    #k=4; q=-2; j=2
    #Tkq = get_Tkq(k, q, j, convention="spherical")
    #print(Tkq)
    
    #k=4; j=2
    #Tk = get_Tk(k, j, convention="spherical")
    #print(Tk)

    # Test the Wigner D matrix
    # j = 8
    # alpha = 2*np.pi*np.random.rand()
    # beta  =   np.pi*np.random.rand()
    # gamma = 2*np.pi*np.random.rand()
    # get_D(j, alpha, beta, gamma)

    #k=4; q=1; j=2
    #check_rotatedTk(k, j, q)
    
    #k=4; j=6
    #check_rotatedBk_ITO(k, j)
    
    # k=2; j=2
    # Tk = get_Tk(k, j, convention="spherical")
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

    # for i in range(2, 13, 2):
    #     fin = "Bkqs_new_" + str(i) + ".dat"
    #     print("Up to k = ", i, "\n")
    #     read_and_print_Bkqs(fin)
    #     print("\n")

    # k = 8; j = 8
    # check_Bqks(k, j, convention="spherical")
    # check_Bqks(k, j, convention="Racah")
    # check_Bqks(k, j, convention="Wybourne")

    #emat_in = np.eye(3); alpha = 0; beta = 90; gamma = 0
    #find_rotated_reference_frame(emat_in, alpha, beta, gamma)

    #fin = "Bkqs_new.dat"
    #read_and_print_Bkqs(fin)

    #print_emat_for_spin_model(np.eye(3))

    pass

