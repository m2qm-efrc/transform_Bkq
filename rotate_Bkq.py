# ======================================================================================================================
# This script is used to rotate the crystal field parameters from one reference frame to another.
# The input crystal field parameters are stored in a file named "Bkqs.dat". The output crystal field parameters are
# stored in a file named "Bkqs_new.dat".
#
# There are two ways to specify the reference frame:
# 1. Use the basis vectors of the initial and target reference frames. The basis vectors are stored in the arrays
#    emat_in and emat_out, respectively. The basis vectors are normalized. The initial and target reference frames
#    are related by a rotation matrix R, which satisfies emat_out = R * emat_in.
# 2. Use the Euler angles from the initial reference frame to the target reference frame. The Euler angles are stored
#    in the array euler_angles (alpha, beta, gamma). The Euler angles are in the extrinsic, zyz convention, with the
#    gamma angle applied first. The Euler angles are in degrees.
# ======================================================================================================================

from functions import transform_Bkqs

if __name__ == "__main__":

    # Input crystal field parameters
    fin = "Bkqs.dat"

    # Ouput crystal field parameters
    fout = "Bkqs_new.dat"

    # Total angular momentum
    j = 8

    # -----------------------------------------------------------------------------------------------
    # Usage 1: Use the basis vectors of the initial and target reference frames
    # -----------------------------------------------------------------------------------------------

    # Initial reference frame, emat = [ex, ey, ez]. Each row is a basis vector.
    emat_in  = np.array([[ 1.000000, 0.000000, 0.000000], [ 0.000000, 0.000000, 1.000000], [ 0.000000,-1.000000, 0.000000]])

    # Target reference frame, emat = [ex, ey, ez]. Each row is a basis vector.
    emat_out = np.eye(3)

    # Rotate the crystal field parameters from the initial reference frame to the target reference frame
    trransform_Bkqs(fin, fout, j, use_emat=True, emat_in=emat_in, emat_out=emat_out)

    # -----------------------------------------------------------------------------------------------
    # Usage 2: Use the Euler angles from the initial reference frame to the target reference frame
    # -----------------------------------------------------------------------------------------------

    # Euler angles from the inital reference frame to the target reference frame. Convention: extrinsic, zyz, gamma-first. Unit: deg
    # alpha = 0; beta = 90; gamma = 0

    # Rotate the crystal field parameters from the initial reference frame to the target reference frame
    # trransform_Bkqs(fin, fout, j, use_emat=False, euler_angles=[alpha,beta,gamma])
    
