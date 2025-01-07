# ==============================================================================
# This script is used to transform 
# crystal field parameters (CFPs) based on extended Stevens operators (ESO) to 
# crystal field parameters (CFPs) based on irreducible spherical tensors (ITO).
# ============================================================================== 

from functions import transform_Bkqs_to_Bqks

if __name__ == "__main__":

    # Bkqs: crystal field parameters based on extended Stevens operators
    # Bqks: crystal field parameters based on irreducible spherical tensors

    # Suppoted conventions for ITOs:
    # 1. "spherical": spherical harmonic tensor operators
    # 2. "Racah": Racah irreducible tensor operators
    # 3. "Wybourne": Wybourne irreducible tensor operators
    # 4. "noname": unnamed irreducible tensor operators

    # Input crystal field parameters
    fin = "Bkqs.dat"

    # Total angular momentum
    j = 8

    # Transform crystal field parameters
    transform_Bkqs_to_Bqks(fin, j, convention="spherical")

