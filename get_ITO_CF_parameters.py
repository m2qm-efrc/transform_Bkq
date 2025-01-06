# ==============================================================================
# This script is used to transform crystal field parameters (CFP) based on extended Stevens operators (ESO) to CFP based on irreducible spherical tensors (ITO) and vice versa.
#
# There are two ways of using this script:
# 1. Transform crystal field parameters (CFP) based on extended Stevens operators (ESO) to CFP based on irreducible spherical tensors (ITO)
# 2. Transform crystal field parameters (CFP) based on irreducible spherical tensors (ITO) to CFP based on extended Stevens operators (ESO)
# ============================================================================== 

from functions import transform_Bkqs_to_Bqks, transform_Bqks_to_Bkqs

if __name__ == "__main__":

    # Bkqs: crystal field parameters based on extended Stevens operators
    # Bqks: crystal field parameters based on irreducible spherical tensors

    # ----------------------------------------------------------------
    # Usage 1: ESO CFP to ITO CFP
    # ----------------------------------------------------------------

    # Input crystal field parameters
    fin = "Bkqs.dat"

    # Total angular momentum
    j = 8

    # Transform crystal field parameters
    transform_Bkqs_to_Bqks(fin, j, convention="spherical")




    # ----------------------------------------------------------------
    # Usage 2: ITO CFP to ESO CFP
    # ----------------------------------------------------------------

    # Input crystal field parameters
    # fin = "Bqks_spherical.dat"

    # Total angular momentum
    # j = 8

    # Transform crystal field parameters
    # transform_Bqks_to_Bkqs(fin, j, convention="spherical")

