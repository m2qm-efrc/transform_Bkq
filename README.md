# Transform crystal field parameters

The purpose of this this code is two fold. 1. Transform the crystal field parameters (CFPs) Bkq for extended Stevens operators (ESOs) under a rotation of the reference frame. 2. Transform the ESO-based crystal field parameters into ITO-based crystal field parameters, and vice versa, where ITO stands for irreducible tensor operators.

## Usage:

1. ```python rotate_Bkq.py``` to rotate the ESO-based crystal field parameters under a rotation of the reference frame. The rotation can be specified in two ways in the code ```rotate_Bkq.py```. The first way is to specify the basis vectors of both the initial and final reference frames. The second way is to use Euler angles. 

2. ```python get_ITO_CF_parameters.py``` to transform the ESO-based crystal field parameters to ITO-based crystal field parameters, and vice versa. Four flavors of the irreducible tensor operators are supported in this code, i.e. Racah, Wybourne, spherical harmonic, and noname. 

An example input file Bkqs.dat is provided for both commands given above. The input file should contain the crystal field parameters in the following format:

```
Column 1: k
Column 2: q
Column 3: Bkq
```

## References:

1. General expressions for Stevens and Racah operator equivalents, Duros et al, 2024, arXiv:2405.08978.
2. Quantum theory of angular momentum, D A Varshalovich, A N Moskalev, and V K Khersonskii, 1988. 
3. Transformation relations for the conventional Okq and normalised Okq Stevens operator equivalents with k=1 to 6 and -k⩽q⩽k, C Rudowicz, 1985

