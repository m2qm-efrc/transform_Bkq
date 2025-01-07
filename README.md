# Transform crystal field parameters

## 1. Overview

The purpose of this this code is twofold. 1. Transform the crystal field parameters (CFPs) Bkq for extended Stevens operators (ESOs) under a rotation of the reference frame. 2. Transform the ESO-based crystal field parameters into ITO-based crystal field parameters, and vice versa, where ITO stands for irreducible tensor operators.

## 2. Usage:

### 2.1 Rotation

The rotation can be specified in two ways. The first way is to specify the basis vectors of both the initial and final reference frames. The second way is to use Euler angles. 

1. Provide the initial ESO-based crystal field parameters in a file, say ```Bkqs.dat```.
2. Modify ```rotate_Bkq.py``` to specify the rotation, the input and output file names, and the total angular momentum J. 
3. ```python rotate_Bkq.py``` to rotate the crystal field parameters.

An example input file ```Bkqs.dat``` is provided along with the source code. The input file should contain the crystal field parameters in the following format:

```
Column 1: k
Column 2: q
Column 3: Bkq
```

### 2.2. ESO-based CFPs to/from ITO-based CFPs

Four flavors of the irreducible tensor operators are supported in this code, i.e. Racah, Wybourne, spherical harmonic, and noname. (Ref 1)

#### 2.2.1 ESO-based CFPs to ITO-based CFPs

1. Provide the initial ESO-based crystal field parameters in a file, say ```Bkqs.dat```.
2. Modify ```get_ITO_CFPs.py``` to specify the input file name and the total angular momentum J. 
3. ```python get_ITO_CFPs.py``` to transform the ESO-based crystal field parameters to ITO-based crystal field parameters.

#### 2.2.2 ITO-based CFPs to ESO-based CFPs

1. Provide the initial ITO-based crystal field parameters in a file, say ```Bkqs_spherical.dat```.
2. Modify ```get_ESO_CFPs.py``` to specify the input file name and the total angular momentum J. 
3. ```python get_ESO_CFPs.py``` to transform the ITO-based crystal field parameters to ESO-based crystal field parameters.

Note that the ITO-based crystal field parameters are complex numbers. The third column of the input file must include both the real and imaginary parts. 

## 3. References:

1. General expressions for Stevens and Racah operator equivalents, Duros et al, 2024, arXiv:2405.08978.
2. Quantum theory of angular momentum, D A Varshalovich, A N Moskalev, and V K Khersonskii, 1988. 
3. Transformation relations for the conventional Okq and normalised Okq Stevens operator equivalents with k=1 to 6 and -k⩽q⩽k, C Rudowicz, 1985
