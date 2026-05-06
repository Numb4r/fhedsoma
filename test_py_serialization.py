import numpy as np
import sys
import os

sys.path.insert(0,"build")
import ckks_python


N = 8192
q = 2**60
print("Gerando as Chaves")
keygen = ckks_python.KeyGenerator(N, q)
sk = keygen.secret_key()
pk = keygen.public_key()