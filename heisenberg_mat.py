import numpy as np
from sympy import symbols, Matrix, simplify, N, re, im, I
from toolz import compose

def M(a, b, c):
    return Matrix([[1, a, c], [0, 1, b], [0, 0, 1]])

# Generators
r = M(1,0,0)
g = M(0,1,0)
b = M(-1,-1,1/2+(1/2)*I)

q_C = r*g*b
q_A = r.inv()*g.inv()*b.inv()

def check_cZ(mat, tol=1e-9):
    # 1) Check if mat is (3,3)
    try:
        if hasattr(mat, 'shape'):
            shape = mat.shape
        else:
            mat = np.array(mat)
            shape = mat.shape

        if shape != (3, 3): return False
    except:
        return False

    # For SymPy matrices, use N() for numerical evaluation; for NumPy, use directly
    def get_val(r_idx, c_idx):
        val = mat[r_idx, c_idx]
        try:
            return float(N(val)) if hasattr(val, 'evalf') else float(val)
        except TypeError: # If symbols remain, float conversion is not possible
            return val

    # 2) Check if diagonal elements are 1 (|val - 1| < tol)
    for i in range(3):
        val = get_val(i, i)
        if isinstance(val, float):
            if abs(val - 1.0) > tol: return False
        elif val != 1: return False # Strict comparison for symbolic cases

    # 3) Check if all elements except diagonal and (0,2) are 0 (|val| < tol)
    zero_positions = [(0, 1), (1, 0), (1, 2), (2, 0), (2, 1)]
    for row, col in zero_positions:
        val = get_val(row, col)
        if isinstance(val, float):
            if abs(val) > tol: return False
        elif val != 0: return False

    # If all conditions are met, return element (0,2)
    return mat[0, 2]

def get_nCA(q):
    n_C = re(q)+im(q)
    n_A = re(q)-im(q)
    return n_C, n_A

check_nCA = compose(get_nCA, check_cZ)

def heisenberg_product(word):
    accum_mul = 1
    mats = [r, g, b]
    for char in word:
        sgn = np.sign(char)
        val = abs(char)
        mat_now = mats[val-1] if sgn > 0 else mats[val-1].inv()
        accum_mul = accum_mul * mat_now
    return accum_mul

if __name__ == '__main__':
    # Test on boundary word elements
    print("q_C = ", q_C)
    print("check_cZ(q_C) = ", check_cZ(q_C))
    print("check_nCA(q_C) = ", check_nCA(q_C))
    
    # Example word testing
    word = [1, 2, 3] # r g b
    q = heisenberg_product(word)
    print(f"Product of {word}:", q)
    print("check_nCA(q):", check_nCA(q))
