import numpy as np

#===============================================================================================================
# Unitary matrix for to include clockwork mixing for U_7x7 for n=3 (ArXiv-1711.02070)

def construct_U(q, n):
    
    zero_vec = np.zeros(n)
    denom = np.sqrt(q**2 - q**(-2*n))
    u_R = np.array([1 / q**j * np.sqrt((q**2 - 1) / denom) for j in range(0, n + 1)])
    
    U_L = np.array([[np.sqrt(2 / (n + 1)) * np.sin(j * k * np.pi / (n + 1))
                    for k in range(1, n + 1)]
                    for j in range(1, n + 1)])
    
    def lambda_k(k):
        return q**2 + 1 - 2 * q * np.cos(k * np.pi / (n + 1))
    
    U_R = np.array([
        [
            np.sqrt(2 / ((n + 1) * lambda_k(k))) *
            (q * np.sin(j * k * np.pi / (n + 1)) - np.sin((j + 1) * k * np.pi / (n + 1)))
            for k in range(1, n + 1)
        ]
        for j in range(0, n + 1)
    ])
    
    # Construct the final matrix
    U_top = np.hstack([zero_vec[:, None], 1 / np.sqrt(2) * U_L, -1 / np.sqrt(2) * U_L])
    U_bottom = np.hstack([u_R[:, None], 1 / np.sqrt(2) * U_R, 1 / np.sqrt(2) * U_R])
    Ucw = np.vstack([U_top, U_bottom])
    
    return Ucw

q = 2.0  
n = 3    
Ucw = construct_U(q, n)
print(Ucw)
