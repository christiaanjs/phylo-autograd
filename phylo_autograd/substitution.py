import autograd.numpy as np
from phylo_autograd.common import A, C, G, T

def eigen_transition_probs(U, lambd, Vt, t):
    diag = np.diag(np.exp(t * lambd))
    return np.dot(U, np.dot(diag, Vt))

class HKY:
    def transition_probs(kappa, pi, t):
        piY = pi[T] + pi[C]
        piR = pi[A] + pi[G]

        beta = -1 / (2.0 * (piR*piY + kappa * (pi[A]*pi[G] + pi[C]*pi[T])))
        A_R = 1.0 + piR * (kappa - 1)
        A_Y = 1.0 + piY * (kappa - 1)
        lambd = np.stack([ # Eigenvalues 
            0,
            beta,
            beta * A_Y,
            beta * A_R
        ])
        U = np.stack([ # Right eigenvectors as columns (rows of transpose)
            [1, 1, 1, 1],
            [1/piR, -1/piY, 1/piR, -1/piY],
            [0, pi[T]/piY, 0, -pi[C]/piY],
            [pi[G]/piR, 0, -pi[A]/piR, 0]
        ]).T

        Vt = np.stack([ # Left eigenvectors as rows
            [pi[A], pi[C], pi[G], pi[T]],
            [pi[A]*piY, -pi[C]*piR, pi[G]*piY, -pi[T]*piR],
            [0, 1, 0, -1],
            [1, 0, -1, 0]
        ])

        return eigen_transition_probs(U, lambd, Vt, t)


        
