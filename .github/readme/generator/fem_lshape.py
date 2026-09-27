"""P1 finite elements for the first Dirichlet eigenpair of the L-shaped domain."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

LAMBDA_1 = 9.6397238440219   # reference value for (-1,1)^2 minus (0,1)x(-1,0)


def lshape_mesh(n):
    """Structured triangulation of the L-shape with h = 1/n (one diagonal per square)."""
    ticks = np.linspace(-1.0, 1.0, 2 * n + 1)
    X, Y = np.meshgrid(ticks, ticks, indexing="ij")
    inside = ~((X > 1e-12) & (Y < -1e-12))                  # drop the open lower-right quadrant
    index = -np.ones(X.shape, dtype=int)
    index[inside] = np.arange(inside.sum())
    pts = np.column_stack([X[inside], Y[inside]])
    tris = []
    for i in range(2 * n):
        for j in range(2 * n):
            cx, cy = (ticks[i] + ticks[i + 1]) / 2, (ticks[j] + ticks[j + 1]) / 2
            if cx > 0 and cy < 0:
                continue
            a, b, c, d = index[i, j], index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]
            tris += [(a, b, c), (a, c, d)]
    tris = np.array(tris)
    x, y = pts[:, 0], pts[:, 1]
    tol = 1e-12
    boundary = ((np.abs(np.abs(x) - 1) < tol) | (np.abs(np.abs(y) - 1) < tol)
                | ((np.abs(x) < tol) & (y <= tol)) | ((np.abs(y) < tol) & (x >= -tol)))
    return pts, tris, boundary


def assemble(pts, tris):
    p = pts[tris]                                   # (T, 3, 2)
    d1, d2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    area = 0.5 * np.abs(det)
    # gradients of the barycentric coordinates
    inv = np.stack([np.stack([d2[:, 1], -d2[:, 0]], 1), np.stack([-d1[:, 1], d1[:, 0]], 1)], 1) / det[:, None, None]
    g1, g2 = inv[:, 0], inv[:, 1]
    grads = np.stack([-g1 - g2, g1, g2], 1)          # (T, 3, 2)
    Ke = area[:, None, None] * np.einsum("tik,tjk->tij", grads, grads)
    Me = area[:, None, None] / 12.0 * (np.ones((3, 3)) + np.eye(3))
    rows = np.repeat(tris, 3, axis=1).ravel()
    cols = np.tile(tris, (1, 3)).ravel()
    N = len(pts)
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(N, N)).tocsr()
    M = sp.coo_matrix((Me.ravel(), (rows, cols)), shape=(N, N)).tocsr()
    return K, M


def first_eigenpair(n):
    pts, tris, boundary = lshape_mesh(n)
    K, M = assemble(pts, tris)
    free = np.flatnonzero(~boundary)
    vals, vecs = eigsh(K[free][:, free], k=1, M=M[free][:, free], sigma=0.0, which="LM")
    u = np.zeros(len(pts))
    u[free] = vecs[:, 0]
    u /= u[np.argmax(np.abs(u))]                     # positive, max = 1
    return pts, tris, u, float(vals[0]), len(free)


if __name__ == "__main__":
    for n in (4, 8, 16, 32, 64):
        _, tris, _, lam, dofs = first_eigenpair(n)
        print(f"h=1/{n:<3d} unknowns={dofs:6d} triangles={len(tris):6d} lambda_h={lam:.6f} error={lam - LAMBDA_1:.3e}")
