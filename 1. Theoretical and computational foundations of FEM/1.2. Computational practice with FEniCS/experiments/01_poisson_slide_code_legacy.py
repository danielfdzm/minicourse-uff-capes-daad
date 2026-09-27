from fenics import *

# Create mesh: unit square, 8x8 triangulation
mesh = UnitSquareMesh(8, 8)

# P1 Lagrange finite elements
V = FunctionSpace(mesh, 'P', 1)

# Exact solution / Dirichlet data
u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]',
                  degree=2)

# Identify boundary
def boundary(x, on_boundary):
    return on_boundary

# Apply Dirichlet BC
bc = DirichletBC(V, u_D, boundary)

u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)

# Bilinear form a(u,v) and linear form L(v)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx

# Solve
u_h = Function(V)
solve(a == L, u_h, bc)

# Compute errors
error_L2 = errornorm(u_D, u_h, 'L2')
error_H1 = errornorm(u_D, u_h, 'H1')
print(f"L2 error: {error_L2:.6e}")
print(f"H1 error: {error_H1:.6e}")

# Plot
plot(u_h)
plot(mesh)
