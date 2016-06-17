from dolfin import *
mesh = UnitSquareMesh(32, 32)
# P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
# P2 = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
# TH = (P2 * P2) * P1
# 
# W = FunctionSpace(mesh, TH)

V = VectorFunctionSpace(mesh, "Lagrange", 2)  # space for velocity
Q = FunctionSpace(mesh, "Lagrange", 1)        # space for pressure
W = V * Q

# define solution
x = SpatialCoordinate(mesh)
y = (1-cos(0.8 * pi * x[0])) * (1-x[0])**2 \
  * (1-cos(0.8 * pi * x[1])) * (1-x[1])**2
U = as_vector((-y.dx(1), y.dx(0))) # velocity
F = - 2 * div(sym(grad(U)))        # forcing term
G = - div(grad(U)) - F             # pressure gradient

# define variational form
u, p = TrialFunctions(W)
v, q = TestFunctions(W)

a = (inner(grad(u),grad(v))+ p * div(v) + q * div(u)) * dx
L = inner(F, v) * dx

bcs = [DirichletBC(W.sub(0), U, "on_boundary")]

# solve
A, b = assemble_system(a, L, bcs, A_tensor = PETScMatrix())
wh = Function(W); uh, ph = wh.split()
solve(A, wh.vector(), b)

# compute errors
L2_error_U = assemble((U-uh)**2 * dx)
H1_error_U = assemble(grad(U-uh)**2 * dx)
H1_error_P = assemble((grad(ph) - G)**2 * dx) 

print "||u - uh; L^2|| = {0:1.4e}".format(L2_error_U)
print "||u - uh; H^1|| = {0:1.4e}".format(H1_error_U)
print "||p - ph; H^1|| = {0:1.4e}".format(H1_error_P)



