from dolfin import *

N = 2**5
T = 1.5
g = Constant(0.0)
f = 1.0

mesh = UnitSquareMesh(N,N)
gamma = 10.0
h = CellSize(mesh)
n = FacetNormal(mesh)

V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)

a = inner(grad(u), grad(v)) * dx + ( -inner(dot(grad(u), n), v) - inner(dot(grad(v), n), u) + Constant(gamma)/h * inner(u,v) ) * ds
L = inner(f,v) * dx - inner(dot(grad(v),n), g) * ds + Constant(gamma)/h * inner(g,v) * ds

uh = Function(V)
solve(a==L, uh)

plot(uh, interactive = True)