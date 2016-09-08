from dolfin import *

N = 2**5
g = Expression(("0.0", "0.0"))
f = Expression(("1.0", "1.0"))
l = Expression(("1.0", "1.0"))

mesh = UnitSquareMesh(N,N)
gamma = 10.0
h = CellSize(mesh)
n = FacetNormal(mesh)

V = VectorFunctionSpace(mesh, "CG", 2)
u = TrialFunction(V)
v = TestFunction(V)

un = dot(u,n) * n
vn = dot(v,n) * n
ut = u - un
vt = v - vn

a = inner(grad(u), grad(v)) * dx + ( -inner(dot(grad(ut), n), vt) - inner(dot(grad(vt), n), ut) + Constant(gamma)/h * inner(ut,vt) ) * ds
L = inner(f,v) * dx - inner(dot(grad(vt),n), g) * ds + Constant(gamma)/h * inner(g,vt) * ds + inner(l,vn) * ds

uh = Function(V)
solve(a==L, uh)

plot(uh, interactive = True)