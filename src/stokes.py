## STOKES PROBLEM ##
from numpy import *
from fenics import *
from mshr import *

# domain = Rectangle(Point(0., 0.), Point(1.0,1.0))
# mesh = generate_mesh(domain, 16)

mesh = UnitSquareMesh(12,12)

# ANOTHER WAY TO DEFINE THE TAYLOR HOOD ON FEniCS 1.7
#P1 = FiniteElement("Lagrange", triangle, 1)
#P2 = FiniteElement("Lagrange", triangle, 2)
#TH = (P2 * P2) * P1
#W = FunctionSpace(mesh, TH)

V = VectorFunctionSpace(mesh, "Lagrange", 2)  # space for velocity
Q = FunctionSpace(mesh, "Lagrange", 1)        # space for pressure
W = V * Q

u, p = TrialFunctions(W)
v, q = TestFunctions(W)

x = SpatialCoordinate(mesh)

# USE: u_exact = as_vector((0, sin(x[0]))) as a solution to verify the convergence 
u_exact = as_vector((0, x[0]*(1-x[0])))   # as_vector() ???
p_exact = 0.5 - x[1]

f = - div(grad(u_exact)) - grad(p_exact)
g = - div(grad(u_exact)) - f             # pressure gradient
# dt = 0.01
# T = 3.0

#f = Constant((0.0, 3.0))
#u_in  = Expression((" 0 ", "x[0]*(x[0] - 1)*sin(pi*omega*t)"), omega=1, t=0.0)
#u_out = Expression((" 0 ", "x[0]*(x[0] - 1)*sin(pi*omega*t)"), omega=1, t=0.0)  # OR PUT SOMETHING ELSE
u_exact = Expression((" 0 ", "x[0]*(1-x[0])" ), domain=mesh, degree=2)
p_exact = Expression("0.5-x[1]", domain=mesh, degree=1)
# 
# u_exact = Expression((" 0 ", "sin(x[0])" ))
# p_exact = Expression("0.5-x[1]")


# plot(u_exact, mesh = mesh, title = "exact velocity")
# plot(p_exact, mesh = mesh, title = "exact pressure")


inflow = DirichletBC(W.sub(0), u_exact, "(x[1] > 1.0 - DOLFIN_EPS) && on_boundary")
outflow = DirichletBC(W.sub(0), u_exact, "(x[1] < DOLFIN_EPS) && on_boundary")
sides = DirichletBC(W.sub(0), Constant((0.0, 0.0)) , "on_boundary && ((x[0] < DOLFIN_EPS) || (x[0] > 1.0 - DOLFIN_EPS))")

# # this is to verify that I am actually applying some BC
# U = Function(W)
# # this applies BC to a vector, where U is a function
# inflow.apply(U.vector())
# outflow.apply(U.vector())
# sides.apply(U.vector())
# 
# plot(U.split()[0])
# interactive()
# exit()

bcs = [inflow, outflow, sides]

 
# a = inner(grad(u), grad(v)) * dx
# b = q * div(u) * dx
# 
# lhs = a + b + adjoint(b)   # STILL NOT CLEAR 
# rhs = inner(f, v) * dx
# 
# A = assemble(lhs, PETScMatrix())
# B = assemble(rhs)
# 
# for bc in bcs:
#     bc.apply(A)
#     bc.apply(B)


F0 = inner(grad(u), grad(v))*dx
F0 += inner(p*Identity(2), grad(v))*dx
F0 -= inner(f, v)*dx

F1 = q*div(u)*dx

F = F0 + F1

a = lhs(F)
L = rhs(F)

A = assemble(a, PETScMatrix())
b = assemble(L)

for bc in bcs:
    bc.apply(A)
    bc.apply(b)


# ----------------------- #
# IN THIS WAY I AM SETTING THE NULL SPACE FOR THE PRESSURE: (STILL MISTERIOUS)
# since p + C for some constant C is still a solution, I take the pressure with mean value 0

constant_pressure = Function(W).vector()
constant_pressure[W.sub(1).dofmap().dofs()] = 1
null_space = VectorSpaceBasis([constant_pressure])
A.set_nullspace(null_space)

# ----------------------- #


U = Function(W)

#solve(lhs == rhs, U, bcs)
# solve(A, U.vector(), B)  # I am putting the solution in the vector U
solve(A, U.vector(), b)

uh, ph = U.split()   # I can't use split(U), because this would not be a proper function, but I can use it in the variational form

plot(uh, title = "computed velocity")
plot(ph, title = "computed pressure")

# IN THIS WAY I CAN PLOT AN EXPRESSION
#plot(u_exact, mesh = mesh, title = "exact velocity")
#plot(p_exact, mesh = mesh, title = "exact pressure")
#interactive()

# Pv_exact = project(u_exact, W.sub(0)) 

# compute errors
L2_error_u = assemble((u_exact-uh)**2 * dx)**.5
H1_error_u = assemble(grad(uh-u_exact)**2 * dx)**.5
H1_error_p = assemble((grad(ph) - g)**2 * dx)**.5 

print "||u - uh; L^2|| = {0:1.4e}".format(L2_error_u)
print "|u - uh; H^1| = {0:1.4e}".format(H1_error_u)
print "||p - ph; H^1|| = {0:1.4e}".format(H1_error_p)