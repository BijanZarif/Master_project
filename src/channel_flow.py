from dolfin import *

mesh  = UnitSquareMesh(12,12, "crossed")    # crossed means the triangles are divided in 2
plot(mesh)
interactive()

# Taylor-Hood elements
V = VectorFunctionSpace(mesh, "Lagrange", 2)
Q = FunctionSpace(mesh, "Lagrange", 1)
# TH = V * Q
W = V * Q
# W = FunctionSpace(mesh, TH)

u, p = TrialFunctions(W)   # u is a trial function of V, while p a trial function of Q
v, q = TestFunctions(W)

up0 = Function(W)
u0, p0 = split(up0)  # u0 is not a function but "part" of a function, just a "symbolic" splitting?

# u0 = Function(V)   # it starts to zero
# p0 = Function(Q)   # it starts to zero

dt = 0.01
T = 3.0
nu = 1.0/8.0
omega = 1.0
rho = 1.0
theta = 0.5 
f0 = Constant((0.0, 0.0))
f = Constant((0.0, 0.0))


# u_in  = Expression((" 0 ", "x[0]*(x[0] - 1)*sin(pi*omega*t)"), omega=1, t=0.5)
u_in  = Expression((" 0 ", "x[0]*(x[0] - 1)*sin(pi*omega*t)"), omega=1, t=0.5)
u_out = Expression((" 0 ", "x[0]*(x[0] - 1)*sin(pi*omega*t)"), omega=1, t=0.5)  # OR PUT SOMETHING ELSE

u_mid = (1.0-theta)*u0 + theta*u
f_mid = (1.0-theta)*f0 + theta*f
p_mid = (1.0-theta)*p0 + theta*p

inflow = DirichletBC(W.sub(0), u_in, "x[1] > 1.0 - DOLFIN_EPS & on_boundary")
outflow = DirichletBC(W.sub(0), u_out, "x[1] < DOLFIN_EPS & on_boundary")

# # this is to verify that I am actually applying some BC
# U = Function(W)
# # this applies BC to a vector, where U is a function
# inflow.apply(U.vector())
# outflow.apply(U.vector())
# 
# plot(U.split()[0])
# interactive()
# exit()

# Neumann condition: sigma.n = 0        on the sides (I put in the variational form and the term on the sides is zero)

bcu = [inflow, outflow]

# F0 = rho*u*v*dx - rho*u0*v*dx + \
#      dt*rho*dot(grad(u_mid), u0)*v*dx + dt*v*dot(grad(u_mid), grad(v)) - dt*p_mid*div(v)*dx - dt*f_mid*v*dx

# dot JUST FOR 1-1 RANK
# inner FOR GREATER RANKS

# Ovind suggested: divide your form in different passages so if you have an error you are going to see exactly
# what part of the form gives the error

F0 = rho*dot(u,v)*dx
F0 -= rho*dot(u0,v)*dx
F0 +=  dt*rho*dot(grad(u_mid)*u0, v)*dx
F0 +=  dt*nu*inner(grad(u_mid), grad(v))*dx
F0 -= dt*p_mid*div(v)*dx
F0 -=  dt*dot(f_mid, v)*dx

F1 = div(u)*q*dx

F = F1 +  F0
a0 = lhs(F)
L0 = rhs(F)


#use assemble_system
# A = assemble(a0) # matrix
# b = assemble(L0) # vector

# This applies the BC to the system
# for bc in bcu:
#     bc.apply(A, b)


t = dt

U = Function(W)   # I want to store my solution here

while t < T + DOLFIN_EPS:
    
    u_in.t = t
    u_out.t = t
    
    # I need to reassemble the system
    A = assemble(a0)
    b = assemble(L0)
    
    # I need the reapply the BC to the new system
    for bc in bcu:
        bc.apply(A, b)
    
    # Ax=b, where U is the x vector    
    solve(A, U.vector(), b)
    
    # Move to next time step (in case I had separate equations)
    # u0.assign(u)
    # p0.assign(p)
    
    # I need to assign up0 because u0 and p0 are not "proper" functions
    up0.assign(U)   # the first part of U is u0, and the second part is p0
    
    t += dt
    print("t = ", t)


plot(u0)
interactive()