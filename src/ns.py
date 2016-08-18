from dolfin import *
set_log_level(ERROR)

# parameters
T   = 1.0 # final time
rho = 1.0
dt  = 0.1 # time step
nu  = 1.0/8 # kinematic viscosity
s   = 0.5 # 0.5 for crank-nicolson, 1.0 for backwards

# exact_solution
U = Expression(("x[1]*(1-x[1])",  "0"), degree = 2)
P = Expression("-2*nu*rho*(1-x[0])", nu = nu, rho = rho, degree = 1)



# defining functionspaces
mesh = UnitSquareMesh(16, 16)
# P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
# P2 = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
# TH = (P2 * P2) * P1 # Taylor-Hood element

#W = FunctionSpace(mesh, TH)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q
# define variational forms
w = Function(W)  # I store my solution in w
u, p = split(w)  # u is the u_n+1
u0 = interpolate(U, V)            # u0 takes the parabolic function U as initial value 
u1 = s * u + (1-s) * u0           # this is my u_mid
v, q = TestFunctions(W)

x = SpatialCoordinate(mesh)
u_exact = as_vector((x[1]*(1-x[1]), 0))
p_exact = 2*nu*rho*(1-x[0])
f = -rho*nu*div(grad(u_exact)) + grad(p_exact) + rho*grad(u_exact)*u_exact
print assemble(inner(f,f)*dx)
#exit()

# The rho is missing
dudt = Constant(1./dt) * inner(u - u0, v) * dx
a = (Constant(nu) * inner(grad(u1), grad(v)) 
     + inner(grad(u1)*u0, v)
     + p * div(v) + q * div(u) + inner(f,v)) * dx


# define boundary conditions
fd = FacetFunction("size_t", mesh)
CompiledSubDomain("x[0]<DOLFIN_EPS").mark(fd, 1)
CompiledSubDomain("x[0]>1-DOLFIN_EPS").mark(fd, 2)
CompiledSubDomain("x[1]>1-DOLFIN_EPS").mark(fd, 3)
CompiledSubDomain("x[1]<DOLFIN_EPS").mark(fd, 4)

plot(fd, interactive())


bcs = [DirichletBC(W.sub(0), U, fd, 1),
       DirichletBC(W.sub(0), U, fd, 3),
       DirichletBC(W.sub(0), U, fd, 4),
       DirichletBC(W.sub(1), Constant(0.), fd, 2)]


# prepare for time loop
u_assigner = FunctionAssigner(V, W.sub(0))
p_assigner = FunctionAssigner(Q, W.sub(1))

# loop over time steps
t = 0
while t < T:
    
    solve(dudt + a == 0, w, bcs)

    # update
    u_assigner.assign(u0, w.sub(0))
    t += dt     

# I don't need to update the pressure at each time step because the pressure at the next time step doesn't appear in my numerical scheme
p0 = Function(Q)
p_assigner.assign(p0, w.sub(1))

# compute errors
print "||u - uh|| = {0:1.4e}".format(errornorm(U, u0, "L2"))
print "||p - ph|| = {0:1.4e}".format(errornorm(P, p0, "L2"))

plot(u0); plot(p0); interactive()
