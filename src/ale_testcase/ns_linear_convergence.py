from dolfin import *
set_log_level(ERROR)

# parameters
T   = 1.0    # final time
dt  = 0.1    # time step
nu  = 1.0    # kinematic viscosity
s   = 1.0    # 0.5 for crank-nicolson, 1.0 for backwards
N   = 2**2

dt *= .5**7

# defining functionspaces
mesh = UnitSquareMesh(N, N)
#P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
#P2 = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
#TH = VectorElement(P2) * P1 # Taylor-Hood element
# 
# P1 = VectorFunctionSpace(mesh, "CG", 1)
# P2 = FunctionSpace(mesh, "CG", 2)
# TH = P2 * P1
# 
# W = FunctionSpace(mesh, TH)
# V = FunctionSpace(mesh, VectorElement(P2))
# Q = FunctionSpace(mesh, P1)
# D = FunctionSpace(mesh, VectorElement(P1))

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q
D = VectorFunctionSpace(mesh, "CG", 1)

# exact_solution
U = Expression(("4*x[1]*(1-x[1])",  "0"), degree = 2)
P = Expression("-8*nu*(1-x[0])", nu = nu, degree = 1)
U_mesh = Expression(("0", "C * cos(r*t)*x[0]*(1-x[0])"), 
                    degree = 2,
                    t = 0, r = .5  * pi, C = 1)

# define variational forms for NS
w = Function(W); u, p = split(w)
u0 = interpolate(U, V)
u1 = Constant(s) * u + Constant(1-s) * u0
u_mesh = interpolate(U_mesh, D)
v, q = TestFunctions(W)


# MAGNE MODIFIED SOMETHING IN THE VARIATIONAL FORM AS WELL
dudt = Constant(1./dt) * inner(u - u0, v) * dx
a = (Constant(nu) * inner(grad(u1), grad(v)) 
     - inner(grad(u1)*(u_mesh), v)
     + p * div(v) + q * div(u) ) * dx
f = Constant((0.,0.)) 

# define Laplace problem
v_mesh = TestFunction(D)
l = inner(grad(u_mesh), grad(v_mesh)) * dx

# define boundary conditions
fd = FacetFunction("size_t", mesh)
CompiledSubDomain("x[0]<DOLFIN_EPS").mark(fd, 1)
CompiledSubDomain("x[0]>1-DOLFIN_EPS").mark(fd, 2)
CompiledSubDomain("x[1]>1-DOLFIN_EPS").mark(fd, 3)
CompiledSubDomain("x[1]<DOLFIN_EPS").mark(fd, 4)

bcs = [DirichletBC(W.sub(0), U, fd, 1),
       DirichletBC(W.sub(0), U, fd, 3),
       DirichletBC(W.sub(0), U, fd, 4),
       DirichletBC(W.sub(1), Constant(0.), fd, 2)]

bcs_mesh = [DirichletBC(D, U_mesh, "on_boundary")]

# prepare for time loop
t = 0
displacement = Function(D)
u_assigner = FunctionAssigner(V, W.sub(0))
p_assigner = FunctionAssigner(Q, W.sub(1))

# loop over time steps
while t < T:
    t += dt
    
    # solve for mesh displacement
    U_mesh.t = t
    solve(l == 0, u_mesh, bcs_mesh)
    displacement.vector()[:] = dt * u_mesh.vector()
    
    ALE.move(mesh, displacement)
    
    # solve NS
    solve(dudt + a == 0, w, bcs)

    # update velocity field
    u_assigner.assign(u0, w.sub(0))
    plot(mesh)
    #print "t = {0:1.3f} : max(u0) = {1:1.4f}".format(t, max(u0.vector()))

p0 = Function(Q)
p_assigner.assign(p0, w.sub(1))

# compute errors
print ""
print "||u - uh|| = {0:1.4e}".format(errornorm(U, u0, "L2"))
print "||p - ph|| = {0:1.4e}".format(errornorm(P, p0, "L2"))

from matplotlib import pyplot
#plot(u0); pyplot.figure(); plot(p0); pyplot.show()
