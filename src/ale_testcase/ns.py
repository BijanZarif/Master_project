from dolfin import *
set_log_level(ERROR)

# parameters
N = [2**2, 2**3, 2**4, 2**5, 2**6]
#N = [2**3]

#dt = 0.05
#dt = 0.025
dt = 0.0125
#dt = 0.00001

T   = 1.0 # final time
rho = 1.0
nu  = 1.0/8.0 # kinematic viscosity
theta   = 0.5 # 0.5 for crank-nicolson, 1.0 for backwards

# exact_solution
u_exact_e = Expression(("x[1]*(1-x[1])",  "0"), degree = 2)
p_exact_e = Expression("-2*nu*rho*(1-x[0])", nu = nu, rho = rho, degree = 1)

for n in N :

    # defining functionspaces
    mesh = UnitSquareMesh(n, n)
    x = SpatialCoordinate(mesh)

    V = VectorFunctionSpace(mesh, "CG", 2)
    P = FunctionSpace(mesh, "CG", 1)
    W = W = VectorFunctionSpace(mesh, "Lagrange", 1)       # space for w (mesh velocity)
    VP = V * P
    
    # define variational forms
    up = Function(VP)  # I store my solution in w
    u, p = split(up)  # u is the u_n+1
    u0 = interpolate(u_exact_e, V)            # u0 takes the parabolic function U as initial value 
    u1 = theta * u + (1-theta) * u0           # this is my u_mid
    
    v, q = TestFunctions(VP)
    
    u_exact = as_vector((x[1]*(1-x[1]), 0))
    p_exact = 2*nu*rho*(1-x[0])
    f = -rho*nu*div(grad(u_exact)) + grad(p_exact) + rho*grad(u_exact)*u_exact
    print assemble(inner(f,f)*dx)
    #exit()
    
    # The rho is missing
    dudt = Constant(1./dt) * inner(u - u0, v) * dx
    a = (Constant(nu) * inner(grad(u1), grad(v)) 
         + inner(grad(u1)*u0, v)
         + p * div(v) + q * div(u) - inner(f,v)) * dx
    
    
    # define boundary conditions
    fd = FacetFunction("size_t", mesh)
    CompiledSubDomain("x[0]<DOLFIN_EPS").mark(fd, 1)
    CompiledSubDomain("x[0]>1-DOLFIN_EPS").mark(fd, 2)
    CompiledSubDomain("x[1]>1-DOLFIN_EPS").mark(fd, 3)
    CompiledSubDomain("x[1]<DOLFIN_EPS").mark(fd, 4)
    
    #plot(fd, interactive())
    
    bcs = [DirichletBC(VP.sub(0), u_exact_e, fd, 1),
           DirichletBC(VP.sub(0), u_exact_e, fd, 3),
           DirichletBC(VP.sub(0), u_exact_e, fd, 4),
           DirichletBC(VP.sub(1), Constant(0.), fd, 2)]
    
    
    # prepare for time loop
    u_assigner = FunctionAssigner(V, VP.sub(0))
    p_assigner = FunctionAssigner(P, VP.sub(1))
    
    # loop over time steps
    t = 0
    while t < T + 1E-9:
        
        solve(dudt + a == 0, up, bcs)
    
        # update
        u_assigner.assign(u0, up.sub(0))
        t += dt     
    
    # I don't need to update the pressure at each time step because the pressure at the next time step doesn't appear in my numerical scheme
    p0 = Function(P)
    p_assigner.assign(p0, up.sub(1))
    
    # compute errors
    print "||u - uh|| = {0:1.4e}".format(errornorm(u_exact_e, u0, "L2"))
    print "||p - ph|| = {0:1.4e}".format(errornorm(p_exact_e, p0, "L2"))
    
    #plot(u0); plot(p0); interactive()
