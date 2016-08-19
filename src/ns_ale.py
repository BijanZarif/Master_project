from dolfin import *
set_log_level(ERROR)

# parameters
#N = [2**2, 2**3, 2**4, 2**5, 2**6]
N = [2**3]

#dt = 0.05
#dt = 0.025
dt = 0.0125
#dt = 0.00001

T   = 1.0 # final time
rho = 1.0
nu  = 1.0/8.0 # kinematic viscosity
theta   = 1 # 0.5 for crank-nicolson, 1.0 for backwards

# exact_solution
u_exact_e = Expression(("x[1]*(1-x[1])",  "0"), degree = 2)
p_exact_e = Expression("-2*nu*rho*(1-x[0])", nu = nu, rho = rho, degree = 1)
v_mesh_e = Expression(("0", "-2*C*cos(4*pi*t)*x[0]*(x[0] - 1)"), t = 0.0, C = 1)


for n in N :

    # defining functionspaces
    mesh = UnitSquareMesh(n, n)
    x = SpatialCoordinate(mesh)

    V = VectorFunctionSpace(mesh, "CG", 2)
    P = FunctionSpace(mesh, "CG", 1)
    W = VectorFunctionSpace(mesh, "Lagrange", 1)       # space for w (mesh velocity)
    VP = V * P
    
    # define variational forms
    #w = Function(W)     # Here I am going to store my mesh velocity
    v_mesh = interpolate(v_mesh_e, W)
    
    up = Function(VP)  # I store my solution in w
    u, p = split(up)  # u is the u_n+1
    u0 = interpolate(u_exact_e, V)            # u0 takes the parabolic function U as initial value 
    u1 = theta * u + (1-theta) * u0           # this is my u_mid
    
    v, q = TestFunctions(VP)
    z = TestFunction(W)
    
    t_ = Constant(0.0)
    w_exact = as_vector((0, -2*cos(4*pi*t_)*x[0]*(x[0]-1)  ))
    u_exact = as_vector((x[1]*(1-x[1]), 0))
    p_exact = 2*nu*rho*(1-x[0])
    f = -rho*nu*div(grad(u_exact)) + grad(p_exact) #+ rho*grad(u_exact)*(u_exact)# - v_mesh_e)
    #print assemble(inner(f,f)*dx)
    #exit()
    
    # ------- Variat. form NS -----
    # The rho is missing
    dudt = Constant(1./dt) * inner(u - u0, v) * dx
    a = (Constant(nu) * inner(grad(u1), grad(v)) 
         #+ inner(grad(u1)*(u0), v)# - v_mesh), v)
         + p * div(v) + q * div(u) - inner(f,v)) * dx
    
    # ----- Var. form Poisson problem -----
    laplace = inner(grad(v_mesh), grad(z))*dx
    
    
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
    
    #bcw = [DirichletBC(W, Constant(0., 0.), fd, 1),
    #       DirichletBC(W, w_up, fd, 3),
    #       DirichletBC(W, w_up, fd, 4),
    #       DirichletBC(W, Constant(0., 0.), fd, 2)]
    
    bcw = DirichletBC(W, v_mesh_e, "on_boundary")
    # I WANT TO PLOT THE BC SO I KNOW IF I DID THINGS OK
    
    # prepare for time loop
    u_assigner = FunctionAssigner(V, VP.sub(0))
    p_assigner = FunctionAssigner(P, VP.sub(1))
    u_x = Function(FunctionSpace(mesh, "DG", 0))
    u_x_ = TestFunction(FunctionSpace(mesh, "DG", 0))
    # Y is the adding displacement (what we add in order to move the mesh)
    Y = Function(W)  # by default this is 0 in all the components
    X = Function(W)  # in here I will put the displacement X^(n+1) = X^n + dt*(w^n)
    
    # loop over time steps
    t = 0
    while t < T + 1E-9:
        
        t_.assign(t)
        v_mesh_e.t = t
        
        # solve for mesh displacement
        solve(laplace == 0, v_mesh, bcw)
    
        # ------ Compute the mesh displacement -------
        Y.vector()[:] = v_mesh.vector()[:]*dt
        X.vector()[:] += Y.vector()[:]
        
        ALE.move(mesh, Y)
        mesh.bounding_box_tree().build(mesh)
        
        # solve the NS equations
        solve(dudt + a == 0, up, bcs)
        
        # update
        u_assigner.assign(u0, up.sub(0))
        solve((u_x-u0[0])*u_x_*dx == 0, u_x)
        plot(u_x, title = "x-component")
        t += dt     
    
    # I don't need to update the pressure at each time step because the pressure at the next time step doesn't appear in my numerical scheme
    p0 = Function(P)
    p_assigner.assign(p0, up.sub(1))
    
    # compute errors
    print "||u - uh|| = {0:1.4e}".format(errornorm(u_exact_e, u0, "L2"))
    print "||p - ph|| = {0:1.4e}".format(errornorm(p_exact_e, p0, "L2"))
    
    #plot(u0); plot(p0); interactive()
