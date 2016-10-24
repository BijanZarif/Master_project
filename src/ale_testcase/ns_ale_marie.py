from dolfin import *
set_log_level(ERROR)

# parameters
#N = [2**2, 2**3, 2**4]#, 2**5, 2**6]
N = [(2**n, 0.5**(2*n)) for n in range(1, 5)]

dt = 0.1
#dt = 0.05
#dt = 0.025
#dt = 0.0125
#dt = 0.00625

T = 0.5 # final time

# MER says: Rho is missing in the variational formulation, so make
# sure it stays at 1 here or fix that...
rho = 1.0
nu = 1.0/8.0 # kinematic viscosity
theta = 1.0 # 0.5 for crank-nicolson, 1.0 for backwards

# Exact solutions represented as dolfin.Expressions
u_exact_e = Expression(("x[1]*(1-x[1])",  "0"), degree=2)
p_exact_e = Expression("2*nu*rho*(1-x[0])", nu=nu, rho=rho, degree=1)
v_mesh_e = Expression(("0", "-2*C*sin(2*pi*t)*x[0]*(x[0] - 1)"),
                      t=0.0, C=1, degree=2)
#v_mesh_e = Expression(("0.0", "0.0"), t=0.0, C=1)

for n, dt in N :

    # Define the mesh
    mesh = UnitSquareMesh(n, n)
    x = SpatialCoordinate(mesh)

    if False:
        V = VectorElement("CG", mesh.ufl_cell(), 2)
        P = FiniteElement("CG", mesh.ufl_cell(), 1)
        VP = FunctionSpace(mesh, V*P)
        V = FunctionSpace(mesh, V)
        P = FunctionSpace(mesh, P)
    else:
        # Finite element spaces for velocity and pressure
        V = VectorFunctionSpace(mesh, "CG", 2)
        P = FunctionSpace(mesh, "CG", 1)
        VP = V * P

    # space for w (mesh velocity)
    W = VectorFunctionSpace(mesh, "Lagrange", 1)

    # Interpolate mesh velocity
    v_mesh = interpolate(v_mesh_e, W)

    # Current velocity-pressure solution
    up = Function(VP)

    # Symbolically split in velocity (u) and pressure (p)
    u, p = split(up)

    # Previous velocity, starting with exact solution as initial guess
    u0 = interpolate(u_exact_e, V)

    
    # Define u_mid
    u1 = theta*u + (1-theta)*u0   # why is this u0? shouldn't it be u1 = theta*u + (1-theta)*u ? Anyway if theta=1 it doesn't change anything

    # Define test functions for NS and for the mesh problem
    v, q = TestFunctions(VP)
    z = TestFunction(W)

    # Constant for time
    t_ = Constant(0.0)

    # Redefinition of exact solutions (MER says why!!!)
    u_exact = as_vector((x[1]*(1-x[1]), 0))
    p_exact = 2*nu*rho*(1-x[0])
    #f = Constant((0., 0.))
    f = -rho*nu*div(grad(u_exact)) + grad(p_exact) + rho*grad(u_exact)*(u_exact - v_mesh_e)

    # ------- Variat. form NS -----
    # The rho is missing
    dudt = Constant(1./dt) * inner(u - u0, v) * dx
    a = (Constant(nu) * inner(grad(u1), grad(v))
         + inner(grad(u1)*(u - v_mesh), v)
         + p * div(v) + q * div(u) - inner(f, v)) * dx
    
     # check the signs in the variational form, the term "p * div(v)" should have a minus, right? But if I change the sign, nothing changes, STRANGE
     # if I plot the pressure, this is almost always zero, STRANGE

    # ----- Var. form Poisson problem -----
    laplace = inner(grad(v_mesh), grad(z))*dx

    # define boundary conditions
    fd = FacetFunction("size_t", mesh)
    CompiledSubDomain("near(x[0], 0.0)").mark(fd, 1)    # bottom
    CompiledSubDomain("near(x[0], 1.0)").mark(fd, 2)    # top
    CompiledSubDomain("near(x[1], 1.0)").mark(fd, 3)    # right wall
    CompiledSubDomain("near(x[1], 0.0)").mark(fd, 4)    # left wall

    #plot(fd, interactive())
    #bcs = [DirichletBC(VP.sub(0), u_exact_e, fd, 1),
    #       DirichletBC(VP.sub(0), u_exact_e, fd, 3),
    #       DirichletBC(VP.sub(0), u_exact_e, fd, 4),
    #       DirichletBC(VP.sub(1), Constant(0.), fd, 2)]

    # Pressure boundary condition not a Dirichlet one, but a Neumann
    # one (to enforce in variational formulation)
    bcs = [DirichletBC(VP.sub(0), u_exact_e, fd, 1),    # bottom
           DirichletBC(VP.sub(0), u_exact_e, fd, 3),    # right
           DirichletBC(VP.sub(0), u_exact_e, fd, 4)]    # left

    #bcw = [DirichletBC(W, Constant(0., 0.), fd, 1),
    #       DirichletBC(W, v_mesh_e, fd, 3),
    #       DirichletBC(W, v_mesh_e, fd, 4),
    #       DirichletBC(W, Constant(0., 0.), fd, 2)]

    bcw = DirichletBC(W, v_mesh_e, "on_boundary")

    # prepare for time loop
    u_assigner = FunctionAssigner(V, VP.sub(0))
    p_assigner = FunctionAssigner(P, VP.sub(1))

    #u_x = Function(FunctionSpace(mesh, "DG", 0))
    #u_x_ = TestFunction(FunctionSpace(mesh, "DG", 0))

    # Y is the adding displacement (what we add in order to move the mesh)
    Y = Function(W)  # by default this is 0 in all the components
    X = Function(W)  # in here I will put the displacement X^(n+1) = X^n + dt*(w^n)

    # loop over time steps
    t = dt
    while t < (T - 1E-9):

        t_.assign(t)
        v_mesh_e.t = t

        # solve the NS equations
        solve(dudt + a == 0, up, bcs)

        # update
        u_assigner.assign(u0, up.sub(0))

        # solve for mesh velocity
        solve(laplace == 0, v_mesh, bcw)

        # ------ Compute the mesh displacement -------
        Y.vector()[:] = v_mesh.vector()[:]*dt
        X.vector()[:] += Y.vector()[:]

        ALE.move(mesh, Y)
        mesh.bounding_box_tree().build(mesh)

        plot(mesh)
        #solve((u_x-u0[0])*u_x_*dx == 0, u_x)
        #plot(u_x, title = "x-component")
        
        
        #plot(p)

        t += dt

    print "t = ", t
    # I don't need to update the pressure at each time step because the pressure at the next time step doesn't appear in my numerical scheme
    #p0 = Function(P)
    #p_assigner.assign(p0, up.sub(1))

    #plot(u0, title="u0")
    #plot(u_exact_e, mesh=mesh, title="u_e")
    #interactive()

    # compute errors
    print "N = {}".format(n)
    print "||u - uh|| = {0:1.4e}".format(errornorm(u_exact_e, u0, "L2"))
    #print "||p - ph|| = {0:1.4e}".format(errornorm(p_exact_e, p0, "L2"))

    #plot(u0); plot(p0); interactive()
