## N-S + ALE + elastic boundary

from dolfin import *

# parameters
N = [2**3]
T = 1.5
mu = 1.0
rho = 1.0
k = 1.0         # elastic constant
theta = 0.5     # 0.5 for Crank-Nicolson, 1.0 for backwards

dt = 0.05
#dt = 0.025
#dt = 0.0125

for n in N :
    
    # define mesh
    mesh = UnitSquareMesh(n, n)
    x = SpatialCoordinate(mesh)
    
    # finite element spaces for velocity and pressure (Taylor-Hood elements)
    V = VectorFunctionSpace(mesh, "CG", 2)
    P = FunctionSpace(mesh, "CG", 1)
    VP = V * P
    W = VectorFunctionSpace(mesh, "CG", 1) # space for mesh velocity
    
    
    u, p = TrialFunctions(VP)   # u is a trial function of V (and the velocity at the following time step
                                # while p a trial function of P
    w = TrialFunction(W)
    
    v, q = TestFunctions(VP)
    z = TestFunction(W)
    
    # The functions are initialized to zero
    # u0 is the velocity at the previous time step, set to zero as initial guess
    up0 = Function(VP)   # Current velocity-pressure solution
    u0, p0 = split(up0)  # u0 is not a function but "part" of a function, just a "symbolic" splitting
                         # REMEMBER: In this way when I update the up0 also the u0 and p0 get updated
                         # This is different from up0.split which I can use if I want to plot
    w0 = Function(W)
    
    f0 = Constant((0.0, 0.0))
    f = Constant((0.0, 0.0))
    u_inlet = Expression(("0.0", "x[0]*(x[0] - 1) + 1"), degree = 2)
    
    #y = 
    
    u_mid = (1.0-theta)*u0 + theta*u
    f_mid = (1.0-theta)*f0 + theta*f
    p_mid = (1.0-theta)*p0 + theta*p
    
    # ----- Variational formulation for NS ---------
    dudt = Constant(1./dt) * inner(u - u0, v) * dx
    a = ( Constant(rho) * inner(grad(u_mid)*(u0 - w0), v)   # ALE term
         + Constant(mu) * inner(grad(u_mid), grad(v))
         - p * div(v)
         + q * div(u)
         - inner(f,v) ) * dx
    #b = - inner(Constant(k) * y, v) * ds
    
    # ----- Variational form. for Laplace/Poisson -----
    laplace = inner(grad(w), grad(z)) * dx
    
    
    # Define boundary conditions
    fd = FacetFunction("size_t", mesh)
    CompiledSubDomain("near(x[0], 0.0)").mark(fd, 1) # left wall (cord)     PHYSICAL BOUNDARY --> here the values of w and u have to be the same
    CompiledSubDomain("near(x[0], 1.0)").mark(fd, 2) # right wall (tissue)  PHYSICAL BOUNDARY --> here the values of w and u have to be the same
    CompiledSubDomain("near(x[1], 1.0)").mark(fd, 3) # top wall (inlet)
    CompiledSubDomain("near(x[1], 0.0)").mark(fd, 4) # bottom wall (outlet)
    
    # Here I need to impose just the Dirichlet conditions. The ones regarding the stresses were already encountered in the
    # weak formulation
    bcs = [DirichletBC(VP.sub(0), u_inlet, fd, 3),   # inlet
           DirichletBC(VP.sub(0), Constant((0.0,0.0)), fd, 1)]   # left wall

    w_up = Expression(("-2*cos(4*pi*t)*x[1]*(x[1] - 1)", "0"), t = 0.5)  # I need the 4*pi so it does multiply cycles
    
    bcs_mesh = [DirichletBC(W, Constant((0.0,0.0)), fd, 1),     # PHYSICAL BOUNDARY --> here the values of w and u have to be the same
                DirichletBC(W, w_up, fd, 2),                       # PHYSICAL BOUNDARY --> here the values of w^(k+1) and u^(k+1) have to be the same
                DirichletBC(W, Constant((0.0,0.0)), fd, 3),
                DirichletBC(W, Constant((0.0,0.0)), fd, 4)]
    
    
    
    # prepare for time loop
    u_assigner = FunctionAssigner(V, VP.sub(0))
    p_assigner = FunctionAssigner(P, VP.sub(1))

    # Y is the adding displacement (what we add in order to move the mesh)
    Y = Function(W)  # by default this is 0 in all the components
    X = Function(W)  # in here I will put the displacement X^(n+1) = X^n + dt*(w^n)
    W_ = Function(W) # here I store my solution

    # loop over time steps
    t = dt
    while t < (T - 1E-9):
        
        w_up.t = t
        
        # solve the NS equations
        solve(dudt + a == 0, up0, bcs)
        
        # update
        u_assigner.assign(u0, up0.sub(0))

        # solve for mesh velocity
        solve(laplace == 0, w, bcs_mesh)
        
        # compute the mesh displacement
        Y.vector()[:] = w.vector()[:]*dt  # this is what I add in order to move the mesh
        X.vector()[:] += Y.vector()[:]

        ALE.move(mesh, Y)
        mesh.bounding_box_tree().build(mesh)
        plot(mesh)
        
        
        t += dt
        