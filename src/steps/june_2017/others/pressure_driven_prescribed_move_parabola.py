## NAVIER-STOKES EQUATIONS ##

# rho * du/dt + rho * (grad(u) . u) - div( nu * grad(u) - pI ) = f
# div( u ) = 0

from dolfin import *

NN = [2**2, 2**3, 2**4, 2**5, 2**6]
dt = 1./NN[2]
NN = [2**3]

x0, x1 = 0.0, 60.0   # [mm]
y0, y1 = 0.0, 4.0  # [mm]

for N in NN:
    
    mesh = RectangleMesh(Point(x0, y0), Point(x1, y1), 10*N, N, "crossed")  # crossed means the triangles are divided in 2
    x = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)
    
    values_x0 = []
    values_x1 = []
    values_x2 = []
    time = []
    x_0 = Point(0.0, 2.0)  # left
    x_1 = Point(30.0, 2.0)    # middle
    x_2 = Point(60.0, 2.0)   # rightf
    
    # Taylor-Hood elements
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    VQ = V * Q
    W = VectorFunctionSpace(mesh, "CG", 1)       # ALE
    
    u, p = TrialFunctions(VQ)   # u is a trial function of V somehow, while p a trial function of Q
    v, q = TestFunctions(VQ)
    
    up0 = Function(VQ)
    u0, p0 = split(up0)  # u0 is not a function but "part" of a function, just a "symbolic" splitting?
    
    w = TrialFunction(W)        # ALE
    z = TestFunction(W)         # ALE
    w0 = Function(W)            # ALE
    
    T = 5     # only for oscillating p_inlet
    mu = 0.700e-3  # [g/(mm * s)]
    rho = 1e-3     # [g/mm^3] 
    theta = 0.5 
    f0 = Constant((0.0, 0.0))
    f = Constant((0.0, 0.0))
    
    ufile = File("results/pressure_driven_prescribed_move/velocity_8_0.0625.pvd")
    pfile = File("results/pressure_driven_prescribed_move/pressure_8_0.0625.pvd")
    outfile = open('fluxes.txt','w')
    
    #p_in  = Constant(1.0)
    amplitude = Constant(9.0)  #[kPa]
    p_in = Expression("a*sin(2*pi*t)", a=amplitude, t=0.0, degree=2)   # only for oscillating p_inlet
    p_out = Constant(0.0)
    #amplitude_move = Constant(0.001)   # the smaller the value, the smaller the deformation
    amplitude_move = Constant(0.0001)
    w_move = Expression(("0.0", "-a*cos(2*pi*t)*x[0]*(x[0] - 60)"), a=amplitude_move, degree = 2, t=0.0)   # ALE
    
    u_mid = (1.0-theta)*u0 + theta*u
    f_mid = (1.0-theta)*f0 + theta*f
    p_mid = (1.0-theta)*p0 + theta*p
    
    # Define boundary conditions
    fd = FacetFunction("size_t", mesh)
    CompiledSubDomain("near(x[0], x0) && on_boundary", x0 = x0).mark(fd, 1) # left wall   
    CompiledSubDomain("near(x[0], x1) && on_boundary", x1 = x1).mark(fd, 2) # right wall 
    #CompiledSubDomain("near(x[1], y1) && (x[0] != x1) && on_boundary", x1 = x1, y1 = y1).mark(fd, 3) # top wall
    #CompiledSubDomain("near(x[1], y0) && (x[0] != x1) && on_boundary", x1 = x1, y0 = y0).mark(fd, 4) # bottom wall 
    CompiledSubDomain("near(x[1], y1) ||(near(x[0], x1) && near(x[1], y1) ) && on_boundary", x1 = x1, y1 = y1).mark(fd, 3) # top wall 
    CompiledSubDomain("near(x[1], y0) ||(near(x[0], x1) && near(x[1], y0) ) && on_boundary", x1 = x1, y0 = y0).mark(fd, 4) # bottom wall 
    ds = Measure("ds", domain = mesh, subdomain_data = fd)
    #plot(fd)
    #interactive()
    
    left_wall = DirichletBC(VQ.sub(0).sub(1), Constant(0.0), fd, 1)   # left wall - I set the y (tangential) component of the velocity to zero
    right_wall = DirichletBC(VQ.sub(0).sub(1), Constant(0.0), fd, 2)   # right wall - I set the y (tangential) component of the velocity to zero
    top_wall = DirichletBC(VQ.sub(0), Constant((0.0, 0.0)) , fd, 3) # top
    bottom_wall = DirichletBC(VQ.sub(0), Constant((0.0, 0.0)) , fd, 4) # bottom
    # Neumann condition: sigma.n = 1        on the inlet
    # Neumann condition: sigma.n = 0        on the outlet
    
    # Conditions for mesh movement    # ALE
    left_wall_w = DirichletBC(W, Constant((0.0, 0.0)), fd, 1)   
    right_wall_w = DirichletBC(W, Constant((0.0, 0.0)), fd, 2)  
    top_wall_w = DirichletBC(W, w_move, fd, 3) # top
    bottom_wall_w = DirichletBC(W, Constant((0.0, 0.0)) , fd, 4)
    
    bcu = [left_wall, right_wall, top_wall, bottom_wall]
    bcw = [left_wall_w, right_wall_w, top_wall_w, bottom_wall_w]   # ALE
    
    F = rho * Constant(dt**-1) * inner(u-u0, v) * dx
    F += rho * inner(grad(u_mid)*(u0-w0), v) * dx
    F += Constant(mu) * inner(grad(u_mid), grad(v)) * dx
    F -= p * div(v) * dx
    F -= q * div(u) * dx
    F += inner(p_in * n, v)*ds(1)
    #F -= inner(p_in * n, v)*ds(2)
    F += inner(p_out * n, v)*ds(2)
    F -= inner(f_mid,v)*dx
    
    a0, L0 = lhs(F), rhs(F)
    
    
            
    a0, L0 = lhs(F), rhs(F)
        
    # ALE
    a1 = inner(grad(w), grad(z))*dx
    L1 = dot(Constant((0.0,0.0)),z)*dx
    
    W_ = Function(W)
    # Y is the adding displacement (what we add in order to move the mesh)
    Y = Function(W)  # by default this is 0 in all the components
    X = Function(W)  # in here I will put the displacement X^(n+1) = X^n + dt*(w^n)
    
    
    t = 0.0
    
    U = Function(VQ)   # I want to store my solution here
    solver = PETScLUSolver()
    
    
    print "N = {}".format(N)
    #while (t - T) <= DOLFIN_EPS:
    while t <= T + 1E-9:
         
        print "solving for t = {}".format(t)
        time.append(t)
        
        # I need to reassemble the system
        A = assemble(a0)
        b = assemble(L0)
        
        # I need the reapply the BC to the new system
        for bc in bcu:
            bc.apply(A, b)
        
        # Ax=b, where U is the x vector    
        solver.solve(A, U.vector(), b)
        
        ####### ALE #########
        # Solving the Poisson problem
        A1 = assemble(a1)
        b1 = assemble(L1)
        
        for bc in bcw:
           bc.apply(A1, b1)
        
        solve(A1, W_.vector(), b1)
        
        # Compute the mesh displacement
        Y.vector()[:] = w0.vector()[:]*dt
        X.vector()[:] += Y.vector()[:]
        
        # Move the mesh
        ALE.move(mesh, Y)
        mesh.bounding_box_tree().build(mesh)
        ####### ALE #########
        
        
        
        # I need to assign up0 because u0 and p0 are not "proper" functions
        up0.assign(U)   # the first part of U is u0, and the second part is p0
        w0.assign(W_)   # ALE
        
        u0, p0 = up0.split() 
        ufile << u0
        pfile << p0
        
        flux = assemble(inner(u0,n)*ds(2))   # flux through the outlet
        outfile.write('%g, %g\n'%(t, flux))
        
        t += dt
        p_in.t = t    # only for oscillating p_inlet
        w_move.t = t   # ALE
        #values_x0.append((U(x_0)[0]**2 + U(x_0)[1]**2)**0.5)
        #values_x1.append((U(x_1)[0]**2 + U(x_1)[1]**2)**0.5)
        #values_x2.append((U(x_2)[0]**2 + U(x_2)[1]**2)**0.5)
        #print values
        #print time
        
        
    print "t_final = {}".format(t - dt)    
    print "dt = {}".format(dt)   
    print("------")
    
    outfile.close()
    
    #print time
    #print values_x0
    #print values_x1
    #print values_x2
    #plt.figure()
    #plt.plot(time, values_x0, time, values_x1, time, values_x2, label = "Velocity in the point over time")
    #plt.plot(time, values_x1, label = "Velocity in the point over time")
    #plt.plot(time, values_x2, label = "Velocity in the point over time")
    #plt.show()