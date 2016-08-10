## NAVIER-STOKES EQUATIONS ##

# rho * du/dt + rho * (grad(u) . u) - div( nu * grad(u) - pI ) = f
# div( u ) = 0

from dolfin import *

#N = [2**2, 2**3, 2**4, 2**5, 2**6]
N = [2**3]

dt = 0.01
#dt = 0.05
#dt = 0.025
#dt = 0.0125

for n in N :
    
   
    mesh = UnitSquareMesh(n, n, "crossed")    # crossed means the triangles are divided in 2
    
    #mesh = UnitSquareMesh(n, n) 
    x = SpatialCoordinate(mesh)
    
    # Taylor-Hood elements
    V = VectorFunctionSpace(mesh, "Lagrange", 2)  # space for u, v
    P = FunctionSpace(mesh, "Lagrange", 1)        # space for p, q
    W = VectorFunctionSpace(mesh, "Lagrange", 1)       # space for w
    Z = VectorFunctionSpace(mesh,"CG",3)
    VP = V * P                  
    
    # TH = V * P
    # VP = FunctionSpace(mesh, TH)
    
    u, p = TrialFunctions(VP)   # u is a trial function of V, while p a trial function of P
    w = TrialFunction(W)
    
    v, q = TestFunctions(VP)
    z = TestFunction(W)
    
    # The functions are initialized to zero
    up0 = Function(VP)
    u0, p0 = split(up0)  # u0 is not a function but "part" of a function, just a "symbolic" splitting?
    w0 = Function(W)
    
    
    T = 1.5
    nu = 1.0/8.0
    rho = 1.0
    theta = 0.5 
    f0 = Constant((0.0, 0.0))
    f = Constant((0.0, 0.0))
    #g = Constant((0.0, 0.0))
    
    #p_in  = Constant(1.0)
    p_out = Constant(1.0) 
    
    u_mid = (1.0-theta)*u0 + theta*u
    f_mid = (1.0-theta)*f0 + theta*f
    p_mid = (1.0-theta)*p0 + theta*p
    
    # ------------------
    
    #w1 = -2*cos(4*pi*t)*x[0]*(x[0] - 1)
    #y1 = -1/(2*pi)*sin(4*pi*t)*x[0]*(x[0] - 1)
    
    #y_t = Expression(( "0", "y1" ))   # displacement
    u_exact = as_vector( (x[1] - ( -1/(2*pi)*sin(4*pi*t)*x[0]*(x[0] - 1) ) )*(x[1] - ( -1/(2*pi)*sin(4*pi*t)*x[0]*(x[0] - 1) ) - 1), \
                        -2*cos(4*pi*t)*x[0]*(x[0] - 1) )
    p_exact = 0.5 - x[1] 
    
    u_exact_e = Expression((" (x[1] - (-1/(2*pi)*sin(4*pi*t)*x[0]*(x[0] - 1)) )*(x[1]) - (-1/(2*pi)*sin(4*pi*t)*x[0]*(x[0] - 1)) - 1) ", \
                            " -2*cos(4*pi*t)*x[0]*(x[0] - 1) " ), t = 0.0, domain=mesh, degree=2)
    p_exact_e = Expression("0.5-x[1]", domain=mesh, degree=1)
        
    u_e = interpolate(u_exact_e,Z)
    
    # MISSING
    # du/dt = as_vector(w1,d(w1)/dt )
    # du_exact = as_vector( -2*cos(4*pi*t)*x[0]*(x[0] - 1), 8*pi*sin(4*pi*t)*x[0]*(x[0] - 1))
    
    # w0 =.....
    
    # f = - rho*nu*div(grad(u_exact)) + grad(p_exact) + rho*dot(grad(u_exact), u_exact - w0) + du_exact
    # I also need the term du/dt because now I chose as exact solution a solution that varies in time
    
    
    
    # ------------- Boundary conditions for Navier-Stokes
    
    # inflow = DirichletBC(VP.sub(1), p_in, "(x[0] < DOLFIN_EPS)&& on_boundary" )
    # outflow = DirichletBC(VP.sub(1), p_out, "(x[0] > (1- DOLFIN_EPS))&& on_boundary" )
    # walls = DirichletBC(VP.sub(0), (0.0, 0.0) , "((x[1] < DOLFIN_EPS)||(x[1] > (1 - DOLFIN_EPS)))&& on_boundary")
    
    w_up = Expression(("0", "-2*cos(4*pi*t)*x[0]*(x[0] - 1)"), t = 0.5)  # I need the 4*pi so it does multiply cycles
    
    inflow = DirichletBC(VP.sub(0), Expression(("x[1]*(1-x[1])", "0")), "near(x[0], 0.0) && on_boundary" )
    outflow = DirichletBC(VP.sub(1), p_out, "near(x[0], 1.0) && on_boundary" )
    upper_wall =  DirichletBC(VP.sub(0), (0.0, 0.0), "near(x[1], 1.0) && on_boundary")
    lower_wall = DirichletBC(VP.sub(0), (0.0, 0.0), "near(x[1], 0.0) && on_boundary")
    
    # -------------- Boundary conditions for Poisson 
    
    up = DirichletBC(W, w_up, "near(x[1], 1.0) && on_boundary")
    down = DirichletBC(W, w_up, "near(x[1], 0.0) && on_boundary")
    
    contour = DirichletBC(W, (0.0, 0.0), " ( near(x[0], 0.0) || near(x[0], 1.0)) \
                          || (near(x[0], 0.0) && near(x[1], 0.0)) \
                          || (near(x[0], 1.0) && near(x[1], 1.0) ) \
                          && on_boundary")
    
    # #this is to verify that I am actually applying some BC
    # U = Function(VP)
    # # this applies BC to a vector, where U is a function
    # inflow.apply(U.vector())
    # outflow.apply(U.vector())
    # walls.apply(U.vector()) 
    # uh, ph = U.split()
    # plot(uh)
    # plot(ph)
    # interactive()
    # exit()
    
    # w = Function(W)
    # up.apply(w.vector())
    # contour.apply(w.vector())   
    # plot(w)
    # interactive()
    # exit()
    
    
    # Neumann condition: sigma.n = 0        on the sides (I put in the variational form and the term on the sides is zero)
    
    bcu = [inflow, outflow, upper_wall, lower_wall]
    bcw = [up, down, contour]
    
    
    #-------- NAVIER-STOKES --------
    
    # Weak formulation
    dudt = Constant(dt**-1) * inner(u-u0, v) * dx
    a = Constant(nu) * inner(grad(u_mid), grad(v)) * dx
    c = inner(grad(u_mid)* (u0 - w0), v) * dx    # term with the mesh velocity w
    d = inner(grad(p), v) * dx  # this is fine because I set sigma.n = 0 in this case, be careful when I apply
                                # some other Neumann BC
    L = inner(f_mid,v)*dx # linear form
    #L = inner(f,v)*dx # linear form
    b = q * div(u) * dx   # from the continuity equation
    
    F = dudt + a + b + c + d - L
    
    # Bilinear and linear forms
    a0, L0 = lhs(F), rhs(F)    
    
    # -------- POISSON PROBLEM FOR w:  div( grad(w) ) = 0 --------
    
    a1 = inner(grad(w), grad(z))*dx
    L1 = dot(Constant((0.0,0.0)),z)*dx  
    #L1 = Expression("0.0")   
    # ----------------------
    
    
    #F0 = rho*dot(u,v)*dx
    #F0 -= rho*dot(u0,v)*dx
    #F0 +=  dt*rho*dot(grad(u_mid)*u0, v)*dx
    #F0 +=  dt*nu*inner(grad(u_mid), grad(v))*dx
    #F0 -= dt*p_mid*div(v)*dx
    #F0 -=  dt*dot(f_mid, v)*dx
    
    #F1 = div(u)*q*dx
    
    #F = F1 +  F0
    #a0 = lhs(F)
    #L0 = rhs(F)
    
    
    #use assemble_system
    # A = assemble(a0) # matrix
    # b = assemble(L0) # vector
    
    # This applies the BC to the system
    # for bc in bcu:
    #     bc.apply(A, b)
    
    
    t = dt
    
    VP_ = Function(VP)   # I want to store my solution here
    W_ = Function(W)
    
    # Y is the adding displacement (what we add in order to move the mesh)
    Y = Function(W)  # by default this is 0 in all the components
    
    X = Function(W)  # in here I will put the displacement X^(n+1) = X^n + dt*(w^n)

    
    
    solver = PETScLUSolver()
    
    
    #print "N = {}".format(n)
    #while (t - T) <= DOLFIN_EPS:
    while t <= T + 1E-9:
        
        w_up.t = t
        u_e.t = t
        
        
        du_exact = as_vector( -2*cos(4*pi*t)*x[0]*(x[0] - 1), 8*pi*sin(4*pi*t)*x[0]*(x[0] - 1))
        f = - rho*nu*div(grad(u_exact)) + grad(p_exact) + rho*dot(grad(u_exact), u_exact - w0) + du_exact
        
        #print float(sin(t))
        print "Solving for t = {}".format(t) 
        
        # ------- Solving the Navier-Stokes equations -------
        
        #print "Solving the Navier-Stokes equations"
        # I need to reassemble the system
        A = assemble(a0)
        b = assemble(L0)
        
        # I need the reapply the BC to the new system
        for bc in bcu:
            bc.apply(A, b)
        
        # Ax = b, where U is the x vector    
        solver.solve(A, VP_.vector(), b)
        
        

        # ------- Solving the Poisson problem -------
        
        #print "Solving the Poisson problem"
        A1 = assemble(a1)
        b1 = assemble(L1)
        
        #poisson = DirichletBC(W, u0, "on_boundary")  # I need to do this so I can update the boundary conditions at every step
        
        #bcw = [up, contour]
        
        for bc in bcw:
           bc.apply(A1, b1)
        
        
        solve(A1, W_.vector(), b1)
        
        
        # ------ Compute the mesh displacement -------
        
        
        Y.vector()[:] = w0.vector()[:]*dt
        X.vector()[:] += Y.vector()[:]
        
        #plot(Y, title="added displacement")
        #plot(X, title = "total displacement" )
        #plot(w0, title="mesh velocity")
        #interactive()
        
        # ------- Move the mesh ------
        
        
        ALE.move(mesh, Y)
        mesh.bounding_box_tree().build(mesh)
        
        # -------- Update of solutions so the cycle can start again --------
        # I need to assign up0 because u0 and p0 are not "proper" functions
        up0.assign(VP_)   # the first part of VP_ is u0, and the second part is p0
        w0.assign(W_)
        
        u0, p0 = VP_.split(True)
        L2_error_u = assemble((u_e-u0)**2 * dx)**.5
        print L2_error_u
        
        #plot(u0, interactive = True)
        
        
        t += dt
        #plot(mesh, interactive = True)
        
        
       
    u0, p0 = VP_.split() 
    plot(u0)
    plot(u_exact)
    #plot(w0)    
    print "t_final = {}".format(t - dt)    
    print "dt = {}".format(dt)   
    #print "T = {}".format(t)
    #print "u(1, 0.5, t = 0.5) = {}".format(U(Point(1, 0.5))[0])
    print("------")    
    
    #plot(u0)
    #plot(p0)
    interactive()
    
    #u0, p0 = VP_.split()
    #ufile = File("velocity_64_0.0125.pvd")
    #ufile << u0
    