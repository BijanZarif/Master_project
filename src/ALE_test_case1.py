## NAVIER-STOKES EQUATIONS ##

# rho * du/dt + rho * (grad(u) . u) - div( nu * grad(u) - pI ) = f
# div( u ) = 0

from dolfin import *

#N = [2**2, 2**3, 2**4, 2**5, 2**6]
N = [2**3]

dt = 0.05
#dt = 0.025
#dt = 0.01
#dt = 0.0125
#dt = 0.001

for n in N :
    
   
    mesh = UnitSquareMesh(n, n, "crossed")    # crossed means the triangles are divided in 2
    
    #mesh = UnitSquareMesh(n, n) 
    x = SpatialCoordinate(mesh)
    
    # Taylor-Hood elements
    V = VectorFunctionSpace(mesh, "Lagrange", 2)  # space for u, v
    P = FunctionSpace(mesh, "Lagrange", 1)        # space for p, q
    W = VectorFunctionSpace(mesh, "Lagrange", 1)       # space for w
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
    
    p_out  = Constant(1.0)
    
    # I want to start from u0 = ((x[1]*(1-x[1]), 0)) as initial condition
    #u0 = as_vector((x[1]*(1-x[1]), 0))
    u0 = Constant((0., 0.)) #Expression((" x[1]*(1-x[1]) ", "0" ), domain=mesh, degree=2)
    
    w_exact = as_vector( (0,0) )
    u_exact = as_vector((x[1]*(1-x[1]), 0))  
    p_exact = 2 - x[0]
    f0 = -rho*nu*div(grad(u_exact)) + grad(p_exact) + grad(u_exact)*(u_exact - w_exact)
    
    u_mid = (1.0-theta)*u0 + theta*u
    f_mid = (1.0-theta)*f0 + theta*f
    #f_mid =  as_vector( (2*nu*rho + 2*rho*x[0]*(x[0] - 1)*(-2*x[1] + 1)*cos(4*pi*t) - 1 ,0) )
    p_mid = (1.0-theta)*p0 + theta*p
    
    # ------------- Boundary conditions for Navier-Stokes
    
    #w_up = Expression(("0", "-2*cos(4*pi*t)*x[0]*(x[0] - 1)"), t = 0.5)  # I need the 4*pi so it does multiply cycles
    
    w_up = Expression(("0", "0"))
    
    u_exact_e = Expression((" x[1]*(1-x[1]) ", "0" ), domain=mesh, degree=2)
    p_exact_e = Expression("2-x[0]", domain=mesh, degree=1)
     # the exact solution also has to satisfy the boundary conditions!!
    
   
    t = Constant(0.0)
    #w_exact = as_vector( (0, -2*cos(4*pi*t)*x[0]*(x[0] - 1)) )
    
    
    #f = -rho*nu*div(grad(u_exact)) + grad(p_exact) + grad(u_exact)*(u_exact - w_exact)
    
    #f_e = Expression(( "2*nu*rho + 2*rho*x[0]*(x[0] - 1)*(-2*x[1] + 1)*cos(4*pi*t) - 1", "0"), t = 0.0, rho=1.0, nu = 1.0/8.0)
    #f = as_vector( (2*nu*rho + 2*rho*x[0]*(x[0] - 1)*(-2*x[1] + 1)*cos(4*pi*t) - 1 ,0) )
    
    
    inflow = DirichletBC(VP.sub(0), u_exact_e, "near(x[0], 0.0) && on_boundary" )
    outflow = DirichletBC(VP.sub(1), p_out,  "near(x[0], 1.0) && on_boundary" )
    upper_wall =  DirichletBC(VP.sub(0), u_exact_e, "near(x[1], 1.0) && on_boundary")
    lower_wall = DirichletBC(VP.sub(0), u_exact_e , "near(x[1], 0.0) && on_boundary")
    
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
    # upper_wall.apply(U.vector())
    # lower_wall.apply(U.vector())
    # uh, ph = U.split()
    # plot(uh)
    # plot(ph)
    # interactive()
    # exit()
    
    # w = Function(W)
    # up.apply(w.vector())
    # down.apply(w.vector())
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
    #L = inner(f_e,v)*dx # linear form
    L = inner(f_mid,v)*dx # linear form
    b = q * div(u) * dx   # from the continuity equation
    
    F = dudt + a + b + c + d - L
    
    # Bilinear and linear forms
    a0, L0 = lhs(F), rhs(F)    
    
    # -------- POISSON PROBLEM FOR w:  div( grad(w) ) = 0 --------
    
    a1 = inner(grad(w), grad(z))*dx
    L1 = dot(Constant((0.0,0.0)),z)*dx  

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
        #f_e.t = t
        #print float(sin(t))
        #print "Solving for t = {}".format(t) 
        
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
        
        u0, p0 = VP_.split()
        plot(u0, interactive = True, title = str(t))
        #plot(u_exact, interactive = True)
        
        
        t += dt
        #plot(mesh)
        
        
        
    u0, p0 = VP_.split()
    #plot(u0, title = "u_h")
    #plot(u_exact, title = "exact", mesh = mesh)
    #plot(p0)
    #interactive()
    #from IPython import embed; embed()
    L2_error_u = assemble((u_exact-u0)**2 * dx)**.5
    print "||u - uh; L^2|| = {0:1.4e}".format(L2_error_u)

    #plot(u0) 
    #plot(u_exact)
    #L2_error_u = assemble((u_exact-u0)**2 * dx)**.5
    #print "||u - uh; L^2|| = {0:1.4e}".format(L2_error_u)
    #print "t_final = {}".format(t - dt)    
    #print "dt = {}".format(dt)   
    #print "T = {}".format(t)
    #print "u(1, 0.5, t = 0.5) = {}".format(U(Point(1, 0.5))[0])
    #print("------")    
    
    #plot(u0)
    #plot(p0)
    #interactive()
    
    #u0, p0 = VP_.split()
    #ufile = File("velocity_64_0.0125.pvd")
    #ufile << u0
    