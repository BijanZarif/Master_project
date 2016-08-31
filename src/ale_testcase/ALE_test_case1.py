## ALE + NAVIER-STOKES EQUATIONS ##
## TEST CASE WITH w = 0 ##

# rho * du/dt + rho * (grad(u) . u) - div( nu * grad(u) - pI ) = f
# div( u ) = 0

from dolfin import *

N = [2**2, 2**3, 2**4, 2**5, 2**6]
#N = [2**3]

#dt = 0.05
#dt = 0.025
#dt = 0.0125
dt = 0.00001

for n in N :
    
    mesh = UnitSquareMesh(n, n)    # crossed means the triangles are divided in 2
    x = SpatialCoordinate(mesh)
    
    # Taylor-Hood elements
    V = VectorFunctionSpace(mesh, "Lagrange", 2)  # space for u, v
    P = FunctionSpace(mesh, "Lagrange", 1)        # space for p, q
    W = VectorFunctionSpace(mesh, "Lagrange", 1)       # space for w
    VP = V * P                  
    
    u, p = TrialFunctions(VP)   # u is a trial function of V, while p a trial function of P
    w = TrialFunction(W)
    
    v, q = TestFunctions(VP)
    z = TestFunction(W)
    
    # The functions are initialized to zero
    up0 = Function(VP)
    u0, p0 = split(up0)  # In this way when I update the up0 also the u0 and p0 get updated
    w0 = Function(W)
     
    #T = 1.5
    T = 1*dt
    nu = 1.0/8.0
    rho = 1.0
    #theta = 0.5 # C-N
    theta = 1.0 # implicit
    
    # I want to start from u0 = ((x[1]*(1-x[1]), 0)) as initial condition
    w_exact = as_vector( (0,0) )  # I use this to compute the f0 "symbolically"
    p_exact = -2*nu*rho*(1 - x[0])
    u_exact = as_vector((x[1]*(1-x[1]), 0))
    u00 = project(u_exact, V)
    assign(up0.sub(0), u00)    # Here I do the assignment for u0, i.e. u0 = u_exact (sort of)
    
    f0 = -rho*nu*div(grad(u_exact)) + grad(p_exact) + rho*grad(u_exact)*(u_exact - w_exact)
    #print assemble(inner(f0,f0)*dx)  # to check the value of f0
    #exit()
    
    # ------------- Boundary conditions for Navier-Stokes
    # Don't forget that the exact solution has to satisfy the boundary conditions!!
    u_exact_e = Expression((" x[1]*(1-x[1]) ", "0" ), domain=mesh, degree=2)
    #p_exact_e = Expression("2*nu*rho*(1-x[0])", domain=mesh, degree=1, nu=nu, rho=rho)
    p_out  = Constant(0.0) # when x[0] = 1 on the boundary, p_exact = 0
    
    inflow = DirichletBC(VP.sub(0), u_exact_e, "near(x[0], 0.0) && on_boundary" )
    outflow = DirichletBC(VP.sub(1), p_out,  "near(x[0], 1.0) && on_boundary" )
    upper_wall =  DirichletBC(VP.sub(0), u_exact_e, "near(x[1], 1.0) && on_boundary")
    lower_wall = DirichletBC(VP.sub(0), u_exact_e , "near(x[1], 0.0) && on_boundary")
    
    # -------------- Boundary conditions for Poisson     
    w_up = Expression(("0", "0"))  # no movement of the boundary
    
    up = DirichletBC(W, w_up, "near(x[1], 1.0) && on_boundary")
    down = DirichletBC(W, w_up, "near(x[1], 0.0) && on_boundary")
    
    contour = DirichletBC(W, (0.0, 0.0), " ( near(x[0], 0.0) || near(x[0], 1.0)) \
                          || (near(x[0], 0.0) && near(x[1], 0.0)) \
                          || (near(x[0], 1.0) && near(x[1], 1.0) ) \
                          && on_boundary")
    # --------------------------
    
    bcu = [inflow, outflow, upper_wall, lower_wall]
    bcw = [up, down, contour]
    
    # ----------------
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
    # -------------------
    
    
    # --------------- VARIATIONAL FORMULATIONS ---------------
    
    #-------- NAVIER-STOKES --------
    u_mid = (1.0-theta)*u0 + theta*u
    
    # Weak formulation
    dudt = rho*Constant(1./dt) * inner(u-u0, v) * dx
    a = rho*Constant(nu) * inner(grad(u_mid), grad(v)) * dx
    c = rho*inner(grad(u_mid)* (u0 - w0), v) * dx    # term with the mesh velocity w
    d = inner(grad(p), v) * dx  # this is fine because I set sigma.n = 0 in this case, be careful when I apply some other Neumann BC
    L = inner(f0,v)*dx # linear form
    b = q * div(u) * dx   # from the continuity equation
    
    F = dudt + a + b + c + d - L
    
    # Bilinear and linear forms
    a0, L0 = lhs(F), rhs(F)    
    
    # -------- POISSON PROBLEM FOR w:  div( grad(w) ) = 0 --------
    a1 = inner(grad(w), grad(z))*dx
    L1 = dot(Constant((0.0,0.0)),z)*dx  
    # ----------------------
        
    VP_ = Function(VP)   # I want to store my solution here
    W_ = Function(W)
    
    # Y is the adding displacement (what we add in order to move the mesh)
    Y = Function(W)  # by default this is 0 in all the components
    X = Function(W)  # in here I will put the displacement X^(n+1) = X^n + dt*(w^n)
    
    solver = PETScLUSolver()
    t = dt
    #print "N = {}".format(n)
    while t <= T + 1E-9:
        
        w_up.t = t
    
        #print "Solving for t = {}".format(t) 
        
        # ------- Solving the Navier-Stokes equations -------
        #print "Solving the Navier-Stokes equations" 
        A = assemble(a0)     # I need to reassemble the system
        b = assemble(L0)
        
        # I need the reapply the BC to the new system
        for bc in bcu:
            bc.apply(A, b)
        
        # Ax = b, where VP_ is the x vector    
        solver.solve(A, VP_.vector(), b)
        
        # ------- Solving the Poisson problem -------
        #print "Solving the Poisson problem"
        A1 = assemble(a1)
        b1 = assemble(L1)
                        
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
        
        # REMEMBER: I can use VP_.split ONLY not plot, because in this way u0 and p0 don't know they are "part" of up0
        #u0, p0 = VP_.split()
        #plot(u0, title = str(t))
        #plot(w0)
        #plot(u_exact, interactive = True)
        #plot(mesh)
        t += dt
        
        
    u0, p0 = VP_.split()
    #plot(u0, title = "u_h")
    #plot(u_exact, title = "exact", mesh = mesh)
    #plot(p0)
    #interactive()
    #from IPython import embed; embed()
    L2_error_u = assemble((u_exact-u0)**2 * dx)**.5
    Linf_error = (interpolate(Expression(("x[1]*(1-x[1])", "0"), degree = 2), V).vector() - interpolate(u0, V).vector()).norm("linf")
    print "||u - uh; L^2|| = {0:1.4e}".format(L2_error_u)
    print "||u - uh; Linf|| = {0:1.4e}".format(Linf_error)
    print "-------------"
    