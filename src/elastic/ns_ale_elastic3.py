## ALE + NAVIER-STOKES EQUATIONS ##

# rho * du/dt + rho * (grad(u) . u - w) - div( mu * grad(u) - pI ) = f
# div( u ) = 0

from dolfin import *

#N = [2**2, 2**3, 2**4, 2**5, 2**6]
N = [2**5]
T = 1.5
mu = 1.0
rho = 1.0
theta = 1.0     # 0.5 for Crank-Nicolson, 1.0 for backwards
gamma = 1e2    # constant for Nitsche method
k = 1e4      # elastic constant
dt = 0.01
g = Constant((0.0,0.0))
T = dt

for n in N : 
   
    mesh = UnitSquareMesh(n, n)  
    x = SpatialCoordinate(mesh)
    normal = FacetNormal(mesh)
    
    h = CellSize(mesh)
    n = FacetNormal(mesh)
    
    # Taylor-Hood elements
    V = VectorFunctionSpace(mesh, "CG", 2)  # space for u, v
    P = FunctionSpace(mesh, "CG", 1)        # space for p, q
    W = VectorFunctionSpace(mesh, "CG", 1)       # space for w
    VP = V * P                  
    
    u, p = TrialFunctions(VP)   # u is a trial function of V, while p a trial function of P
    w = TrialFunction(W)
    
    v, q = TestFunctions(VP)
    z = TestFunction(W)
    
    # Defining the normal and tangential components    
    un = dot(u,n) * n
    vn = dot(v,n) * n
    ut = u - un
    vt = v - vn
    
    # The functions are initialized to zero
    up0 = Function(VP)
    u0, p0 = split(up0)  # u0 is not a function but "part" of a function, just a "symbolic" splitting?
                         # REMEMBER: In this way when I update the up0 also the u0 and p0 get updated
                         # This is different from up0.split which I can use if I want to plot
    w0 = Function(W)
    
    X = Function(W)  # in here I will put the displacement X^(n+1) = X^n + dt*(w^n)
    Y = Function(W)  # by default this is 0 in all the components, Y is the adding displacement (what we add in order to move the mesh)
    
    f0 = Constant((0.0, 0.0))
    f = Constant((0.0, 0.0))
    u_inlet = Expression(("0.0", "-1*fabs(x[0]*(x[0] - 1))"), degree = 2)
    
    u_mid = (1.0-theta)*u0 + theta*u
    f_mid = (1.0-theta)*f0 + theta*f
    p_mid = (1.0-theta)*p0 + theta*p
    
        
    # Define boundary conditions
    fd = FacetFunction("size_t", mesh)
    CompiledSubDomain("near(x[0], 0.0)").mark(fd, 1) # left wall (cord)     PHYSICAL BOUNDARY --> here the values of w and u have to be the same
    CompiledSubDomain("near(x[0], 1.0)").mark(fd, 2) # right wall (tissue)  PHYSICAL BOUNDARY --> here the values of w and u have to be the same
    CompiledSubDomain("near(x[1], 1.0)").mark(fd, 3) # top wall (inlet)
    CompiledSubDomain("near(x[1], 0.0)").mark(fd, 4) # bottom wall (outlet)
    ds = Measure("ds", domain = mesh, subdomain_data = fd)
    
    #plot(fd)
    #interactive()
    
    # Here I need to impose just the Dirichlet conditions. The ones regarding the stresses were already encountered in the
    # weak formulation
    bcu = [DirichletBC(VP.sub(0), u_inlet, fd, 3),   # inlet at the topo wall
           DirichletBC(VP.sub(0), Constant((0.0,0.0)), fd, 1)]   # left wall
    bcw = [DirichletBC(W, Constant((0.0,0.0)), fd, 1),
                DirichletBC(W, Constant((0.0,0.0)), fd, 3),
                DirichletBC(W, Constant((0.0,0.0)), fd, 4),
            #    DirichletBC(W, dot(u0,unit)*unit, fd, 2)]
                DirichletBC(W, u0, fd, 2)]  # PHYSICAL BOUNDARY --> here the values of w^(k+1) and u^(k+1) have to be the same


    
    # check the BC are correct
    #U = Function(VP)
    #for bc in bcu: bc.apply(U.vector())
    #plot(U.sub(0))
    #interactive()
    

    #-------- NAVIER-STOKES --------
    # Weak formulation
    dudt = Constant(1./dt) * Constant(rho) * inner(u - u0, v) * dx
    a = ( Constant(rho) * inner(grad(u_mid)*(u0 - w0), v)   # ALE term
         + Constant(mu) * inner(grad(u_mid), grad(v))
         - p * div(v)                               # CHECK THIS TERM
         - q * div(u)                               # from the continuity equation, maybe put a - q*div(u) to make the system symmetric
         - inner(f,v) ) * dx
    
    # Boundary term with elastic constant
    b = Constant(k) * inner(dot(X + Constant(dt)*u_mid, n ) * n, v) * ds(2)    # what should I use here as displacement?
    
    # b = inner(Constant(k) * dot(X+Constant(dt)*u, normal) * normal, v) * ds(2)    # what should I use here as displacement?
    c = ( -inner(dot(grad(ut), n), vt) - inner(dot(grad(vt), n), ut) + Constant(gamma)/h * inner(ut,vt)
            - inner(dot(grad(vt),n), g) + Constant(gamma)/h * inner(g,vt) ) * ds(2)
                                                                                                
        
    # Bilinear and linear forms
    # THE SIGNS NOW SHOULD BE CORRECT!! it's +b and not -b
    F = dudt + a + b + c
    a0, L0 = lhs(F), rhs(F)    
    
    # -------- POISSON PROBLEM FOR w:  div( grad(w) ) = 0 --------
    a1 = inner(grad(w), grad(z))*dx
    L1 = dot(Constant((0.0,0.0)),z)*dx   
    # ----------------------
    
    # I want to store my solutions here
    VP_ = Function(VP)   
    W_ = Function(W)
    

    solver = PETScLUSolver()
    t = 0.0
    while t <= T + 1E-9:
    
        print "Solving for t = {}".format(t) 
        #tissue = DirichletBC(W, up0.sub(0), "near(x[0], 1.0, 0.1) && on_boundary")   
        #bcw = [tissue, fixed]
        
        # Solving the Navier-Stokes equations
        # I need to reassemble the system
        A = assemble(a0)
        b = assemble(L0)
        
        # I need the reapply the BC to the new system
        for bc in bcu:
            bc.apply(A, b)
        
        # Ax = b, where U is the x vector    
        solver.solve(A, VP_.vector(), b)
        
        # Solving the Poisson problem
        A1 = assemble(a1)
        b1 = assemble(L1)
         
        #bcw[3] = DirichletBC(W, dot(u0, unit) * unit, fd, 2)   # updating the boundary value u0
        bcw[3] = DirichletBC(W, u0, fd, 2)   # updating the boundary value u0
        
        
        for bc in bcw:
           bc.apply(A1, b1)
        
        solve(A1, W_.vector(), b1)

        up0.assign(VP_)   # the first part of VP_ is u0, and the second part is p0
        w0.assign(W_)
        
        
        # Compute the mesh displacement
        Y.vector()[:] = w0.vector()[:]*dt
        X.vector()[:] += Y.vector()[:]
        
        # Move the mesh
        ALE.move(mesh, Y)
        mesh.bounding_box_tree().build(mesh)
        
        # I need to assign up0 because u0 and p0 are not "proper" functions
        # plot(u0)
        #plot(mesh)
        
        #u0, p0 = VP_.split()
        #plot(u0)
        
        normal = FacetNormal(mesh)
        
        t += dt
        #break
plot(u0)
interactive()
