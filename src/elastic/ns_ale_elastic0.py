## ALE + NAVIER-STOKES EQUATIONS ##
# This file is to "prepare" for the elastic version, for now it's just ALE

# rho * du/dt + rho * (grad(u) . u - w) - div( nu * grad(u) - pI ) = f
# div( u ) = 0

from dolfin import *

#N = [2**2, 2**3, 2**4, 2**5, 2**6]
N = [2**3]
T = 1.5
nu = 1.0
rho = 1.0
k = 1.0         # elastic constant
theta = 0.5     # 0.5 for Crank-Nicolson, 1.0 for backwards

dt = 0.01
#dt = 0.05
#dt = 0.025
#dt = 0.0125

for n in N : 
   
    mesh = UnitSquareMesh(n, n)  
    x = SpatialCoordinate(mesh)
    
    # Taylor-Hood elements
    V = VectorFunctionSpace(mesh, "CG", 2)  # space for u, v
    P = FunctionSpace(mesh, "CG", 1)        # space for p, q
    W = VectorFunctionSpace(mesh, "CG", 1)       # space for w
    VP = V * P                  
    
    u, p = TrialFunctions(VP)   # u is a trial function of V, while p a trial function of P
    w = TrialFunction(W)
    
    v, q = TestFunctions(VP)
    z = TestFunction(W)
    
    # The functions are initialized to zero
    up0 = Function(VP)
    u0, p0 = split(up0)  # u0 is not a function but "part" of a function, just a "symbolic" splitting?
                         # REMEMBER: In this way when I update the up0 also the u0 and p0 get updated
                         # This is different from up0.split which I can use if I want to plot
    w0 = Function(W)
    
    f0 = Constant((0.0, 0.0))
    f = Constant((0.0, 0.0))
    
    p_in  = Constant(1.0)
    p_out = Constant(0.0) 
    
    u_mid = (1.0-theta)*u0 + theta*u
    f_mid = (1.0-theta)*f0 + theta*f
    p_mid = (1.0-theta)*p0 + theta*p
    
    # ------- Boundary conditions for Navier-Stokes --------
    w_up = Expression(("0", "-2*cos(4*pi*t)*x[0]*(x[0] - 1)"), t = 0.5)  # I need the 4*pi so it does multiply cycles
    
    inflow = DirichletBC(VP.sub(1), p_in, "near(x[0], 0.0) && on_boundary" )
    outflow = DirichletBC(VP.sub(1), p_out, "near(x[0], 1.0) && on_boundary" )
    upper_wall =  DirichletBC(VP.sub(0), w_up, "near(x[1], 1.0) && on_boundary")
    lower_wall = DirichletBC(VP.sub(0), (0.0, 0.0) , "near(x[1], 0.0) && on_boundary")
    
    # ------- Boundary conditions for Poisson -------
    up = DirichletBC(W, w_up, "near(x[1], 1.0) && on_boundary")   
    contour = DirichletBC(W, (0.0, 0.0), " ( near(x[0], 0.0) || near(x[0], 1.0) || near(x[1], 0.0 )) \
                          || (near(x[0], 0.0) && near(x[1], 0.0)) \
                          || (near(x[0], 1.0) && near(x[1], 1.0) ) \
                          && on_boundary")
    
    bcu = [inflow, outflow, upper_wall, lower_wall]
    bcw = [up, contour]
    
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
    
    #-------- NAVIER-STOKES --------
    # Weak formulation
    dudt = Constant(1./dt) * inner(u - u0, v) * dx
    a = ( Constant(rho) * inner(grad(u_mid)*(u0 - w0), v)   # ALE term
         + Constant(nu) * inner(grad(u_mid), grad(v))
         - p * div(v)                               # CHECK THIS TERM
         + q * div(u)                               # from the continuity equation, maybe put a - q*div(u) to make the system symmetric
         - inner(f,v) ) * dx

    
    # Bilinear and linear forms
    F = dudt + a
    a0, L0 = lhs(F), rhs(F)    
    
    # -------- POISSON PROBLEM FOR w:  div( grad(w) ) = 0 --------
    a1 = inner(grad(w), grad(z))*dx
    L1 = dot(Constant((0.0,0.0)),z)*dx   
    # ----------------------
    
    # I want to store my solutions here
    VP_ = Function(VP)   
    W_ = Function(W)
    
    # Y is the adding displacement (what we add in order to move the mesh)
    Y = Function(W)  # by default this is 0 in all the components
    X = Function(W)  # in here I will put the displacement X^(n+1) = X^n + dt*(w^n)

    solver = PETScLUSolver()
    t = dt
    while t <= T + 1E-9:
        
        w_up.t = t
        print "Solving for t = {}".format(t) 
        
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
        
        for bc in bcw:
           bc.apply(A1, b1)
        
        solve(A1, W_.vector(), b1)
        
        # Compute the mesh displacement
        Y.vector()[:] = w0.vector()[:]*dt
        X.vector()[:] += Y.vector()[:]
        
        # Move the mesh
        ALE.move(mesh, Y)
        mesh.bounding_box_tree().build(mesh)
        
        # -------- Update of solutions so the cycle can start again --------
        # I need to assign up0 because u0 and p0 are not "proper" functions
        up0.assign(VP_)   # the first part of VP_ is u0, and the second part is p0
        w0.assign(W_)
        
        #u0, p0 = VP_.split()
        #plot(u0)
        plot(mesh)
        
        t += dt
