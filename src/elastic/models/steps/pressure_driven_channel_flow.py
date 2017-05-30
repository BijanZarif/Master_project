## NAVIER-STOKES EQUATIONS ##

# rho * du/dt + rho * (grad(u) . u) - div( nu * grad(u) - pI ) = f
# div( u ) = 0

from dolfin import *

#NN = [2**2, 2**3, 2**4, 2**5, 2**6]
NN = [2**4]
#dt = 0.1
#dt = 0.05
dt = 0.025
#dt = 0.0125

# Dimensions of the model: bottom wall is 4 mm, tissue wall is 60 mm
x0, x1 = 0.0, 4   # [mm]
y0, y1 = 0.0, 60  # [mm]


for N in NN :
    
    mesh = RectangleMesh(Point(x0, y0), Point(x1, y1), N, N, "crossed")  
    #mesh = UnitSquareMesh(N, N, "crossed")  
    x = SpatialCoordinate(mesh)
    
    # Taylor-Hood elements
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    # TH = V * Q
    W = V * Q
    # W = FunctionSpace(mesh, TH)
    
    u, p = TrialFunctions(W)   # u is a trial function of V somehow, while p a trial function of Q
    v, q = TestFunctions(W)
    
    up0 = Function(W)
    u0, p0 = split(up0)  # u0 is not a function but "part" of a function, just a "symbolic" splitting?
    
    # u0 = Function(V)   # it starts to zero
    # p0 = Function(Q)   # it starts to zero
    
    
    T = 10.0
    mu = 0.700e-3  # [g/(mm * s)]
    rho = 1e-3     # [g/mm^3] 
    
    # OLD VALUES IN THE TEST CASE
    # mu = 1.0/8.0
    # rho = 1.0
    
    theta = 0.5 
    f0 = Constant((0.0, 0.0))
    f = Constant((0.0, 0.0))
    #g = Constant((0.0, 0.0))
    t = dt
    
    
    # oscillating
    amplitude_in = Constant(15.0)
    p_in = Expression("a*sin(2*pi*t)", a=amplitude_in, t=t, degree=2)
    
    #p_in  = Constant(15.0)
    p_out = Constant(0.0) 
    
    u_mid = (1.0-theta)*u0 + theta*u
    f_mid = (1.0-theta)*f0 + theta*f
    p_mid = (1.0-theta)*p0 + theta*p
    
    # Define boundary conditions
    fd = FacetFunction("size_t", mesh)
    CompiledSubDomain("near(x[0], x0) && on_boundary", x0 = x0).mark(fd, 1) # left wall (cord)  
    CompiledSubDomain("near(x[0], x1) && on_boundary", x1 = x1).mark(fd, 2) # right wall (tissue) 
    #CompiledSubDomain("near(x[1], y1) && (x[0] != x1) && on_boundary", x1 = x1, y1 = y1).mark(fd, 3) # top wall
    #CompiledSubDomain("near(x[1], y0) && (x[0] != x1) && on_boundary", x1 = x1, y0 = y0).mark(fd, 4) # bottom wall 
    CompiledSubDomain("near(x[1], y1) ||(near(x[0], x1) && near(x[1], y1) ) && on_boundary", x1 = x1, y1 = y1).mark(fd, 3) # top wall 
    CompiledSubDomain("near(x[1], y0) ||(near(x[0], x1) && near(x[1], y0) ) && on_boundary", x1 = x1, y0 = y0).mark(fd, 4) # bottom wall 

    ds = Measure("ds", domain = mesh, subdomain_data = fd)
    #plot(fd)
    #interactive()
    
    top_wall = DirichletBC(W.sub(1), p_in, fd, 3)   # top wall
    btm_wall = DirichletBC(W.sub(1), p_out, fd, 4)   # btm wall
    left_wall = DirichletBC(W.sub(0), Constant((0.0, 0.0)) , fd, 1) # left
    right_wall = DirichletBC(W.sub(0), Constant((0.0, 0.0)) , fd, 2) # right
    
    # #this is to verify that I am actually applying some BC
    # U = Function(W)
    # # this applies BC to a vector, where U is a function
    # inflow.apply(U.vector())
    # outflow.apply(U.vector())
    # walls.apply(U.vector())
    # 
    # uh, ph = U.split()
    # plot(uh)
    # plot(ph)
    # interactive()
    # exit()
    
    # Neumann condition: sigma.n = 0        on the sides (I put in the variational form and the term on the sides is zero)
    
    bcu = [left_wall, right_wall, top_wall, btm_wall]
    
    # F0 = rho*u*v*dx - rho*u0*v*dx + \
    #      dt*rho*dot(grad(u_mid), u0)*v*dx + dt*v*dot(grad(u_mid), grad(v)) - dt*p_mid*div(v)*dx - dt*f_mid*v*dx
    
    # dot JUST FOR 1-1 RANK
    # inner FOR GREATER RANKS
    
    # Ovind suggested: divide your form in different passages so if you have an error you are going to see exactly
    # what part of the form gives the error
    
    dudt = Constant(dt**-1) * inner(u-u0, v) * dx
    a = Constant(mu) * inner(grad(u_mid), grad(v)) * dx
    b = q * div(u) * dx
    c = inner(grad(u_mid)*u0, v) * dx
    d = inner(grad(p), v) * dx
    L = inner(f_mid,v)*dx
    F = dudt + a + b + c + d - L
    
    a0, L0 = lhs(F), rhs(F)
    
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
    
    
    
    U = Function(W)   # I want to store my solution here
    solver = PETScLUSolver()
    
    
    print "N = {}".format(N)
    #while (t - T) <= DOLFIN_EPS:
    while t <= T + 1E-9:
        
         
        print "solving for t = {}".format(t) 
        
        # I need to reassemble the system
        A = assemble(a0)
        b = assemble(L0)
        
        # I need the reapply the BC to the new system
        bcu = [left_wall, right_wall, top_wall, btm_wall]
        for bc in bcu:
            bc.apply(A, b)
        
        # Ax=b, where U is the x vector    
        solver.solve(A, U.vector(), b)
        
        # Move to next time step (in case I had separate equations)
        # u0.assign(u)
        # p0.assign(p)
        
        # I need to assign up0 because u0 and p0 are not "proper" functions
        up0.assign(U)   # the first part of U is u0, and the second part is p0
        
        u01, p01 = U.split()
        plot(u01, key="u01", title = str("velocity at time t= ") + str(t))
        plot(p01, key='p01', title = str("pressure at time t= ") + str(t))
        
        
        t += dt
        p_in.t = t
        
    print "t_final = {}".format(t - dt)    
    print "dt = {}".format(dt)   
    #print "T = {}".format(t)
    #print "u(1, 0.5, t = 0.5) = {}".format(U(Point(1, 0.5))[0])
    print("------")    
    
    #plot(u0)
    #plot(p0)
    interactive()
    
    #u0, p0 = U.split()
    #ufile = File("velocity_64_0.0125.pvd")
    #ufile << u0
    