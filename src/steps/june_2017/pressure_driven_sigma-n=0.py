## NAVIER-STOKES EQUATIONS ##

# rho * du/dt + rho * (grad(u) . u) - div( nu * grad(u) - pI ) = f
# div( u ) = 0

from matplotlib import pyplot as plt
from dolfin import *

#N = [2**2, 2**3, 2**4, 2**5, 2**6]
N = [2**3]
#dt = 0.1
#dt = 0.05
dt = 0.025
#dt = 0.0125

x0, x1 = 0.0, 60.0   # [mm]
y0, y1 = 0.0, 4.0  # [mm]

for n in N :
    
    mesh = RectangleMesh(Point(x0, y0), Point(x1, y1), 10*n, n, "crossed")  # crossed means the triangles are divided in 2
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
    # TH = V * Q
    W = V * Q
    # W = FunctionSpace(mesh, TH)
    
    u, p = TrialFunctions(W)   # u is a trial function of V somehow, while p a trial function of Q
    v, q = TestFunctions(W)
    
    up0 = Function(W)
    u0, p0 = split(up0)  # u0 is not a function but "part" of a function, just a "symbolic" splitting?
    
    # T = 5 
    T = 5      # only for oscillating p_inlet
    mu = 0.700e-3  # [g/(mm * s)]
    rho = 1e-3     # [g/mm^3] 
    theta = 0.5 
    f0 = Constant((0.0, 0.0))
    f = Constant((0.0, 0.0))
    
    ufile = File("results/v_sigma-n=0_8_0.025.pvd")
    pfile = File("results/p_sigma-n=0_8_0.025.pvd")
    
    #p_in  = Constant(1.0)
    amplitude = Constant(12.0)
    #amplitude = Constant(6.0)  # [kPa]
    p_in = Expression("a*sin(2*pi*t)", a=amplitude, t=0.0, degree=2)   # only for oscillating p_inlet
    p_out = Constant(0.0) 
    
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
    
    left_wall = DirichletBC(W.sub(0).sub(1), Constant(0.0), fd, 1)   # left wall - I set the y (tangential) component of the velocity to zero
    right_wall = DirichletBC(W.sub(0).sub(1), Constant(0.0), fd, 2)   # right wall - I set the y (tangential) component of the velocity to zero
    #top_wall = DirichletBC(W.sub(0), Constant((0.0, 0.0)) , fd, 3) # top
    bottom_wall = DirichletBC(W.sub(0), Constant((0.0, 0.0)) , fd, 4) # bottom
    # Neumann condition: sigma.n = 1        on the inlet
    # Neumann condition: sigma.n = 0        on the outlet
    
    bcu = [left_wall, right_wall, bottom_wall]
    
    F = rho * Constant(dt**-1) * inner(u-u0, v) * dx
    F += rho * inner(grad(u_mid)*u0, v) * dx
    F += Constant(mu) * inner(grad(u_mid), grad(v)) * dx
    F -= p * div(v) * dx
    F -= q * div(u) * dx
    F += inner(p_in * n, v)*ds(1)
    #F -= inner(p_in * n, v)*ds(2)
    F += inner(p_out * n, v)*ds(2)
    F -= inner(f_mid,v)*dx
    
    a0, L0 = lhs(F), rhs(F)
    
    t = 0.0
    
    U = Function(W)   # I want to store my solution here
    solver = PETScLUSolver()
    
    
    print "N = {}".format(n)
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
        
        # I need to assign up0 because u0 and p0 are not "proper" functions
        up0.assign(U)   # the first part of U is u0, and the second part is p0
        
        u0, p0 = up0.split() 
        #plot(u0, key="u0", title = str("velocity at time t= ") + str(t))
        #plot(p0, key="p0", title = str("pressure at time t= ") + str(t))
        #interactive()
        
        ufile << u0
        pfile << p0
        
        t += dt
        p_in.t = t    # only for oscillating p_inlet
        #values_x0.append((U(x_0)[0]**2 + U(x_0)[1]**2)**0.5)
        #values_x1.append((U(x_1)[0]**2 + U(x_1)[1]**2)**0.5)
        #values_x2.append((U(x_2)[0]**2 + U(x_2)[1]**2)**0.5)
        #print values
        #print time
        
        
    print "t_final = {}".format(t - dt)    
    print "dt = {}".format(dt)   
    #print "T = {}".format(t)
    print("------")
    
    #print time
    #print values_x0
    #print values_x1
    #print values_x2
    #plt.figure()
    #plt.plot(time, values_x0, time, values_x1, time, values_x2, label = "Velocity in the point over time")
    #plt.plot(time, values_x1, label = "Velocity in the point over time")
    #plt.plot(time, values_x2, label = "Velocity in the point over time")
    #plt.show()
    
    
    #plot(u0)
    #plot(p0)
    #interactive()
    