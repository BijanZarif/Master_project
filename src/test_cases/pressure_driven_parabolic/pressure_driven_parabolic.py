## NAVIER-STOKES EQUATIONS ##

# rho * du/dt + rho * (grad(u) . u) - div( nu * grad(u) - pI ) = f
# div( u ) = 0

from matplotlib import pyplot as plt
from dolfin import *

#NN = [2**2, 2**3, 2**4, 2**5, 2**6]
NN = [2**5]
#h = [1./i for i in NN]
h = [1./NN[0]]
errsL2_velocity = []
errsH1_velocity = []
errsL2_pressure = []

x0, x1 = 0.00, 1.00   # [mm]
y0, y1 = 0.00, 1.00  # [mm]


for N in NN:
    
    for dt in h:
        print "******"
        print "N = {}".format(N)
        print "dt = {}".format(dt)
    
        mesh = RectangleMesh(Point(x0, y0), Point(x1, y1), N, N, "crossed")  # crossed means the triangles are divided in 2
        x = SpatialCoordinate(mesh)
        n = FacetNormal(mesh)
        
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
        
        
        T = 15 
        mu = 1.0/8.0  # [g/(mm * s)]
        rho = 1     # [g/mm^3] 
        theta = 1 
        f0 = Constant((0.0, 0.0))
        f = Constant((0.0, 0.0))
        
        u_exact = as_vector(( 1.0/(2*mu) * x[1]*(1-x[1]) , 0))
        p_exact = 1 - x[0]
        
        ufile = File("results_parabolic/velocity.pvd")
        pfile = File("results_parabolic/pressure.pvd")
        
        p_in  = Constant(1.0)
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
    
        
        left_wall = DirichletBC(W.sub(0).sub(1), Constant(0.0), fd, 1)   # left wall - I set the y (tangential) component of the velocity to zero
        right_wall = DirichletBC(W.sub(0).sub(1), Constant(0.0), fd, 2)   # right wall - I set the y (tangential) component of the velocity to zero
        top_wall = DirichletBC(W.sub(0), Constant((0.0, 0.0)) , fd, 3) # top
        bottom_wall = DirichletBC(W.sub(0), Constant((0.0, 0.0)) , fd, 4) # bottom
        # Neumann condition: sigma.n = 1        on the inlet
        # Neumann condition: sigma.n = 0        on the outlet
        
        bcu = [left_wall, right_wall, top_wall, bottom_wall]
        
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
        
        #while (t - T) <= DOLFIN_EPS:
        while t <= T + 1E-9:
             
            print "solving for t = {}".format(t)
    
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
            ufile << u0
            pfile << p0
                
            t += dt
            
        # These are the errors for a certain dt
        L2_error_u = assemble((u_exact-u0)**2 * dx)**.5
        H1_error_u = assemble(grad(u0-u_exact)**2 * dx)**.5
        L2_error_p = assemble((p_exact - p0)**2 * dx)**.5
        
        # print "||u - uh; L^2|| = {0:1.4e}".format(L2_error_u)
        # print "|u - uh; H^1| = {0:1.4e}".format(H1_error_u)
        # print "||p - ph; L^2|| = {0:1.4e}".format(L2_error_p)
        
        # errL2_velocity and the others are vectors that contain the error for all dt
        errsL2_velocity.append(L2_error_u)
        errsH1_velocity.append(H1_error_u)
        errsL2_pressure.append(L2_error_p)
        
        print "t_final = {}".format(t - dt)
        
print errsL2_velocity
print errsL2_pressure
print errsH1_velocity

