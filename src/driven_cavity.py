## DRIVEN CAVITY - test case ##

from dolfin import *

#N = [2**2, 2**3, 2**4, 2**5, 2**6]
N = [128]
dt = 0.1
#dt = 0.05
#dt = 0.025
#dt = 0.0125

T = 2.5
nu = 1.0/1000.0
rho = 1.0
theta = 0.5 


for n in N:

    mesh  = UnitSquareMesh(n,n, "crossed")
    
    #plot(mesh)
    #interactive()
    
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = V * Q
    
    u, p = TrialFunctions(W)   # u is a trial function of V, while p a trial function of Q
    v, q = TestFunctions(W)
    
    up0 = Function(W)
    u0, p0 = split(up0)  # u0 is not a function but "part" of a function, just a "symbolic" splitting?
    

    f0 = Constant((0.0, 0.0))
    f = Constant((0.0, 0.0))
    p0 = Constant(0.0)
    
    u_mid = (1.0-theta)*u0 + theta*u
    f_mid = (1.0-theta)*f0 + theta*f
    p_mid = (1.0-theta)*p0 + theta*p
    
    # The BC should be correct
    up = DirichletBC(W.sub(0), (1.0, 0.0), "(x[1] > 1.0 - DOLFIN_EPS)&& on_boundary" )
    walls = DirichletBC(W.sub(0), (0.0, 0.0), "(x[1] < (1.0- DOLFIN_EPS))&& on_boundary" )
    
    
    # # this is to verify that I am actually applying some BC
    # U = Function(W)
    # # this applies BC to a vector, where U is a function
    # up.apply(U.vector())
    # walls.apply(U.vector())
    # 
    # uh, ph = U.split()
    # plot(uh)
    # interactive()
    # exit()
    
    bcu = [up, walls]
    
    dudt = Constant(dt**-1) * inner(u-u0, v) * dx
    a = Constant(nu) * inner(grad(u_mid), grad(v)) * dx
    b = q * div(u) * dx   # from continuity equation
    c = inner(grad(u_mid)*u0, v) * dx
    d = inner(p, div(v)) * dx
    L = inner(f_mid,v)*dx
    F = dudt + a + b + c + d - L
    
    a0, L0 = lhs(F), rhs(F)
    
    
    
    t = dt
    
    U = Function(W)   # I want to store my solution here
    u, p = U.split()
    solver = PETScLUSolver()
    #ufile = File("velocity.pvd")
    while t <= T:
        
        print "solving for t = {}".format(t)
        
        # I need to reassemble the system
        A = assemble(a0)
        b = assemble(L0)
        
        # I need the reapply the BC to the new system
        for bc in bcu:
            bc.apply(A, b)
        
        # Ax=b, where U is the x vector    
        solver.solve(A, U.vector(), b)
        
        # Move to next time step (in case I had separate equations)
        # u0.assign(u)
        # p0.assign(p)
        
        # I need to assign up0 because u0 and p0 are not "proper" functions
        up0.assign(U)   # the first part of U is u0, and the second part is p0
        
        #ufile << up0.split()[0]
        
        t += dt
            
        
    #plot(u0)
    #interactive()
    
    V1 = U.function_space().sub(0).sub(0).collapse()  # this is V from the beginning
    
    psi = TrialFunction(V1)
    q1 = TestFunction(V1)
    a = dot(grad(psi), grad(q1))*dx
    #L = curl(u) * q1 * dx
    L = dot(u[1].dx(0) - u[0].dx(1), q1)*dx
    
    g = Constant(0.0)
    bc_psi = DirichletBC(V1, g, "on_boundary")
    psi = Function(V1)
    solve(a == L, psi, bc_psi)
    
    #plot(psi)
    #interactive()
    
    print "dt = {}".format(dt)
    print "N = {}".format(n)
    print "min of streamfunction = {}".format(min(psi.vector()))
    print "-------"