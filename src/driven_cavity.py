## DRIVEN CAVITY - test case ##

from dolfin import *

#N = [2**2, 2**3, 2**4, 2**5, 2**6]
N = [16]
dt = 0.01
#dt = 0.05
#dt = 0.025
#dt = 0.0125

T = 2.5
nu = 1.0/1000.0
rho = 1.0
theta = 1.0 
def epsilon(u):
    return 0.5*(grad(u)+grad(u).T)

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
    
    dudt = Constant(1.0/dt) * inner(u-u0, v) * dx
    a = Constant(nu) * 2*inner(epsilon(u_mid), epsilon(v)) * dx
    b = q * div(u_mid) * dx   # from continuity equation
    #c = inner(dot(grad(u_mid),u0), v) * dx
    c = inner(grad(u_mid)*u0, v) * dx
    # c = inner(grad(u0)*u0, v) * dx
    d = inner(p_mid, div(v)) * dx
    L = inner(f_mid,v)*dx
    #F = dudt + a + b + c + d - L
    F = dudt + a + b + c + d - L
    
    a0, L0 = lhs(F), rhs(F)
    
    
    
    t = dt
    
    U = Function(W)   # I want to store my solution here
    u, p = U.split()
    U.vector()[:] = 1.0
    for bc in bcu:
        bc.apply(U.vector())
    #plot(U[0], mode="color")
    ff = FacetFunction("size_t", mesh)
    ff.set_all(0)
    CompiledSubDomain("(x[1] > 1.0 - DOLFIN_EPS)&& on_boundary").mark(ff, 1)
    CompiledSubDomain("(x[1] < (1.0- DOLFIN_EPS))&& on_boundary").mark(ff, 2)
    print assemble(sqrt(dot(U[0],U[0]))*ds(1, domain=mesh, subdomain_data=ff))
    print assemble(sqrt(dot(U[0],U[0]))*ds(2, domain=mesh, subdomain_data=ff))
    
    #interactive()
    #exit()
    solver = PETScLUSolver()
    #solver = KrylovSolver("gmres", "ilu")
    #solver.parameters["relative_tolerance"] = 1e-10
    #solver.parameters["absolute_tolerance"] = 1e-10
    #solver.parameters["maximum_iterations"] = 1000
    #solver.parameters["monitor_convergence"] = True
    #solver.parameters["nonzero_initial_guess"] = True
    #ufile = File("velocity.pvd")
    while (t - T) <= DOLFIN_EPS :
        
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
    
    # V1 = U.function_space().sub(0).sub(0).collapse()  # this is V from the beginning
    V1 = FunctionSpace(mesh, "CG", 1)
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
    psifile = File("psi64.pvd")
    psifile << psi
    
    print "dt = {}".format(dt)
    print "N = {}".format(n)
    print "min of streamfunction = {}".format(min(psi.vector()))
    print "-------" 