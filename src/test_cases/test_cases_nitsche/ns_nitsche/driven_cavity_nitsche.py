## DRIVEN CAVITY - test case ##

from dolfin import *

#N = [2**2, 2**3, 2**4, 2**5, 2**6]
N = [2**7]
#dt = 0.1
#dt = 0.05
#dt = 0.025
dt = 0.0125

T = 2.5
nu = Constant(1.0/1000.0)
rho = 1.0
#theta = 1.0
theta = 0.5

gamma = Constant(10.0)
g1 = Constant((0.0, 0.0))

# When theta = 0.5, if I put u_mid instead of u in the continuity equation, the results go crazy

# On the FEniCS book it says u_mid so the system is symmetric [Eleonora]
# On all the other references I found it says u^{n+1}
print "***************"
print "* theta = {} *".format(theta)
print "***************"

def epsilon(u):
    return 0.5*(grad(u)+grad(u).T)

for n in N:

    mesh  = UnitSquareMesh(n,n, "crossed")
    normal = FacetNormal(mesh)
    h = CellSize(mesh)
    
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
    
    # -------------
    
    # Define boundaries
    fd = FacetFunction("size_t", mesh)
    CompiledSubDomain("near(x[0], 0.0) && on_boundary").mark(fd, 1) # left wall (cord)    
    CompiledSubDomain("near(x[0], 1.0) && on_boundary").mark(fd, 2) # right wall (tissue)  
    CompiledSubDomain("near(x[1], 1.0) && on_boundary").mark(fd, 3) # top wall (inlet)
    CompiledSubDomain("near(x[1], 0.0) || (near(x[0], 1.0) && near(x[1], 1.0)) && on_boundary").mark(fd, 4) # bottom wall (outlet)
    ds = Measure("ds", domain = mesh, subdomain_data = fd)
    #plot(fd)
    #interactive()
    
    # bcu = [DirichletBC(W.sub(0), Constant((1.0, 0.0)), fd, 3),
    #        DirichletBC(W.sub(0), Constant((0.0, 0.0)), fd, 4),
    #        DirichletBC(W.sub(0), Constant((0.0, 0.0)), fd, 1),
    #        DirichletBC(W.sub(0), Constant((0.0, 0.0)), fd, 2)]
    
    ## check the BC are correct
    #U = Function(W)
    #for bc in bcu: bc.apply(U.vector())
    #plot(U.sub(0))
    #interactive()
    
    
    bcu = [DirichletBC(W.sub(0), Constant((1.0, 0.0)), fd, 3),
           DirichletBC(W.sub(0), Constant((0.0, 0.0)), fd, 4),
           DirichletBC(W.sub(0), Constant((0.0, 0.0)), fd, 1)]
    
    # # check the BC are correct
    # U = Function(W)
    # for bc in bcu: bc.apply(U.vector())
    # plot(U.sub(0))
    # interactive()
    
    # -------------
    
    dudt = Constant(dt**-1) * inner(u-u0, v) * dx
    #a = Constant(nu) * 2.0*inner(epsilon(u_mid), epsilon(v)) * dx
    a = nu * inner(grad(u_mid), grad(v)) * dx
    
    a += inner(p_mid*normal, v)*ds(2) #+ inner(q*normal, u_mid)*ds(2) - inner(q*normal, g)*ds(2)
    a += - nu * inner(grad(u_mid)*normal, v)*ds(2) - nu * inner(grad(v)*normal, u_mid)*ds(2) + gamma * h**-1 * inner(u_mid,v)*ds(2)
    a += nu * inner(grad(v)*normal, g1)*ds(2) - gamma * h**-1 * inner(g1, v)*ds(2)
    
    
    # When theta = 0.5, if I put u_mid instead of u in the continuity equation, the results go crazy
    #----------------
    #b = q * div(u_mid) * dx   # from continuity equation
    #----------------

    b = q * div(u) * dx   # from continuity equation

    c = inner(grad(u_mid)*u0, v) * dx
    # c = inner(grad(u0)*u0, v) * dx
    d = inner(p_mid, div(v)) * dx
    L = inner(f_mid,v)*dx
    F = dudt + a + b + c + d - L
    
    a0, L0 = lhs(F), rhs(F)
    
    
    
    t = dt
    
    U = Function(W)   # I want to store my solution here
    u, p = U.split()

    # U.vector()[:] = 1.0
    # for bc in bcu:
    #     bc.apply(U.vector())
    # #plot(U[0], mode="color")
    # ff = FacetFunction("size_t", mesh)
    # ff.set_all(0)
    # CompiledSubDomain("(x[1] > 1.0 - DOLFIN_EPS)&& on_boundary").mark(ff, 1)
    # CompiledSubDomain("(x[1] < (1.0- DOLFIN_EPS))&& on_boundary").mark(ff, 2)
    # print assemble(sqrt(dot(U[0],U[0]))*ds(1, domain=mesh, subdomain_data=ff))
    # print assemble(sqrt(dot(U[0],U[0]))*ds(2, domain=mesh, subdomain_data=ff))
    
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
    
    #while (t - T) <= DOLFIN_EPS :
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
        
        # Move to next time step (in case I had separate equations)
        # u0.assign(u)
        # p0.assign(p)
        
        # I need to assign up0 because u0 and p0 are not "proper" functions
        up0.assign(U)   # the first part of U is u0, and the second part is p0
        #u0, p0 = U.split()
        #plot(u0, title = str(t))
        
        #ufile << up0.split()[0]
        
        t += dt
            
    #plot(u0)
    #interactive()
    
    # V1 = U.function_space().sub(0).sub(0).collapse()  # this is V from the beginning
    V1 = FunctionSpace(mesh, "CG", 1)
    psi = TrialFunction(V1)
    q1 = TestFunction(V1)
    a1 = dot(grad(psi), grad(q1))*dx
    #L = curl(u) * q1 * dx
    L1 = dot(u[1].dx(0) - u[0].dx(1), q1)*dx
    
    g = Constant(0.0)
    bc_psi = DirichletBC(V1, g, "on_boundary")
    psi = Function(V1)
    solve(a1 == L1, psi, bc_psi)
    
    #plot(psi)
    #interactive()
    
    print "t_final = {}".format(t - dt)
    print "dt = {}".format(dt)
    print "N = {}".format(n)
    print "min of streamfunction = {}".format(min(psi.vector()))
    print "-------"
    
    
#if n == 64:
#   print "N = 64"
psifile = File("results/psi64_0.0125.pvd")
psifile << psi