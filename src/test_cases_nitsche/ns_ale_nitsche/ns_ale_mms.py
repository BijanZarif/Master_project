from dolfin import *
set_log_level(50)
#N = [(2**n, 0.5**(2*n)) for n in range(1, 5)]

N = [2**4, 2**5, 2**6, 2**7]
# N = [2**5]

T = 1.0
DT = [1./N[i] for i in range(len(N))]

rho = 1.0
mu = 1.0/8.0
theta = 1.0
t = Constant(0.0)
C = 0.1

def sigma(u,p):
    return mu*grad(u) - p*Identity(2)

for dt in DT:
    for n in N:
        
        mesh = UnitSquareMesh(n, n)
        x = SpatialCoordinate(mesh)
        normal = FacetNormal(mesh)
        
        ##EP: Why do you use Expression? It is better to use directly UFL
        
        u_exact_e = Expression(("sin(2*pi*x[1])*cos(2*pi*x[0])*cos(t)", "-sin(2*pi*x[0])*cos(2*pi*x[1])*cos(t)"), t = t)
        w_exact_e = Expression(("C*sin(2*pi*x[1])*cos(t)", "0.0"), C=C, t = t)
        # w_exact_e = Expression(("0.0", "0.0"))
        p_exact_e = Expression("cos(x[0])*cos(x[1])*cos(t)", t = t)
        
        #Write exact solution using UFL
        u_exact = as_vector(( sin(2*pi*x[1])*cos(2*pi*x[0])*cos(t) , -sin(2*pi*x[0])*cos(2*pi*x[1])*cos(t) ))
        w_exact = as_vector(( C*sin(2*pi*x[1])*cos(t) , 0.0))
        # w_exact = as_vector((0.0, 0.0))
        p_exact = cos(x[0])*cos(x[1])*cos(t)
        
        #It is better to use diff(u_exact, t)
        dudt = diff(u_exact, t)
        #dudt =  as_vector(( -sin(2*pi*x[1])*cos(2*pi*x[0])*sin(t) , sin(2*pi*x[0])*cos(2*pi*x[1])*sin(t) ))
        f = rho * dudt + rho * grad(u_exact)*(u_exact - w_exact) - div(mu*grad(u_exact) - p_exact*Identity(2))
        #print "dudt = {}".format(dudt(1))
        
        
        #exit()
        
        # Taylor-Hood elements
        V = VectorFunctionSpace(mesh, "CG", 2)  # space for u, v
        Ve = VectorFunctionSpace(mesh, "CG", 4) # space to interpolate exact solution
        P = FunctionSpace(mesh, "CG", 1)        # space for p, q
        W = VectorFunctionSpace(mesh, "CG", 1)       # space for w
        VP = V * P                  
        
        u, p = TrialFunctions(VP)   # u is a trial function of V, while p a trial function of P
        w = TrialFunction(W)
        
        v, q = TestFunctions(VP)
        z = TestFunction(W)
        
        up0 = Function(VP)
        u0, p0 = split(up0)
        w0 = Function(W)
        
        u_exact_int = interpolate(u_exact_e, V)
        assign(up0.sub(0), u_exact_int) # I want to start with u0 = u_exact_e as initial condition
        
        X = Function(W)  # in here I will put the displacement X^(n+1) = X^n + dt*(w^n)
        Y = Function(W)
        
        # I want to store my solutions here
        VP_ = Function(VP)   
        W_ = Function(W)
        
        ## This is WRONG: f0 is not zero. If you use implicit Euler nothing changes.
        ## Change it if you want to use CN
        f0 = Constant((0.0,0.0))
        u_mid = (1.0-theta)*u0 + theta*u
        f_mid = (1.0-theta)*f0 + theta*f

        # Define boundary conditions
        fd = FacetFunction("size_t", mesh)
        CompiledSubDomain("near(x[0], 0.0) && on_boundary").mark(fd, 1) # left wall (cord)    
        CompiledSubDomain("near(x[0], 1.0) && on_boundary").mark(fd, 2) # right wall (tissue)  
        # CompiledSubDomain("near(x[1], 1.0) ||( near(x[0], 1.0) && near(x[1], 1.0) ) && on_boundary").mark(fd, 3) # top wall (inlet)
        # CompiledSubDomain("near(x[1], 0.0) ||( near(x[0], 1.0) && near(x[1], 0.0) ) && on_boundary").mark(fd, 4) # bottom wall (outlet)
        CompiledSubDomain("near(x[1], 1.0)").mark(fd, 3) # top wall (inlet)
        CompiledSubDomain("near(x[1], 0.0)").mark(fd, 4) # bottom wall (outlet)

        ds = Measure("ds", domain = mesh, subdomain_data = fd)
        
        bcu = [DirichletBC(VP.sub(0), u_exact_e, fd, 1),
               # DirichletBC(VP.sub(0), u_exact_e, fd, 2),
               DirichletBC(VP.sub(0), u_exact_e, fd, 3),
               DirichletBC(VP.sub(0), u_exact_e, fd, 4)]
        
        bcw = [DirichletBC(W, w_exact_e, "on_boundary")]
    
        F = Constant(1./dt) * rho * inner(u - u0, v) * dx
        F += rho * inner(grad(u_mid)*(u0 - w0), v) * dx
        F += mu * inner(grad(u_mid), grad(v)) * dx
        F -= inner(p*Identity(2), grad(v)) * dx
        F -= inner(q, div(u)) * dx
        # If you use CN then you need to define the exact stress at tmid -> sigma(u_mid, p_mid)
        F -= inner(sigma(u_exact,p_exact)*normal, v) * ds(2)
        F -= inner(f_mid, v) * dx
        
        a0, L0 = lhs(F), rhs(F)
        
        a1 = inner(grad(w), grad(z)) * dx
        L1 = dot(Constant((0.0,0.0)),z)*dx  
        
        
        t_ = dt
        while t_ < (T - 1E-9):
            
            t.assign(t_)    # in this way the constant t should be updated with the value t_
            # If t is a constant, everything that depends from t is automatically updated
            # u_exact_e.t = t_
            # w_exact_e.t = t_
            # p_exact_e.t = t_
            # a0, L0 = lhs(F), rhs(F)
            # 
            # a1 = inner(grad(w), grad(z)) * dx
            # L1 = dot(Constant((0.0,0.0)),z)*dx  
    
            # plot(u_exact_e, key="uex", title="uex", mesh=mesh)
            # interactive()
            
            A = assemble(a0)
            b = assemble(L0)
            
            for bc in bcu:
                bc.apply(A,b)
        
            solve(A, VP_.vector(), b)
            
            # Solving the Poisson problem
            A1 = assemble(a1)
            b1 = assemble(L1)
            
            # Do I need to update the boundary conditions? Updating the time should be enough, no?
            
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
            
            # plot(mesh)
            
            assign(up0, VP_)
            # u0, p0 = VP_.split()
            #plot(u0, title = str(t))
            # plot(u0, key = "u0", title = "u0", mesh=mesh)
            interactive()
            
            t_ += dt
            
    
        print "||u - uh||_H1 = {0:1.4e}".format(errornorm(u_exact_e, VP_.sub(0), "H1"))

        #Error in L2        
        # ||u - uh|| = 2.2131e-03
        # ||u - uh|| = 4.0381e-03
        # ||u - uh|| = 4.0333e-03
        # ||u - uh|| = 4.0331e-03
        # ||u - uh|| = 2.3938e-03
        # ||u - uh|| = 2.3744e-03
        # ||u - uh|| = 2.3695e-03
        # ||u - uh|| = 2.3693e-03
        # ||u - uh|| = 1.7012e-03
        # ||u - uh|| = 1.6177e-03
        # ||u - uh|| = 1.6128e-03
        # ||u - uh|| = 1.6125e-03
        # ||u - uh|| = 1.3738e-03
        # ||u - uh|| = 1.2606e-03
        # ||u - uh|| = 1.2555e-03
        # ||u - uh|| = 1.2552e-03
