from dolfin import *
from tabulate import tabulate
set_log_level(50)
#N = [(2**n, 0.5**(2*n)) for n in range(1, 5)]

N = [2**4, 2**5]
#N = [2**7]
T = 1.0
DT = [1./N[i] for i in range(len(N))]

rho = 1.0
mu = 1.0/8.0
theta = 0.5
C = 0.1

u_errors = [[j for j in range(len(N))] for i in range(len(DT))]
p_errors = [[j for j in range(len(N))] for i in range(len(DT))]
def sigma(u,p):
    return 2.0*mu*sym(grad(u)) - p*Identity(2)

i = 0

for dt in DT:
    print "dt = {}".format(dt)

    j=0
    for n in N:
        
        t_ = 0.0
        t0 = Constant(0.0)
        t1 = Constant(dt)
        
        print "n = {}".format(n)

        mesh = UnitSquareMesh(n, n)
        x = SpatialCoordinate(mesh)
        normal = FacetNormal(mesh)
        
        ##EP: Why do you use Expression? It is better to use directly UFL
        
        u_exact_e = Expression(("sin(2*pi*x[1])*cos(2*pi*x[0])*cos(t)", "-sin(2*pi*x[0])*cos(2*pi*x[1])*cos(t)"), t = t1, degree=4)
        #w_exact_e = Expression(("C*sin(2*pi*x[1])*cos(t)", "0.0"), C=C, t = t1, degree=4)
        w_exact_e = Expression(("0.0", "0.0"))
        p_exact_e = Expression("cos(x[0])*cos(x[1])*cos(t)", t = t1, degree=4)
        
        #Write exact solution using UFL
        u_exact0 = as_vector(( sin(2*pi*x[1])*cos(2*pi*x[0])*cos(t0) , -sin(2*pi*x[0])*cos(2*pi*x[1])*cos(t0) ))
        u_exact1 = as_vector(( sin(2*pi*x[1])*cos(2*pi*x[0])*cos(t1) , -sin(2*pi*x[0])*cos(2*pi*x[1])*cos(t1) ))
        #w_exact0 = as_vector(( C*sin(2*pi*x[1])*cos(t0) , 0.0))
        #w_exact1 = as_vector(( C*sin(2*pi*x[1])*cos(t1) , 0.0))
        w_exact0 = Constant((0.0,0.0))
        w_exact1 = Constant((0.0,0.0))
        p_exact0 = cos(x[0])*cos(x[1])*cos(t0)
        p_exact1 = cos(x[0])*cos(x[1])*cos(t1)
        
        #It is better to use diff(u_exact, t)
        dudt0 = diff(u_exact0, t0)
        dudt1 = diff(u_exact1, t1)
        
        #dudt =  as_vector(( -sin(2*pi*x[1])*cos(2*pi*x[0])*sin(t) , sin(2*pi*x[0])*cos(2*pi*x[1])*sin(t) ))
        f0 = rho * dudt0 + rho * grad(u_exact0)*(u_exact0 - w_exact0) - div(2.0*mu*sym(grad(u_exact0)) - p_exact0*Identity(2))
        f1 = rho * dudt1 + rho * grad(u_exact1)*(u_exact1 - w_exact1) - div(2.0*mu*sym(grad(u_exact1)) - p_exact1*Identity(2))
        #print "dudt = {}".format(dudt(1))
        
        
        #exit()
        
        # Taylor-Hood elements
        V = VectorFunctionSpace(mesh, "CG", 2)  # space for u, v
        #Ve = VectorFunctionSpace(mesh, "CG", 4) # space to interpolate exact solution
        P = FunctionSpace(mesh, "CG", 1)        # space for p, q
        W = VectorFunctionSpace(mesh, "CG", 2)       # space for w
        VP = V * P
        
        assigner = FunctionAssigner(V, VP.sub(0))
        
        u, p = TrialFunctions(VP)   # u is a trial function of V, while p a trial function of P
        w = TrialFunction(W)
        
        v, q = TestFunctions(VP)
        z = TestFunction(W)
        
        up0 = Function(VP)
        u0, p0 = split(up0)
        w0 = Function(W)
        w_1 = Function(W)
        
        u_exact_int = project(u_exact0, V)
        w_exact_int = project(w_exact0, W)
        assign(up0.sub(0), u_exact_int) # I want to start with u0 = u_exact_e as initial condition
        assign(w0, w_exact_int)
        
        X = Function(W)  # in here I will put the displacement X^(n+1) = X^n + dt*(w^n)
        #Xn_1 = Function(W)  # in here I will put the displacement X^(n+1) = X^n + dt*(w^n)
        Y = Function(W)
        
        # I want to store my solutions here
        VP_ = Function(VP)   
        W_ = Function(W)
        
        u_mid = (1.0-theta)*u0 + theta*u
        f_mid = (1.0-theta)*f0 + theta*f1   # at every time step I should have f0 which is the f_exact calculated at the time t=i
                                            # while f is the f_exact at the time t=i+1
        sigma_mid = (1.0-theta)*sigma(u_exact0,p_exact0) + theta*sigma(u_exact1,p_exact1)
        

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
        F += 2.0 * mu * inner(sym(grad(u_mid)), sym(grad(v))) * dx
        F -= p * div(v) * dx
        F -= q * div(u) * dx
        F -= inner(sigma_mid*normal, v) * ds(2)
        F -= inner(f_mid, v) * dx
        
        a0, L0 = lhs(F), rhs(F)
        
        a1 = inner(grad(w), grad(z)) * dx
        L1 = dot(Constant((0.0,0.0)),z)*dx  
        
        
        while t_ < (T - 1E-9):
            

            t0.assign(t_)    # in this way the constant t should be updated with the value t_
            t1.assign(t_ + dt)
                        
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
    
            up0.assign(VP_) # the first part of VP_ is u0, and the second part is p0
            w0.assign(W_)
            
            
            # Compute the mesh displacement
            #Needs to be modified!! Make a second order scheme!            
            #Y.vector()[:] = w0.vector()[:]*dt
            #X.vector()[:] += Y.vector()[:]

            # With the trapezoidal rule: X1 = dt/2*(w1 - w0) + X0
            Y.vector()[:] = 0.5*dt*(w0.vector()[:] + w_1.vector()[:])
            X.vector()[:] += Y.vector()[:]
            w_1.assign(w0)

             # Move the mesh
            ALE.move(mesh, Y)
            mesh.bounding_box_tree().build(mesh)

            #plot(mesh)
            #plot(u0, key="u0", title = "u0", mesh=mesh)
            #plot(u_exact_e, key="uexact", title = "uexact", mesh=mesh)
            
            # foo = project(u0, V)
            # error_f = interpolate(u_exact_e, V)
            # error_f.vector().axpy(-1, foo.vector())
            # 
            # try:
            #     pp_plt.plot(error_f)
            # except NameError:
            #     pp_plt = plot(error_f)
    
            
            
            t_ += dt

        print "t_ = ", t_
        print "t1 = ",  float(t1)
        # print "||u - uh||_H1 = {0:1.4e}".format(errornorm(u_exact_e, VP_.sub(0), "H1"))
        # print "||p - ph||_L2 = {0:1.4e}".format(errornorm(p_exact_e, VP_.sub(1), "L2"))
        # print "||w - wh||_H1 = {0:1.4e}".format(errornorm(w_exact_e, W_, "H1"))

        u_errors[i][j] = "{0:1.4e}".format(errornorm(u_exact_e, VP_.sub(0), "H1"))
        j +=1
        # print "{0:1.4e}".format(errornorm(u_exact_e, VP_.sub(0), "H1"))
        # print "||p - ph||_L2 = {0:1.4e}".format(errornorm(p_exact_e, VP_.sub(1), "L2"))
        # print "||w - wh||_H1 = {0:1.4e}".format(errornorm(w_exact_e, W_, "H1"))
    i +=1
    



