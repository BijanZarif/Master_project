from dolfin import *
from tabulate import tabulate
import math

set_log_level(50)
#N = [(2**n, 0.5**(2*n)) for n in range(1, 5)]

N = [2**3, 2**4, 2**5]
#N = [2**4]
T = 1.0
DT = [1./(float(N[i])) for i in range(len(N))]
#DT = [1./10000]
rho = 1.0
mu = 1.0/8.0
theta = 1.0
# theta = 0.5

#theta = 0.5
theta = 1.0
C = 0.1

u_errors = [[j for j in range(len(N))] for i in range(len(DT))]
p_errors = [[j for j in range(len(N))] for i in range(len(DT))]
w_errors = [[j for j in range(len(N))] for i in range(len(DT))]

def sigma(u,p):
    return 2.0*mu*sym(grad(u)) - p*Identity(2)

def exact_solutions(mesh, C, t):

    x = SpatialCoordinate(mesh)
    u_e = as_vector(( sin(2*pi*x[1])*cos(2*pi*x[0])*cos(t) , -sin(2*pi*x[0])*cos(2*pi*x[1])*cos(t) ))
    p_e = cos(x[0])*cos(x[1])*cos(t)
    #w_e = as_vector(( C*sin(2*pi*x[1])*cos(t) , 0.0))
    w_e = as_vector((x[0]-x[0], x[0]-x[0]))
    return u_e, p_e, w_e
    
def f(rho, mu, dudt, u_e1, u_e0, p_e, w_e):
    ff = rho * dudt + rho * grad(u_e0)*(u_e0 - w_e) - div(2.0*mu*sym(grad(u_e1)) - p_e*Identity(2))
    return ff

for dt in DT:
    print "="*20
    print "dt = {}".format(dt)
    i = 0 
    for n in N:
        
        j = 0 
        t_ = 0.0
        t0 = Constant(0.0)
        t1 = Constant(dt)
        
        print "n = {}".format(n)

        mesh = UnitSquareMesh(n, n)
        x = SpatialCoordinate(mesh)
        normal = FacetNormal(mesh)      
        fd = FacetFunction("size_t", mesh)
        CompiledSubDomain("near(x[0], 0.0) && on_boundary").mark(fd, 1) # left wall (cord)    
        CompiledSubDomain("near(x[0], 1.0) && on_boundary").mark(fd, 2) # right wall (tissue)  
        # CompiledSubDomain("near(x[1], 1.0) ||( near(x[0], 1.0) && near(x[1], 1.0) ) && on_boundary").mark(fd, 3) # top wall (inlet)
        # CompiledSubDomain("near(x[1], 0.0) ||( near(x[0], 1.0) && near(x[1], 0.0) ) && on_boundary").mark(fd, 4) # bottom wall (outlet)
        CompiledSubDomain("near(x[1], 1.0)").mark(fd, 3) # top wall (inlet)
        CompiledSubDomain("near(x[1], 0.0)").mark(fd, 4) # bottom wall (outlet)

        ds = Measure("ds", domain = mesh, subdomain_data = fd)

        (u_exact0, p_exact0, w_exact0) = exact_solutions(mesh, C, t0)
        (u_exact1, p_exact1, w_exact1) = exact_solutions(mesh, C, t1)

        #Write exact solution using UFL
        # w_exact0 = Constant((0.0,0.0))
        # w_exact1 = Constant((0.0,0.0))
        
        #It is better to use diff(u_exact, t)
        dudt0 = diff(u_exact0, t0)
        dudt1 = diff(u_exact1, t1)
        
        #dudt =  as_vector(( -sin(2*pi*x[1])*cos(2*pi*x[0])*sin(t) , sin(2*pi*x[0])*cos(2*pi*x[1])*sin(t) ))
        f0 = f(rho, mu, dudt0, u_exact0, u_exact1, p_exact0, w_exact0)
        f1 = f(rho, mu, dudt1, u_exact0, u_exact1, p_exact1, w_exact1)
        #print "dudt = {}".format(dudt(1))
        
        
        #exit()
        
        # Taylor-Hood elements
        V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        V0 = VectorFunctionSpace(mesh, "CG", 2)
        P0 = FunctionSpace(mesh, "CG", 1)
        V1 = VectorFunctionSpace(mesh, "CG", 1)
        Ve = VectorFunctionSpace(mesh, "CG", 4)
        Pe = FunctionSpace(mesh, "CG", 4)

        # Make a mixed space
        TH = V * P
        VP = FunctionSpace(mesh, TH)
                
        u, p = TrialFunctions(VP)   # u is a trial function of V, while p a trial function of P
        w = TrialFunction(V0)
        
        v, q = TestFunctions(VP)
        z = TestFunction(V0)
        
        up0 = Function(VP)
        u0, p0 = split(up0)
        w0 = Function(V0)
        w1 = Function(V0)
        
        assign(up0.sub(0), project(u_exact0, V0))
        assign(w0, project(w_exact0, V0))
        
        X = Function(V0)  # in here I will put the displacement X^(n+1) = X^n + dt*(w^n)
        #Xn_1 = Function(W)  # in here I will put the displacement X^(n+1) = X^n + dt*(w^n)
        Y = Function(V0)
        
        # I want to store my solutions here
        VP_ = Function(VP)   
        W_ = Function(V0)
        
        u_mid = (1.0-theta)*u0 + theta*u
        f_mid = (1.0-theta)*f0 + theta*f1   # at every time step I should have f0 which is the f_exact calculated at the time t=i
                                            # while f is the f_exact at the time t=i+1
        sigma_mid = (1.0-theta)*sigma(u_exact0, p_exact0) + theta*sigma(u_exact1, p_exact1)
        

        # Define boundary conditions
        
        bcu = [DirichletBC(VP.sub(0), project(u_exact1, Ve), fd, 1),
               # DirichletBC(VP.sub(0), u_exact_e, fd, 2),
               DirichletBC(VP.sub(0), project(u_exact1, Ve), fd, 3),
               DirichletBC(VP.sub(0), project(u_exact1, Ve), fd, 4)]
        
        bcw = [DirichletBC(V0, project(w_exact1, Ve), "on_boundary")]
    
        F = Constant(1./dt) * rho * inner(u - u0, v) * dx
        #F += rho * inner(grad(u_mid)*(u0 - w0), v) * dx
        F += rho * inner(grad(u0)*(u0), v) * dx        
        F += 2.0 * mu * inner(sym(grad(u_mid)), sym(grad(v))) * dx
        F -= p * div(v) * dx
        F -= q * div(u) * dx
        F -= inner(sigma_mid*normal, v) * ds(2)
        F -= inner(f_mid, v) * dx
        
        a0, L0 = lhs(F), rhs(F)
        
        a1 = inner(grad(w), grad(z)) * dx

        L1 = inner(-div(grad(w_exact1)),z) * dx  
        
        
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
            Y.vector()[:] = 0.5*dt*(w0.vector()[:] + w1.vector()[:])
            X.vector()[:] += Y.vector()[:]
            w1.assign(w0)

            ALE.move(mesh, project(Y, V1))
            mesh.bounding_box_tree().build(mesh)            
            
            t_ += dt
        
        plot(u0, mesh=mesh)
        u_errors[i][j] = "{0:1.4e}".format(errornorm(project(u_exact1, Ve), VP_.sub(0), "H1"))
        w_errors[i][j] = "{0:1.4e}".format(errornorm(project(w_exact1, Ve), W_, "H1"))
        print "||u - ph||_H1 = {0:1.4e}".format(errornorm(project(u_exact1, Ve) , VP_.sub(0), "H1"))  

        t1.assign(t_ - dt*(1-theta))
        p_errors[i][j] = "{0:1.4e}".format(errornorm(project(p_exact1, Pe), VP_.sub(1), "L2"))
        j +=1

        print "||p - ph||_L2 = {0:1.4e}".format(errornorm(project(p_exact1, Pe), VP_.sub(1), "L2"))
        print "||w - wh||_H1 = {0:1.4e}".format(errornorm(project(w_exact1, Ve), W_, "H1"))
    
    i +=1

def convergence_rates(errors, hs):
    rates = [(math.log(errors[i+1]/errors[i]))/(math.log(hs[i+1]/hs[i])) for i in range(len(hs)-1)]

    return rates

print "u_errors = ", u_errors
# print rate(u_errors)
# 
print "w_errors = ", w_errors
# print rate(w_errors)
# 
print "p_errors = ", p_errors
# print rate(p_errors)


