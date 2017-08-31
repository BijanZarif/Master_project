from dolfin import *
from tabulate import tabulate
import math

set_log_level(60)

N = [2**3, 2**4, 2**5]
T = 1.0
DT = [1./(float(N[i])) for i in range(len(N))]
rho = 1.0
mu = 1.0/8.0
theta = 0.5
C = 0.1

u_errors = [[0 for j in range(len(N))] for i in range(len(DT))]
p_errors = [[0 for j in range(len(N))] for i in range(len(DT))]
w_errors = [[0 for j in range(len(N))] for i in range(len(DT))]

def sigma(mu, u, p, Ve, Pe):
    #project(u, Ve)
    #project(p, Pe)
    return mu*grad(u) - p*Identity(2)

def exact_solutions(mesh, C, t):
    x = SpatialCoordinate(mesh)
    u_e = as_vector(( sin(2*pi*x[1])*cos(2*pi*x[0])*cos(t), -sin(2*pi*x[0])*cos(2*pi*x[1])*cos(t)))
    p_e = cos(x[0])*cos(t)   # the p_e is not the same as in the test case that I had written, there's a *cos(x[1]) missing
    w_e = as_vector(( C*sin(2*pi*x[1])*cos(t) , 0.0))
    #w_e = as_vector((x[0]-x[0], x[0]-x[0]))
    return u_e, p_e, w_e
    
def f(rho, mu, dudt, u_e, p_e, w_e):
    ff = rho * dudt + rho * grad(u_e)*(u_e - w_e) - mu*div(grad(u_e)) + grad(p_e)
    #ff = rho * dudt - div(mu*grad(u_e)) + grad(p_e)
    return ff


ii = 0 
for dt in DT:
    print "="*20
    print "dt = {}".format(dt)
    jj = 0
    for n in N:

        t_ = 0.0
        t0 = Constant(0.0)
        t1 = Constant(dt)
        
        print "n = {}".format(n)

        mesh = UnitSquareMesh(n, n)
        x = SpatialCoordinate(mesh)
        normal = FacetNormal(mesh)      
        fd = FacetFunction("size_t", mesh)
        CompiledSubDomain("on_boundary").mark(fd, 1) 
        CompiledSubDomain("near(x[0], 1.0) && on_boundary").mark(fd, 2) 
        CompiledSubDomain("near(x[1], 0.0) && on_boundary").mark(fd, 2) 
        CompiledSubDomain("near(x[1], 1.0) && on_boundary").mark(fd, 2) 

        ds = Measure("ds", domain = mesh, subdomain_data = fd)

        (u_exact0, p_exact0, w_exact0) = exact_solutions(mesh, C, t0)
        (u_exact1, p_exact1, w_exact1) = exact_solutions(mesh, C, t1)

        
        #It is better to use diff(u_exact, t)
        dudt0 = diff(u_exact0, t0)
        dudt1 = diff(u_exact1, t1)
        
        #dudt =  as_vector(( -sin(2*pi*x[1])*cos(2*pi*x[0])*sin(t) , sin(2*pi*x[0])*cos(2*pi*x[1])*sin(t) ))
        f0 = f(rho, mu, dudt0, u_exact0, p_exact0, w_exact0)
        f1 = f(rho, mu, dudt1, u_exact1, p_exact1, w_exact1)
        #print "dudt = {}".format(dudt(1))
        
        
        #exit()
        
        # Taylor-Hood elements
        V = VectorFunctionSpace(mesh, "Lagrange", 2)
        P = FunctionSpace(mesh, "Lagrange", 1)
        W = VectorFunctionSpace(mesh, "CG", 2)  # Here I put a 2 otherwise I couldn't do the project on line 93
        Ve = VectorFunctionSpace(mesh, "CG", 4) # Spaces for the exact solution
        Pe = FunctionSpace(mesh, "CG", 4)       # Spaces for the exact solution

        V0 = VectorFunctionSpace(mesh, "CG", 2) # Used to project the exact solution
        VPW = MixedFunctionSpace([V,P,W])
        #u, p, w = TrialFunctions(VPW)
        v, q, z = TestFunctions(VPW)
        
        upw0 = Function(VPW)
        u0, p0, w0 = split(upw0)
        U = Function(W)     # total displacement
        
        # I am assigning to u0 and w0 as initial conditions the u_exact and w_exact at time t0
        assign(upw0.sub(0), project(u_exact0, V0))
        assign(upw0.sub(2), project(w_exact0, V0))
        
        # I'll store my solution here, and I am initializing them to zero
        VPW_ = Function(VPW)
        u, p, w = split(VPW_)
        U = Function(W)    # total displacement
        
        u_mid = (1.0-theta)*u0 + theta*u
        f_mid = (1.0-theta)*f0 + theta*f1
        sigma_mid = (1.0-theta)*sigma(mu, u_exact0, p_exact0, Ve, Pe) + theta*sigma(mu, u_exact1, p_exact1, Ve, Pe)
        
        # Variational form
        F = Constant(1./dt) * rho * inner(u - u0, v) * dx
        F += rho * inner(grad(u_mid)*(u0 - w0), v) * dx
        F += mu * inner(grad(u_mid), grad(v)) * dx
        F -= p * div(v) * dx
        F -= q * div(u) * dx
        F -= inner(dot(sigma_mid,normal), v) * ds(2)
        F -= inner(f_mid, v) * dx
        a0, L0 = lhs(F), rhs(F)

        # Variational form for the Poisson problem for the w
        a1 = inner(grad(w), grad(z))*dx
        L1 = - inner(div(grad(w_exact1)), z)*dx

        # I added the variational form for the w because I am solving in a coupled way
        F -= a1 - L1
        
        while t_ < (T - 1E-9):
            
            t0.assign(t_)    # in this way the constant t should be updated with the value t_
            t1.assign(t_ + dt)
            
            # Here I re-set the BC at every cycle, updating the time for the u_exact1 and w_exact1
            bcu = [DirichletBC(VPW.sub(0), project(u_exact1, Ve), fd, 1)]
            bcw = [DirichletBC(VPW.sub(2), project(w_exact1, Ve), "on_boundary")]

            solve(F == 0, VPW_, bcu + bcw, solver_parameters={"newton_solver": {"relative_tolerance": 1e-20}})
            
            plot(VPW_.sub(0), key="u", mesh=mesh)
            plot(VPW_.sub(1), key="p", mesh=mesh)
            
            upw0.assign(VPW_)
            v_, p_, w_ = VPW_.split(True)
            
            w_.vector()[:] *= float(Constant(dt))
            U.vector()[:] += w_.vector()[:]
            ALE.move(mesh, w_)
            mesh.bounding_box_tree().build(mesh)
            
            t_ += dt
            
            
            
        u_errors[ii][jj] = "{0:1.4e}".format(errornorm(VPW_.sub(0), project(u_exact1, Ve), norm_type="H1", degree_rise=3))
        w_errors[ii][jj] = "{0:1.4e}".format(errornorm(VPW_.sub(2), project(w_exact1, Ve), norm_type="H1", degree_rise=3))
        print "||u - uh||_H1 = {0:1.4e}".format(errornorm(VPW_.sub(0), project(u_exact1, Ve), norm_type="H1", degree_rise=3))  
        print "||p - ph||_L2 = {0:1.4e}".format(errornorm(VPW_.sub(1), project(p_exact1, Pe), norm_type="L2", degree_rise=3))
        print "||w - wh||_H1 = {0:1.4e}".format(errornorm(VPW_.sub(2), project(w_exact1, Ve), norm_type="H1", degree_rise=3))

        t1.assign(t_ - dt*(1-theta))
        p_errors[ii][jj] = "{0:1.4e}".format(errornorm(VPW_.sub(1), project(p_exact1, Pe), norm_type="L2", degree_rise=3))
        jj +=1

    
    ii +=1

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


