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
    #w_e = as_vector(( C*sin(2*pi*x[1])*cos(t) , 0.0))
    w_e = as_vector((x[0]-x[0], x[0]-x[0]))
    return u_e, p_e, w_e
    
def f(rho, mu, dudt, u_e, p_e, w_e):
    ff = rho * dudt + rho * grad(u_e)*(u_e - w_e) - mu*div(grad(u_e)) + grad(p_e)
    #ff = rho * dudt - div(mu*grad(u_e)) + grad(p_e)
    return ff

# fileu = File("u.pvd")
# fileu0 = File("u0_ex.pvd")
# fileu1 = File("u1_ex.pvd")
# 
# filep = File("p.pvd")
# filep0 = File("p0_ex.pvd")
# filep1 = File("p1_ex.pvd")

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
        gamma = 1e2
        h = CellSize(mesh)
        
        fd = FacetFunction("size_t", mesh)
        CompiledSubDomain("on_boundary").mark(fd, 1) 
        CompiledSubDomain("near(x[0], 1.0) && on_boundary").mark(fd, 2)  # right boundary
        CompiledSubDomain("near(x[1], 0.0) && on_boundary").mark(fd, 2)  # btm boundary
        CompiledSubDomain("near(x[1], 1.0) && on_boundary").mark(fd, 2)  # top boundary
        # at the end the boundary "2" includes every boundary except the left boundary
        # while the boundary "1" is just the left one
        

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
        V0 = VectorFunctionSpace(mesh, "CG", 2)
        P0 = FunctionSpace(mesh, "CG", 1)
        
        V1 = VectorFunctionSpace(mesh, "CG", 1)
        Ve = VectorFunctionSpace(mesh, "CG", 4)
        Pe = FunctionSpace(mesh, "CG", 4)

        # Make a mixed space
        VP = V * P
                     
        #u, p = TrialFunctions(VP)   # u is a trial function of V, while p a trial function of P
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
        VP_ = Function(VP); (u,p) = split(VP_) 
        W_ = Function(V0)
        
        u_mid = (1.0-theta)*u0 + theta*u
        f_mid = (1.0-theta)*f0 + theta*f1   # at every time step I should have f0 which is the f_exact calculated at the time t=i
                                            # while f is the f_exact at the time t=i+1
        sigma_mid = (1.0-theta)*sigma(mu, u_exact0, p_exact0, Ve, Pe) + theta*sigma(mu, u_exact1, p_exact1, Ve, Pe)
    
        F = Constant(1./dt) * rho * inner(u - u0, v) * dx
        F += rho * inner(grad(u_mid)*(u_mid - w0), v) * dx  
        F += mu * inner(grad(u_mid), grad(v)) * dx
        F -= p * div(v) * dx
        F -= q * div(u) * dx
        F -= inner(dot(sigma_mid,normal), v) * ds(2)
        
        #F -= inner(dot(sigma, normal),v) * ds(1)   # I added this part because now I don't have Dirichlet condition anymore on the left boundary,
                                                    # so this term is not zero because the test function v is not zero on the boundary
        #Nitsche term
        F += ( - mu * inner(grad(u_mid)*normal,v) - mu * inner(grad(u_mid)*normal,v) - gamma * h**-1 * inner(u,v) +
              mu * inner(grad(v) * normal, u_exact1) + gamma * h**-1 * inner(u_exact1, v) ) * ds(1)
        F += inner(p*normal,v) * ds(1)
        
        F -= inner(f_mid, v) * dx
        
        a0, L0 = lhs(F), rhs(F)
        
        a1 = inner(grad(w), grad(z)) * dx

        L1 = inner(-div(grad(w_exact1)),z) * dx  
        
        
        while t_ < (T - 1E-9):
            

            t0.assign(t_)    # in this way the constant t should be updated with the value t_
            t1.assign(t_ + dt)
        
            #bcu = [DirichletBC(VP.sub(0), project(u_exact1, Ve), fd, 1)]
            # applying this boundary condition only on the left boundary
            
            #bcu = [DirichletBC(VP.sub(0), project(u_exact1, Ve), "on_boundary")]
            bcw = [DirichletBC(V0, project(w_exact0, Ve), "on_boundary")]

            # fileu << project(u0, V0)
            # fileu0 << project(u_exact0, V0)
            # fileu1 << project(u_exact1, V0)
            # 
            # filep << project(p0, P0)
            # filep0 << project(p_exact0, P0)
            # filep1 << project(p_exact1, P0)

            #A = assemble(a0)
            #b = assemble(L0)
            
            #for bc in bcu:
            #    bc.apply(A,b)
        
            solve(F == 0, VP_, solver_parameters={"newton_solver":
                                            {"relative_tolerance": 1e-20}})
            #solve(A, VP_.vector(), b)
            plot(VP_.sub(0), key="u", mesh=mesh)
            plot(VP_.sub(1), key="p", mesh=mesh)
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


        u_errors[ii][jj] = "{0:1.4e}".format(errornorm(VP_.sub(0), project(u_exact1, Ve), norm_type="H1", degree_rise=3))
        w_errors[ii][jj] = "{0:1.4e}".format(errornorm(W_, project(w_exact1, Ve), norm_type="H1", degree_rise=3))
        print "||u - uh||_H1 = {0:1.4e}".format(errornorm(VP_.sub(0), project(u_exact1, Ve), norm_type="H1", degree_rise=3))  
        print "||w - wh||_H1 = {0:1.4e}".format(errornorm(W_, project(w_exact1, Ve), norm_type="H1", degree_rise=3))

        t1.assign(t_ - dt*(1-theta))
        p_errors[ii][jj] = "{0:1.4e}".format(errornorm(VP_.sub(1), project(p_exact1, Pe), norm_type="L2", degree_rise=3))
        jj +=1


        print "||p - ph||_L2 = {0:1.4e}".format(errornorm(VP_.sub(1), project(p_exact1, Pe), norm_type="L2", degree_rise=3))
    
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



