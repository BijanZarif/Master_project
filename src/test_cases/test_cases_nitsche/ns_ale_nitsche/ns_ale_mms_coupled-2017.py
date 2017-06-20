from dolfin import *
import math

import sympy

#set_log_level(60)

# N = [2**2, 2**3]#, 2**5]
# T = 1.0
# DT = [1./float(n) for n in N]
# rho = 1.0
# mu = 1.0/8.0
# theta = 1.0
# C = Constant(0.0)

def analytical_expression(nu=None, rho=None):
    x, y, t = sympy.symbols('x[0], x[1], t')
    u = (y*(1.0 - y)*t, 0.0)
    p = (1.0 - x)*t
    w = (0.0, 0.0)
    
    # Compute the gradient of u
    grad_u = ((sympy.diff(u[0], x), sympy.diff(u[0], y)),
              (sympy.diff(u[1], x), sympy.diff(u[1], y)))

    # Compute grad(u) * u
    grad_u_x_u = (grad_u[0][0]*(u[0] - w[0]) + grad_u[0][1]*(u[1] - w[1]),
                  grad_u[1][0]*(u[0] - w[0]) + grad_u[1][1]*(u[1] - w[1]))

    # Compute sigma
    sigma = [[nu*grad_u[i][j] for j in range(2)]
             for i in range(2)]
    sigma[0][0] = sigma[0][0] - p
    sigma[1][1] = sigma[1][1] - p

    for i in range(2):
        for j in range(2):
            sigma[i][j] = sympy.simplify(sigma[i][j])
    
    # Compute div(sigma)
    div_sigma = (sympy.diff(sigma[0][0], x) + sympy.diff(sigma[0][1], y),
                 sympy.diff(sigma[1][0], x) + sympy.diff(sigma[1][1], y))

    f = [sympy.simplify(rho*sympy.diff(u[i], t) + rho*grad_u_x_u[i] - div_sigma[i]) for i in range(2)]

    print "Analytical solutions:"

    u = tuple(sympy.printing.ccode(u[i]) for i in range(2))
    p = sympy.printing.ccode(p)
    f = tuple(sympy.printing.ccode(f[i]) for i in range(2))
    sigma = [[sympy.printing.ccode(sigma[i][j]) for j in range(2)]
             for i in range(2)]

    return (u, p, w, sigma, f)

def solve_system(n=4, dt=0.1, k=1):
    
    # Define space and time. Make sure to initialize t with the
    # initial time.
    mesh = UnitSquareMesh(n, n)
    t = Constant(0.0)
    T = 1.0

    # Define model parameters:
    rho = Constant(1.0)
    nu = Constant(1.0)

    # Define analytical solutions
    (u_e, p_e, w_e, sigma_e, f_e) = analytical_expression(nu=nu, rho=rho)
    print "u_e = ", u_e
    print "p_e = ", p_e
    print "f_e = ", f_e
    print "sigma_e = ", sigma_e
    
    # Convert analytical representations to FEniCS Expressions
    u_e = Expression(u_e, degree=k+2, t=t)
    p_e = Expression(p_e, degree=k+2, t=t)
    f = Expression(f_e, degree=k+2, t=t)
    # Sidenote: there is something not implemented with tensor valued
    # Expressions, using this work around instead by splitting sigma
    # into its rows sigma0 and sigma1
    sigma0 = Expression(sigma_e[0], degree=k+2, t=t)
    sigma1 = Expression(sigma_e[1], degree=k+2, t=t)

    # Define Taylor-Hood elements of order k
    V = VectorElement("CG", mesh.ufl_cell(), k+1)
    Q = FiniteElement("CG", mesh.ufl_cell(), k)
    M = FunctionSpace(mesh, MixedElement(V, Q))
    
    # Define functions and test functions
    up = Function(M)     # Current solution
    up_ = Function(M)    # Previous solution

    (u, p) = split(up)
    (u_, p_) = split(up_)
    (v, q) = TestFunctions(M)

    # Define solver parameters
    dt = Constant(dt)

    # Define short hand for sigma
    sigma = lambda u: nu*grad(u) - p*Identity(2)

    # Define boundary traction s
    n = FacetNormal(mesh) 
    s = as_vector((dot(sigma0, n), dot(sigma1, n)))

    
    # Define variational formulation for Navier-Stokes
    F = inner(rho*((u - u_)/dt + grad(u)*u), v)*dx() \
        + inner(sigma(u), grad(v))*dx() \
        + div(u)*q*dx() \
        - inner(s, v)*ds() \
        - inner(f, v)*dx() 
    
    bc = DirichletBC(M.sub(0), u_e, "!near(x[0], 0.0) && on_boundary")
    
    # Initialize u_ with interpolated exact initial condition
    assign(up_.sub(0), interpolate(u_e, M.sub(0).collapse()))
    
    # Step in time:
    while (float(t) < T):

        # Update time
        t.assign(float(t) + float(dt))
        print "Solving at t = ", float(t)

        # Solve system
        solve(F == 0, up, bc)

        # Update previous solution
        up_.assign(up)

        plot(u, key="u")
        plot(u_e, mesh=mesh, key="u_e")

    # Compute error at t
    print "\| u - u_e \| = ", math.sqrt(assemble(inner(u - u_e, u - u_e)*dx()))
    print "\| u - u_e \| = ", math.sqrt(assemble(inner(p - p_e, p - p_e)*dx()))
    
    interactive()

if __name__ == "__main__":

    solve_system(n=8, dt=0.25)
