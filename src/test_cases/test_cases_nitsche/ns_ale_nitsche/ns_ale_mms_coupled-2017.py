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


def solve_system(n=4, dt=0.1, k=1):
    
    # Define space and time. Make sure to initialize t with the
    # initial time.
    mesh = UnitSquareMesh(n, n)
    t = Constant(0.0)
    T = 1.0

    # Define model parameters:
    rho = Constant(1.0)
    nu = Constant(1.0)

    # Define analytical solution
    f = Expression(("2.0 - t", "0.0"), degree=1, t=t)
    w = Expression(("0.1", "0.1"), degree=0, t=t)
    
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

    # Define boundary pressure p
    n = FacetNormal(mesh) 
    p0 = as_vector((t, 0.0))
    p1 = as_vector((0.0, 0.0))
    
    # Define some boundaries
    boundary = FacetFunction("size_t", mesh, 0)
    CompiledSubDomain("near(x[1], 0.0) && on_boundary").mark(boundary, 1) 
    CompiledSubDomain("near(x[1], 1.0) && on_boundary").mark(boundary, 1) 
    CompiledSubDomain("near(x[0], 0.0) && on_boundary").mark(boundary, 2) 
    CompiledSubDomain("near(x[0], 1.0) && on_boundary").mark(boundary, 3) 

    ds = Measure("ds", domain=mesh, subdomain_data=boundary)
    
    # Define variational formulation for Navier-Stokes
    F = inner(rho*((u - u_)/dt + grad(u)*(u - w)), v)*dx() \
        + inner(sigma(u), grad(v))*dx() \
        + div(u)*q*dx() \
        - inner(p0, v)*ds(2) \
        - inner(p1, v)*ds(3) \
        - inner(f, v)*dx() 
    
    bc = DirichletBC(M.sub(0), (0.0, 0.0), boundary, 1)
    
    # Initialize u_ with interpolated exact initial condition
    u_e = Expression(("x[1]*(1.0 - x[1])", "0.0"), degree=2, t=t)
    u0 = interpolate(u_e, M.sub(0).collapse())
    assign(up_.sub(0), u0)
    p_e = Expression("t*(1.0 - x[0])", degree=1, t=T)
    pT = interpolate(p_e, M.sub(1).collapse())
    
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
        plot(p, key="p")

        # Move mesh
        ALE.move(mesh, w)
        mesh.bounding_box_tree().build(mesh)

        u_vec = up.split(deepcopy=True)[0].vector().array()
        p_vec = up.split(deepcopy=True)[1].vector().array()
        print max(p_vec)
        print min(p_vec)
        print max(u_vec)
        print min(u_vec)

    # Compute error at t
    print "\| u - u_e \| = ", math.sqrt(assemble(inner(u - u0, u - u0)*dx()))
    print "\| p - p_e \| = ", math.sqrt(assemble(inner(p - pT, p - pT)*dx()))
    
    interactive()

if __name__ == "__main__":

    solve_system(n=8, dt=0.25)
