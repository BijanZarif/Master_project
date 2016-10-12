## STOKES PROBLEM ##

# - div( nu * grad(u) - pI ) = f
# div( u ) = 0

from numpy import *
from math import *
from matplotlib import pyplot as plt
from fenics import *
from mshr import *   # I need this if I want to use the functions below to create a mesh


NN = [2**2, 2**3, 2**4, 2**5, 2**6]
#NN = [2**2]

mu = Constant(1.0)
Re = Constant(300)
beta = Constant(10.0)

h = [1./i for i in NN]
h2 = [1./(i**2) for i in NN]
errsL2 = []
errsH1 = []
errsL2pressure = []
errsH1pressure = []
rates1 = []
rates2 = []
rates3 = []

for N in NN:
    
    mesh = RectangleMesh(Point(0.0, 0.0), Point(20.0, 4.0), 4*N, N)
    #plot(mesh)
    
    normal = FacetNormal(mesh)
    x = SpatialCoordinate(mesh)
    h1 = CellSize(mesh)
    
    V = VectorFunctionSpace(mesh, "Lagrange", 2)  # space for velocity
    Q = FunctionSpace(mesh, "Lagrange", 1)        # space for pressure
    W = V * Q
    
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    
    u_exact = as_vector((x[1] + sin(pi*x[1]), x[0] + sin(pi*x[0]) ))
    p_exact = x[0] + cos(pi*x[0])
    f_exact = as_vector(( -pi*sin(pi*x[0]) + Re**-1 * pi**2 * sin(pi*x[1]) + 1, Re**-1 * pi**2 * sin(pi*x[0]) ))
    
    u_exact_e = Expression(("x[1] + sin(pi*x[1])","x[0] + sin(pi*x[0])"), domain = mesh, degree =2)
    p_exact_e = Expression("x[0] + cos(pi*x[0])", domain=mesh, degree = 1)
    f_exact_e = Expression(("-pi*sin(pi*x[0]) + 1.0/Re * pi*pi * sin(pi*x[1]) + 1.0","1.0/Re * pi*pi * sin(pi*x[0])"), Re=Re, domain=mesh, degree=2)
    
    #plot(u_exact_e, mesh = mesh, title = "exact velocity")
    #plot(p_exact_e, mesh = mesh, title = "exact pressure")
    
    
    F0 = Re**-1 * inner(grad(u),grad(v))*dx - inner(p, div(v))*dx 
    F0 += (- Re**-1 * inner(grad(u)*normal, v) + inner(p*normal,v) + inner(q*normal,u) - Re**-1 * inner(grad(v)*normal,u) + beta * h1**-1 * Re**-1 * inner(u,v))*ds # Nitsche method
    F0 += - inner(f_exact_e,v)*dx + (Re**-1 * inner(grad(v)*normal,u_exact_e) - inner(q*normal,u_exact_e) -  beta * h1**-1 * Re**-1 * inner(u_exact_e,v)) * ds # RHS
    F0 += - inner(q,div(v))*dx # continuity equation 
    
    a = lhs(F0)
    L = rhs(F0)
    
    A = assemble(a, PETScMatrix())
    b = assemble(L)
    
        # ----------------------- #
    # IN THIS WAY I AM SETTING THE NULL SPACE FOR THE PRESSURE
    # since p + C for some constant C is still a solution, I take the pressure with mean value 0
    
    constant_pressure = Function(W).vector()
    constant_pressure[W.sub(1).dofmap().dofs()] = 1
    null_space = VectorSpaceBasis([constant_pressure])
    A.set_nullspace(null_space)
    
    # ----------------------- #
    
    U = Function(W)
    solve(A, U.vector(), b)
    uh, ph = U.split()
    
    