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
    
    plot(u_exact_e, mesh = mesh, title = "exact velocity")
    plot(p_exact_e, mesh = mesh, title = "exact pressure")