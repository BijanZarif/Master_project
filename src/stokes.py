## STOKES PROBLEM ##

# - div( nu * grad(u) - pI ) = f
# div( u ) = 0

from numpy import *
from math import *
from matplotlib import pyplot as plt
from fenics import *
from mshr import *   # I need this if I want to use the functions below to create a mesh

# domain = Rectangle(Point(0., 0.), Point(1.0,1.0))
# mesh = generate_mesh(domain, 16)

N = [2**2, 2**3, 2**4, 2**5, 2**6]
h = [1./i for i in N]
h2 = [1./(i**2) for i in N]
errsL2 = []
errsH1 = []
errsL2pressure = []
rates1 = []
rates2 = []
rates3 = []

for n in N:

    mesh = UnitSquareMesh(n,n)
    
    # ANOTHER WAY TO DEFINE THE TAYLOR HOOD ON FEniCS 1.7
    #P1 = FiniteElement("Lagrange", triangle, 1)
    #P2 = FiniteElement("Lagrange", triangle, 2)
    #TH = (P2 * P2) * P1
    #W = FunctionSpace(mesh, TH)
    
    V = VectorFunctionSpace(mesh, "Lagrange", 2)  # space for velocity
    Q = FunctionSpace(mesh, "Lagrange", 1)        # space for pressure
    W = V * Q
    
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    
    x = SpatialCoordinate(mesh)
    nu = 1.0/8.0
    
    
    # I have to remember that the u_exact has to satisfy as well the boundary conditions (and not only the system of equations)
    # that's why there's the pi*x[0], so the sin is 0 on the right boundary (i.e. x[0] = 1))
    u_exact = as_vector((0, sin(pi*x[0]))) # to use as a solution to verify the convergence 
    # u_exact = as_vector((0, x[0]*(1-x[0])))   # as_vector() ???
    p_exact = 0.5 - x[1]
    
    f = - nu*div(grad(u_exact)) + grad(p_exact)   # I changed the sign in the gradient
    
    # Since the pressure is defined up to some constant, we compare the gradients
    g =  nu*div(grad(u_exact)) + f             # pressure gradient
    
    # u_exact_e = Expression((" 0 ", "x[0]*(1-x[0])" ), domain=mesh, degree=2)
    u_exact_e = Expression((" 0 ", "sin(pi*x[0])" ))
    p_exact_e = Expression("0.5-x[1]", domain=mesh, degree=1)
    
    # plot(u_exact_e, mesh = mesh, title = "exact velocity")
    # plot(p_exact_e, mesh = mesh, title = "exact pressure")
    
    
    inflow = DirichletBC(W.sub(0), u_exact_e, "(x[1] > 1.0 - DOLFIN_EPS) && on_boundary")
    outflow = DirichletBC(W.sub(0), u_exact_e, "(x[1] < DOLFIN_EPS) && on_boundary")
    sides = DirichletBC(W.sub(0), Constant((0.0, 0.0)) , "on_boundary && ((x[0] < DOLFIN_EPS) || (x[0] > 1.0 - DOLFIN_EPS))")
    # bc_V = DirichletBC(W.sub(0), u_exact_e, "on_boundary")
    
    # # this is to verify that I am actually applying some BC
    # U = Function(W)
    # # this applies BC to a vector, where U is a function
    # inflow.apply(U.vector())
    # outflow.apply(U.vector())
    # sides.apply(U.vector())
    # 
    # plot(U.split()[0])
    # interactive()
    # exit()
    
    bcs = [inflow, outflow, sides]
    #bcs = [bc_V]
    
    # BY MAGNE
    # a = inner(grad(u), grad(v)) * dx
    # b = q * div(u) * dx
    # 
    # lhs = a + b + adjoint(b)   # STILL NOT CLEAR 
    # rhs = inner(f, v) * dx
    # 
    # A = assemble(lhs, PETScMatrix())
    # B = assemble(rhs)
    # 
    # for bc in bcs:
    #     bc.apply(A)
    #     bc.apply(B)
    
    
    F0 = nu*inner(grad(u), grad(v))*dx
    F0 -= inner(p*Identity(2), grad(v))*dx
    F0 -= inner(f, v)*dx
    
    F1 = q*div(u)*dx
    
    F = F0 + F1
    
    a = lhs(F)
    L = rhs(F)
    
    A = assemble(a, PETScMatrix())
    b = assemble(L)
    
    for bc in bcs:
        bc.apply(A)
        bc.apply(b)
    
    
    # ----------------------- #
    # IN THIS WAY I AM SETTING THE NULL SPACE FOR THE PRESSURE
    # since p + C for some constant C is still a solution, I take the pressure with mean value 0
    
    constant_pressure = Function(W).vector()
    constant_pressure[W.sub(1).dofmap().dofs()] = 1
    null_space = VectorSpaceBasis([constant_pressure])
    A.set_nullspace(null_space)
    
    # ----------------------- #
    
    
    U = Function(W)
    
    #solve(lhs == rhs, U, bcs)
    # solve(A, U.vector(), B)  # I am putting the solution in the vector U
    solve(A, U.vector(), b)
    
    uh, ph = U.split()   # I can't use split(U), because this would not be a proper function, but I can use it in the variational form
    
    #plot(uh, title = "computed velocity")
    #plot(ph, title = "computed pressure")
    
    # IN THIS WAY I CAN PLOT AN EXPRESSION
    #plot(u_exact, mesh = mesh, title = "exact velocity")
    #plot(p_exact, mesh = mesh, title = "exact pressure")
    #interactive()
    
    # compute errors "by hands"
    # 'assemble' carrying out the integral
    L2_error_u = assemble((u_exact-uh)**2 * dx)**.5
    H1_error_u = assemble(grad(uh-u_exact)**2 * dx)**.5
    L2_error_p = assemble((p_exact - ph)**2 * dx)**.5
    # H1_error_p = assemble((grad(ph) - g)**2 * dx)**.5
   
    errsL2.append(L2_error_u)
    errsH1.append(H1_error_u)
    errsL2pressure.append(L2_error_p)
    
    print "||u - uh; L^2|| = {0:1.4e}".format(L2_error_u)
    print "|u - uh; H^1| = {0:1.4e}".format(H1_error_u)
    print "||p - ph; L^2|| = {0:1.4e}".format(L2_error_p)
    #print "||p - ph; H^1|| = {0:1.4e}".format(H1_error_p)
    
    
print errsL2
for i in range(len(h)-1):
    rates1.append(math.log(errsL2[i+1]/errsL2[i])/math.log(h[i+1]/h[i]) )

print rates1
#print range(len(h)-1)

# errsH1 and h^2 are parallel hence the convergence rate is 2

plt.loglog(h, errsH1, label = 'Error H1 norm')
plt.loglog(h, h2, label = 'h^2')
plt.loglog(h,h, label = 'h')
plt.xlabel('h')
plt.ylabel('error')
plt.title('Rate of convergence')
plt.grid(True)


# TO PUT THE LEGEND OUTSIDE THE FIGURE
# fig = plt.figure
# ax = plt.subplot(111)
# box = ax.get_position()
# ax.set_position([ box.x0, box.y0, box.width*0.8, box.height ])
# 
# ax.legend(loc = 'center left', bbox_to_anchor = (1,0.5))
# plt.show()

plt.legend(loc = 'best')
plt.savefig("convergence_sine.png")
plt.show()


# I don't plot for the polynomial because the error is 0
# (it doesn't make sense to plot it)


# in order to see whether the convergence is quadratic, I have to
# plot h^2 and if the two lines are parallel then the convergence of the
# error is quadratic


