## STOKES PROBLEM ##

# - div( nu * grad(u) - pI ) = f
# div( u ) = 0

from numpy import *
from math import *
from matplotlib import pyplot as plt
from fenics import *
from mshr import *   # I need this if I want to use the functions below to create a mesh


N = [2**2, 2**3, 2**4, 2**5, 2**6]
h = [1./i for i in N]
h2 = [1./(i**2) for i in N]
errsL2 = []
errsH1 = []
errsL2pressure = []
errsH1pressure = []
rates1 = []
rates2 = []
rates3 = []
gamma = Constant(100.0)
g = Constant((0.0, 0.0))

for n in N:

    mesh = UnitSquareMesh(n,n)
    
    normal = FacetNormal(mesh)
    h = CellSize(mesh)
    
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
    nu = Constant(1.0/8.0)
    
    
    # I have to remember that the u_exact has to satisfy as well the boundary conditions (and not only the system of equations)
    # that's why there's the pi*x[0], so the sin is 0 on the right boundary (i.e. x[0] = 1))
    u_exact = as_vector((0, sin(pi*x[0]))) # to use as a solution to verify the convergence 
    p_exact = 0.5 - x[1]            # this function has mean value zero (its integral in [0,1] x [0,1] is zero)
                                    # hence, I can use it as exact solution to compare it with the numerical solution
                                    # since I put the constraint that mean_value(pressure) = 0
                                    # which is equivalent to setting the null space of the matrix A as done later in the code
    
    f = - nu*div(grad(u_exact)) + grad(p_exact)   # I changed the sign in the gradient
    
    # Since the pressure is defined up to some constant, we compare the gradients
    g =  nu*div(grad(u_exact)) + f             # pressure gradient
    
    
    u_exact_e = Expression((" 0 ", "sin(pi*x[0])" ))
    p_exact_e = Expression("0.5-x[1]", domain=mesh, degree=1)
    
    # plot(u_exact_e, mesh = mesh, title = "exact velocity")
    # plot(p_exact_e, mesh = mesh, title = "exact pressure")
    
    # -------- BC -------

    # Define boundary conditions
    fd = FacetFunction("size_t", mesh)
    CompiledSubDomain("near(x[0], 0.0) && on_boundary").mark(fd, 1) # left wall (cord)    
    CompiledSubDomain("near(x[0], 1.0) && on_boundary").mark(fd, 2) # right wall (tissue)  
    CompiledSubDomain("near(x[1], 1.0) && on_boundary").mark(fd, 3) # top wall (inlet)
    CompiledSubDomain("near(x[1], 0.0) && on_boundary").mark(fd, 4) # bottom wall (outlet)
    ds = Measure("ds", domain = mesh, subdomain_data = fd)
    #plot(fd)
    #interactive()

    # bcs = [DirichletBC(W.sub(0), u_exact_e, fd, 3),
    #        DirichletBC(W.sub(0), u_exact_e, fd, 4),
    #        DirichletBC(W.sub(0), Constant((0.0, 0.0)), fd, 1),
    #        DirichletBC(W.sub(0), Constant((0.0, 0.0)), fd, 2)]
    
    ## check the BC are correct
    #U = Function(W)
    #for bc in bcs: bc.apply(U.vector())
    #plot(U.sub(0))
    #interactive()
    
    
    
    # BC for Nitsche, I am not setting the BC strongly on the right wall
    bcs = [DirichletBC(W.sub(0), u_exact_e, fd, 3),
           DirichletBC(W.sub(0), u_exact_e, fd, 4),
           DirichletBC(W.sub(0), Constant((0.0, 0.0)), fd, 1)]
    
    # ---------------
    
    F0 = nu*inner(grad(u), grad(v))*dx
    F0 -= inner(p*Identity(2), grad(v))*dx
    F0 -= inner(f, v)*dx
    
    # ------ Nitsche
    F0 += inner(p*normal, v)*ds(2)
    
    F0 -= nu * inner(grad(u) * normal,v) * ds(2)
    F0 -= nu * inner(grad(v) * normal,u) * ds(2)
    F0 += gamma * h**-1 * inner(u,v) * ds(2) 
    F0 += nu * inner(grad(v) * normal, g) * ds(2)
    F0 -= gamma * h**-1 * inner(g,v) * ds(2)
    
    # -------
    
    F1 = -q*div(u)*dx  # continuity equation 
    
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
    #H1_error_p = assemble((grad(ph) - g)**2 * dx)**.5
   
    errsL2.append(L2_error_u)
    errsH1.append(H1_error_u)
    errsL2pressure.append(L2_error_p)
    #errsH1pressure.append(H1_error_p)
    
    print "||u - uh; L^2|| = {0:1.4e}".format(L2_error_u)
    print "|u - uh; H^1| = {0:1.4e}".format(H1_error_u)
    print "||p - ph; L^2|| = {0:1.4e}".format(L2_error_p)
    #print "||p - ph; H^1|| = {0:1.4e}".format(H1_error_p)
    
    
#print errsL2
# for i in range(len(h)-1):
#    rates1.append(math.log(errsH1[i+1]/errsH1[i])/math.log(h[i+1]/h[i]) )

for i in range(len(h)-1):
    rates1.append(math.log(errsL2pressure[i+1]/errsL2pressure[i])/math.log(h[i+1]/h[i]) )

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
fig = plt.figure
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([ box.x0, box.y0, box.width*0.8, box.height ])

ax.legend(loc = 'center left', bbox_to_anchor = (1,0.5))
plt.show()

#plt.legend(loc = 'best')
#plt.savefig("convergence_sine.png")
#plt.show()


# I don't plot for the polynomial because the error is 0
# (it doesn't make sense to plot it)


# in order to see whether the convergence is quadratic, I have to
# plot h^2 and if the two lines are parallel then the convergence of the
# error is quadratic


