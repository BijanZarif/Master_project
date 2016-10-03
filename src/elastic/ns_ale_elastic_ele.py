## ALE + NAVIER-STOKES EQUATIONS ##

# rho * du/dt + rho * (grad(u) . u - w) - div( mu * grad(u) - pI ) = f
# div( u ) = 0
import sys
sys.path.append("~/Repositories/Master_project/src/magne")
from dolfin import *
from tangent_and_normal import *

#N = [2**2, 2**3, 2**4, 2**5, 2**6]
NN = [2**4]
T = 40
#T = 1.5
mu = 1.0
rho = 1.0
theta = 1.0     # 0.5 for Crank-Nicolson, 1.0 for backwards
gamma = 20   # constant for Nitsche method

use_projected_normal = True

# VALUES OF k SHOULD BE NEGATIVE (OR I CHANGE THE SIGN IN THE VARIATIONAL FORM)
# k BIG: the tissue is stiff, k SMALL: the tissue is more flexible
k = Constant(1e-1)      # elastic
#k = - Constant(1e6)       # stiff
k_bottom = 1e2
k_top = 1e2
k_middle = 1e-1

#k = Expression( "(x[1]<1)*k_bottom + (x[1]>1.9)*k_top + (x[1]>1 || x[1]<1.9)*k_middle", k_bottom = k_bottom, k_top = k_top, k_middle = k_middle )
#k = Expression( "(x[1]<0.1)*k_bottom + (x[1]>0.9)*k_top + (x[1]>0.1 || x[1]<0.9)*k_middle", k_bottom = k_bottom, k_top = k_top, k_middle = k_middle )

# -------

dt = 0.05
g = Constant(0.0)


x0, x1 = 0.0, 1.0
y0, y1 = 0.0, 1.0


for N in NN : 
   
    mesh = RectangleMesh(Point(x0, y0), Point(x1, y1), N, 2*N)  
    x = SpatialCoordinate(mesh)
    

    # h = CellSize(mesh)
    
    h = mesh.hmax()
    # Taylor-Hood elements
    V = VectorFunctionSpace(mesh, "CG", 2)  # space for u, v
    P = FunctionSpace(mesh, "CG", 1)        # space for p, q
    W = VectorFunctionSpace(mesh, "CG", 1)       # space for w
    VP = V * P                  
    
    u, p = TrialFunctions(VP)   # u is a trial function of V, while p a trial function of P
    w = TrialFunction(W)
    
    v, q = TestFunctions(VP)
    z = TestFunction(W)


    # normal = FacetNormal(mesh)
    normal = FacetNormal(V.mesh())
    tangent = cross(as_vector((0,0,1)), as_vector((normal[0], normal[1], 0)))
    tangent = as_vector((tangent[0], tangent[1]))

    if use_projected_normal == True:
        tangent = nodal_tangent(V)
        normal = nodal_normal(V)
        #tangent = boundary_projection(tangent, V)
        #normal = boundary_projection(normal, V)
    
    # Defining the normal and tangential components    
    un = dot(u, normal)
    vn = dot(v, normal)
    ut = dot(u, tangent)
    vt = dot(v, tangent)
    
    # The functions are initialized to zero
    up0 = Function(VP)
    u0, p0 = split(up0)  # u0 is not a function but "part" of a function, just a "symbolic" splitting?
                         # REMEMBER: In this way when I update the up0 also the u0 and p0 get updated
                         # This is different from up0.split which I can use if I want to plot
    
    w0 = Function(W)
    
    X = Function(W)  # in here I will put the displacement X^(n+1) = X^n + dt*(w^n)
    Y = Function(W)  # by default this is 0 in all the components, Y is the adding displacement (what we add in order to move the mesh)
    
    f0 = Constant((0.0, 0.0))
    f = Constant((0.0, 0.0))
    t = 0.0
    # u_inlet = Expression(("0.0", "(-1*fabs(x[0]*(x[0] - 1)))*sin(t*2*pi)"), t=t, degree = 2)
    u_inlet = Expression(("0.0", "0.1*(-1*fabs(x[0]*(x[0] - 1)))"), t=t, degree = 2)
    
    # parabolic initial flow
    #up0.assign(interpolate(Expression(("0.0", "-1*fabs(x[0]*(x[0] - 1))", "0.0"), degree = 2), VP))
    #plot(u0, interactive = True)
    
    u_mid = (1.0-theta)*u0 + theta*u
    f_mid = (1.0-theta)*f0 + theta*f
    p_mid = (1.0-theta)*p0 + theta*p
    
        
    # Define boundary conditions
    fd = FacetFunction("size_t", mesh)
    CompiledSubDomain("near(x[0], x0) && on_boundary", x0 = x0).mark(fd, 1) # left wall (cord)     PHYSICAL BOUNDARY --> here the values of w and u have to be the same
    CompiledSubDomain("near(x[0], x1) && on_boundary", x1 = x1).mark(fd, 2) # right wall (tissue)  PHYSICAL BOUNDARY --> here the values of w and u have to be the same
    CompiledSubDomain("near(x[1], y1) && (x[0] != x1) && on_boundary", x1 = x1, y1 = y1).mark(fd, 3) # top wall (inlet)
    CompiledSubDomain("near(x[1], y0) && (x[0] != x1) && on_boundary", x1 = x1, y0 = y0).mark(fd, 4) # bottom wall (outlet)
    # CompiledSubDomain("near(x[1], y1) && on_boundary", x1 = x1, y1 = y1).mark(fd, 3) # top wall (inlet)
    # CompiledSubDomain("near(x[1], y0) && on_boundary", x1 = x1, y0 = y0).mark(fd, 4) # bottom wall (outlet)

    ds = Measure("ds", domain = mesh, subdomain_data = fd)
    
    #plot(fd)
    #interactive()
    
    # Here I need to impose just the Dirichlet conditions. The ones regarding the stresses were already encountered in the
    # weak formulation
    bcu = [DirichletBC(VP.sub(0), u_inlet, fd, 3),   # inlet at the topo wall
           DirichletBC(VP.sub(0), Constant((0.0,0.0)), fd, 1)]   # left wall
    bcw = [DirichletBC(W, Constant((0.0,0.0)), fd, 1),
            DirichletBC(W, u0, fd, 2),                      # or   DirichletBC(W, dot(u0,unit)*unit, fd, 2)]   # if unit = (1,0)
            DirichletBC(W.sub(1), Constant(0.), fd, 4),
           #DirichletBC(W, Constant((0.,0.)), fd, 4),
           DirichletBC(W, Constant((0.,0.)), fd, 3)]
                           

    # bcw = [DirichletBC(W, Constant((0.0,0.0)), fd, 1),
    #         DirichletBC(W, u0, fd, 2),
    #        DirichletBC(W, Constant((0.,0.)), fd, 4),
    #        DirichletBC(W, Constant((0.,0.)), fd, 3)]
    
    # check the BC are correct
    #U = Function(VP)
    #for bc in bcu: bc.apply(U.vector())
    #plot(U.sub(0))
    #interactive()
    

    # -------- NAVIER-STOKES --------
    # Weak formulation
    dudt = Constant(1./dt) * Constant(rho) * inner(u - u0, v) * dx
    a = ( Constant(rho) * inner(grad(u_mid)*(u0 - w0), v)   # ALE term
         + Constant(mu) * inner(grad(u_mid), grad(v))
         - p * div(v)                               # CHECK THIS TERM
         - q * div(u)                               # from the continuity equation, maybe put a - q*div(u) to make the system symmetric
         - inner(f,v) ) * dx
    
    # Boundary term with elastic constant
    # I put a minus in this term, as it follows from the computations of the variational form. But then I want the term [ky] to be negative,
    # so I put a negative value of the [k]
 
    b = k * inner(dot(X + dt * u, normal), vn) * ds(2)    # what should I use here as displacement?
                                                                                                
    c = ( - dot(grad(u)*normal, tangent) * vt - dot(grad(v)*normal, tangent) * ut
          + Constant(gamma)/h * ut*vt + dot(grad(v)*normal, tangent) * g - Constant(gamma)/h * g*vt ) * ds(2)
    
    # Bilinear and linear forms
    F = dudt + a + b + c
    
    a0, L0 = lhs(F), rhs(F)    
    
    # -------- POISSON PROBLEM FOR w:  div( grad(w) ) = 0 --------
    a1 = inner(grad(w), grad(z))*dx
    L1 = dot(Constant((0.0,0.0)),z)*dx   
    # ----------------------
    
    # I want to store my solutions here
    VP_ = Function(VP)   
    W_ = Function(W)
    

    solver = PETScLUSolver()

    #file = File("u.pvd")

    while t <= T + 1E-9:
            
        print "Solving for t = {}".format(t)  
        
        # Solving the Navier-Stokes equations
        # I need to reassemble the system
        A = assemble(a0)
        b = assemble(L0)
        
        # I need the reapply the BC to the new system
        for bc in bcu:
            bc.apply(A, b)
        
        # Ax = b, where U is the x vector    
        solver.solve(A, VP_.vector(), b)
        
        # Solving the Poisson problem
        A1 = assemble(a1)
        b1 = assemble(L1)
         
        #bcw[3] = DirichletBC(W, dot(u0, unit) * unit, fd, 2)   # updating the boundary value u0
        bcw[1] = DirichletBC(W, u0, fd, 2)   # updating the boundary value u0
        
        
        for bc in bcw:
           bc.apply(A1, b1)
        
        solve(A1, W_.vector(), b1)

        up0.assign(VP_)   # the first part of VP_ is u0, and the second part is p0
        w0.assign(W_)
        
        
        # Compute the mesh displacement
        Y.vector()[:] = w0.vector()[:]*dt
        X.vector()[:] += Y.vector()[:]

        # check the values of the tangential and normal components
        aa, bb = assemble(inner(u0,normal)**2 * ds(2))**.5, assemble(inner(u0, tangent)**2 * ds(2))**.5
        print "{0:1.4e} {1:1.4e}".format(aa, bb)
        
        # Move the mesh
        ALE.move(mesh, Y)
        mesh.bounding_box_tree().build(mesh)

        plot(mesh)
        
        # WE NEED THIS TO UPDATE THE NORMAL AND TANGENT, OTHERWISE WE ALWAYS USE THE NORMAL AND TANGENT FROM THE INITIAL MESH
        if use_projected_normal == True:
           normal.assign(nodal_normal(V))
           tangent.assign(nodal_tangent(V))
    
        un = dot(u, normal)
        vn = dot(v, normal)
        ut = dot(u, tangent)
        vt = dot(v, tangent)
        
        u0, p0 = VP_.split()
        plot(u0, key="u0")
        #file << u0

        t += dt
        
        u_inlet.t = t