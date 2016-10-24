from dolfin import *

#N = [(2**n, 0.5**(2*n)) for n in range(1, 5)]

N = [2**3]
dt = 0.05

T = 1.0

rho = 1.0
mu = 1.0/8.0
theta = 1.0
t = Constant(0.0)

def sigma(u,p):
    return mu*grad(u) - p*Identity(2)

for n in N:
    
    mesh = UnitSquareMesh(n, n)
    x = SpatialCoordinate(mesh)
    normal = FacetNormal(mesh)
    
    u_exact_e = Expression(("sin(2*pi*x[1])*cos(2*pi*x[0])*cos(t)", "-sin(2*pi*x[0])*cos(2*pi*x[1])*cos(t)"), pi = pi,t = 0.0)
    w_exact_e = Expression(("sin(2*pi*x[1])*cos(t)", "0.0"), pi = pi, t = 0.0)
    p_exact_e = Expression("cos(x[0])*cos(x[1])*cos(t)",t = 0.0)
    
    u_exact = as_vector(( sin(2*pi*x[1])*cos(2*pi*x[0])*cos(t) , -sin(2*pi*x[0])*cos(2*pi*x[1])*cos(t) ))
    w_exact = as_vector(( sin(2*pi*x[1])*cos(t) , 0.0))
    p_exact = cos(x[0])*cos(x[1])*cos(t)
    
    dudt =  as_vector(( -sin(2*pi*x[1])*cos(2*pi*x[0])*sin(t) , sin(2*pi*x[0])*cos(2*pi*x[1])*sin(t) ))
    f = rho * dudt + rho * grad(u_exact)*(u_exact - w_exact) - div(mu*grad(u_exact) - p_exact*Identity(2))
    #print "dudt = {}".format(dudt(1))
    
    
    exit()
    
    # Taylor-Hood elements
    V = VectorFunctionSpace(mesh, "CG", 2)  # space for u, v
    P = FunctionSpace(mesh, "CG", 1)        # space for p, q
    W = VectorFunctionSpace(mesh, "CG", 1)       # space for w
    VP = V * P                  
    
    u, p = TrialFunctions(VP)   # u is a trial function of V, while p a trial function of P
    w = TrialFunction(W)
    
    v, q = TestFunctions(VP)
    z = TestFunction(W)
    
    up0 = Function(VP)
    u0, p0 = split(up0)
    w0 = Function(W)
    
    u0 = interpolate(u_exact_e, V)   # I want to start with u0 = u_exact_e as initial condition
    
    f0 = Constant((0.0, 0.0))
    f = Constant((0.0, 0.0))
    
    X = Function(W)  # in here I will put the displacement X^(n+1) = X^n + dt*(w^n)
    Y = Function(W)
    
    u_mid = (1.0-theta)*u0 + theta*u
    f_mid = (1.0-theta)*f0 + theta*f
    
    # Define boundary conditions
    fd = FacetFunction("size_t", mesh)
    CompiledSubDomain("near(x[0], x0) && on_boundary", x0 = x0).mark(fd, 1) # left wall (cord)     PHYSICAL BOUNDARY --> here the values of w and u have to be the same
    CompiledSubDomain("near(x[0], x1) && on_boundary", x1 = x1).mark(fd, 2) # right wall (tissue)  PHYSICAL BOUNDARY --> here the values of w and u have to be the same
    CompiledSubDomain("near(x[1], y1) ||(near(x[0], x1) && near(x[1], y1) ) && on_boundary", x1 = x1, y1 = y1).mark(fd, 3) # top wall (inlet)
    CompiledSubDomain("near(x[1], y0) ||(near(x[0], x1) && near(x[1], y0) ) && on_boundary", x1 = x1, y0 = y0).mark(fd, 4) # bottom wall (outlet)
    ds = Measure("ds", domain = mesh, subdomain_data = fd)
    
    bcu = [DirichletBC(VP.sub(0), u_exact_e, fd, 1),
           DirichletBC(VP.sub(0), u_exact_e, fd, 3),
           DirichletBC(VP.sub(0), u_exact_e, fd, 4)]
    
    bcw = DirichletBC(W, w_exact_e, "on_boundary")

    a = Constant(1./dt) * rho * inner(u - u0, v) * dx
    a += rho * inner(grad(u_mid)*(u0 - w0), v) * dx
    a += mu * inner(grad(u_mid), grad(v)) * dx
    a -= inner(p*Identity(2), grad(v)) * dx
    a -= inner(q, div(u)) * dx
    a -= inner(sigma(u_exact,p_exact)*normal, v) * ds(2)
    a -= inner(f_mid, v) * dx

    laplace = inner(grad(w), grad(z)) * dx
    
    t = dt    
    while t < (T - 1E-9):
        
        t_.assign(t)
        u_exact_e.t = t
        w_exact_e.t = t
        p_exact_e.t = t
        
        