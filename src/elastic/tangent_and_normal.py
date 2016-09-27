from dolfin import *

# Solve the linear system and return the solution
def boundary_projection(f, V):
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(u,v) * ds + Constant(0)*inner(u,v)*dx # ensure existing diagonal
    L = inner(f, v) * ds

    # solve
    A, b = assemble_system(a, L)
    A.ident_zeros() #fix interior dofs
    Pf = Function(V)    # projection of f
    solve(A, Pf.vector(), b)
    return Pf

# Return the solution of the linear system, where I put n_ufl instead of f
def nodal_normal(V):
    n_ufl =  FacetNormal(mesh)
    return boundary_projection(n_ufl, V)

    
def nodal_tangent(V):
    n_ufl =  FacetNormal(mesh)
    t_ufl = cross(as_vector((0,0,1)), as_vector((n_ufl[0], n_ufl[1], 0)))
    # remove third component
    t_ufl = as_vector((t_ufl[0], t_ufl[1]))
    
    return boundary_projection(t_ufl, V)
    
if __name__ == "__main__":
    import mshr
    from matplotlib import pyplot

    domain = mshr.Circle(Point(0.,0), 1.0)
    mesh = mshr.generate_mesh(domain, 10)
    V = VectorFunctionSpace(mesh, "CG", 2)

    n = nodal_normal(V)
    t = nodal_tangent(V)

    n_expr = Expression(("c*x[0]", "c*x[1]"), c = 1., degree = 1)
    t_expr = Expression(("-c*x[1]", "c*x[0]"), c = 1., degree = 1)

    n_exact = Function(V)
    DirichletBC(V, n_expr, "on_boundary").apply(n_exact.vector())

    t_exact = Function(V)
    DirichletBC(V, t_expr, "on_boundary").apply(t_exact.vector())
    
    # errors
    n_error = n.copy(deepcopy = True); n_error.vector()[:] -= n_exact.vector()
    t_error = t.copy(deepcopy = True); t_error.vector()[:] -= t_exact.vector()
    
    def nicer_plot(f, title = ""):
        pyplot.figure()
        p = plot(f, title = title)
        p.scale = 5.0
        pyplot.xlim(-2,2);  pyplot.ylim(-2,2)
        return p

    plot(n, title = "nodal normal")
    plot(t, title = "nodal tangent")

    #plot(n_exact, title = "'exact' normal")
    #plot(t_exact, title = "'exact' tangent")

    plot(n_error, title = " 'error' in the normal")
    plot(t_error, title = " 'error' in the tangent")
    interactive()



       
