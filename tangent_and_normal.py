from dolfin import *
import mshr

domain = mshr.Circle(Point(0.,0), 1.0)
circle_mesh = mshr.generate_mesh(domain, 20)
    
def boundary_projection(f, V):
    u, v = TrialFunction(V), TestFunction(V)
    #ds = Measure("ds", domain=circle_mesh)
    #dx = Measure("dx", domain=circle_mesh)
    a = inner(u,v) * ds + Constant(0)*inner(u,v)*dx # ensure existing diagonal
    L = inner(f, v) * ds

    # solve
    A, b = assemble_system(a, L)
    A.ident_zeros() #fix interior dofs
    Pf = Function(V)
    solve(A, Pf.vector(), b)

    return Pf

def nodal_normal(V):

    n_ufl =  FacetNormal(V.mesh())
    return boundary_projection(n_ufl, V)

    
def nodal_tangent(V):
    #import IPython; IPython.embed()
    n_ufl =  FacetNormal(V.mesh())
    t_ufl = cross(as_vector((0,0,1)), as_vector((n_ufl[0], n_ufl[1], 0)))
    # remove third component
    t_ufl = as_vector((t_ufl[0], t_ufl[1]))
    
    return boundary_projection(t_ufl, V)
    
if __name__ == "__main__":
    import mshr
    from matplotlib import pyplot
    
    V = VectorFunctionSpace(circle_mesh, "CG", 2)

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

    nicer_plot(n, title = "nodal normal")
    nicer_plot(t, title = "nodal tangent")

    nicer_plot(n_exact, title = "'exact' normal")
    nicer_plot(t_exact, title = "'exact' tangent")

    nicer_plot(n_error, title = " 'error' in the normal")
    nicer_plot(t_error, title = " 'error' in the tangent")
    pyplot.show()



       
