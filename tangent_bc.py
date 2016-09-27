from dolfin import *
from tangent_and_normal import nodal_tangent, nodal_normal


class DirectionalDirichletBC(object):
    def __init__(self, V, values):
        self.V = V
        self.values = values

    def get_boundary_dofs(self):
        """ return the dofs on the boundary """
        from numpy import array, where

        if V.num_sub_spaces() == 0:
            one = 1
        else:
            one = (1,) * V.num_sub_spaces()

        v = Function(V).vector()
        DirichletBC(self.V, one, "on_boundary").apply(v)
        
        boundary_dofs = where(v > 0)[0].astype("intc")
        bs = self.V.dofmap().block_size()
        boundary_dofs.shape = (len(boundary_dofs)/ bs, bs)

        self.boundary_dofs = boundary_dofs
        return boundary_dofs

    def get_direction(self):
        raise NotImplementedError

    def evaluate_tangent(self):
        from numpy import where, sum
        if not hasattr(self, "boundary_dofs"):
            self.get_boundary_dofs()

        if not hasattr(self, "t"):
            self.get_direction()

        # exctract boundary values
        t_array = self.t_array = self.t.vector().array()[self.boundary_dofs]

        # determine BC dofs
        self.bc_dofs = where(abs(t_array[:,0]) > abs(t_array[:,1]), 
                             *self.boundary_dofs.T)

        # compute squared magnitude (vector not exactly normalized)
        self.n2= sum(t_array**2, 1)

    def apply(self, other):
        if isinstance(other, GenericMatrix):
            self.apply_matrix(other)

        elif isinstance(other, GenericVector):
            self.apply_vector(other)

    def apply_matrix(self, A):
        from numpy import array
        if not hasattr(self, "bc_dofs"):
            self.evaluate_tangent()

        # zero bc rows
        A.zero_local(self.bc_dofs)

        # loop over boundary dofs
        for j in xrange(len(self.bc_dofs)):
            rows = self.bc_dofs[[j]]
            cols = self.boundary_dofs[j]
            vals = self.t_array[j] #.vector()[cols]
            A.set(vals, rows, cols)

        A.apply("insert")

    def apply_vector(self, b):
        b[self.bc_dofs] = self.values * self.n2


class TangentDirichletBC(DirectionalDirichletBC):
    def get_direction(self):
        from tangent_and_normal import nodal_tangent
        t = self.t = nodal_tangent(self.V)
        return t

class NormalDirichletBC(DirectionalDirichletBC):
    def get_direction(self):
        from tangent_and_normal import nodal_normal
        t = self.t = nodal_normal(self.V)
        return t


if __name__ == "__main__":
    import mshr
    from matplotlib import pyplot
    parameters["plotting_backend"] = "matplotlib"
    apply_nitsche = False

    domain = mshr.Circle(Point(0.,0), 1.0)
    mesh = mshr.generate_mesh(domain, 40)

    V = VectorFunctionSpace(mesh, "CG", 2)

    x = SpatialCoordinate(mesh)

    f = Constant((0, 0.)) 
    g = Constant(0.)
    h = 1.0

    u, v = TrialFunction(V), TestFunction(V)
    n = FacetNormal(mesh)
    t = cross(as_vector((0,0,1)), as_vector((n[0], n[1], 0)))
    t = as_vector((t[0], t[1]))

    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    if apply_nitsche:
        bcs = []

        a += (- inner(grad(u)*n,v) * ds
              - inner(grad(v)*n,u) * ds
              + Constant(100) * CellSize(mesh)**-1 * inner(u,v) * ds)
        L += (- inner(grad(v)*n, t) * ds
              + Constant(100) * CellSize(mesh)**-1 * inner(t,v) * ds)

    else:
        bc_t = TangentDirichletBC(V, h)
        bc_n = NormalDirichletBC(V, 0)
        bcs = [bc_t]


    A = assemble(a)
    b = assemble(L)

    for bc in bcs:
        bc.apply(A)
        bc.apply(b)

    uh = Function(V)
    solve(A, uh.vector(), b)

    # compare with exact solution
    U = as_vector((-x[1], x[0])); Uh = project(U, V)
    
    error = assemble((U-uh)**2 * dx)**0.5
    print "Error = {0:1.4e}".format(error)

    pyplot.figure();plot(uh)
    pyplot.xlim(-1.5, 1.5); pyplot.ylim(-1.5, 1.5)
    pyplot.figure();plot(project(U-uh,V))
    pyplot.xlim(-1.5, 1.5); pyplot.ylim(-1.5, 1.5)
    pyplot.show()
    
