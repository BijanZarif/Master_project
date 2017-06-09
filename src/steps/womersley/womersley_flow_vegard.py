from dolfin import *
from pylab import where, find,array, linspace, sin, cos, sinh, cosh, log
import scipy.interpolate as Spline
import pylab as plt
set_log_active(False)

h = 2
l = 60

P1 = Point(0,-h)
P2 = Point(l,h)

E1 = [1]
H = [1]
E2 = [1]

print r'\begin{center}'+ '\n'  +r'  \begin{tabular}{l | l | l | l | l | l}' + '\n'
print r'    $\Delta t$ & dofs & L2 error & rate & H1 error & rate \\ \hline'

for j in range(1,5):
	Nx = 2*2**j
	Ny = 2*2**j

	mesh = RectangleMesh(P1,P2,Nx,Ny)

	V = VectorFunctionSpace(mesh,'CG',2)
	P = FunctionSpace(mesh,'CG',1)
	W = VectorFunctionSpace(mesh,'CG',3)
	VP = V*P

	u,p = TrialFunctions(VP)
	v,q = TestFunctions(VP)

	eps = 1e-8

	class Walls(SubDomain):
		def inside(self,x,on_bnd):
			return abs(h-abs(x[1])) < eps and on_bnd
	class InOut(SubDomain):
		def inside(self,x,on_bnd):
			return x[0] < eps or x[0] > l - eps and on_bnd



	bnd = FacetFunction('size_t',mesh)
	Walls().mark(bnd,1)
	InOut().mark(bnd,2)

	om = 2*pi
	C = 1000
	rho_f = Constant(1./1000)		# g/mm
	nu_f = Constant(0.658)			# mm**2/s
	mu_f = Constant(nu_f*rho_f)		# g/(mm*s)
	K = float(sqrt(om/(2*nu_f)))

	pressure = Expression('rho*C*cos(omega*t)*(x[0]-L)',t=0,L=l,omega=om,C=C,rho=rho_f)

	def ss(x):
		return sin(x)*sinh(x)
	def cc(x):
		return cos(x)*cosh(x)

	def f1(w,x3):
		return cc(K*x3)*cc(K*h) + ss(K*x3)*ss(K*h)
	def f2(w,x3):
		return cc(K*x3)*ss(K*h) - ss(K*x3)*cc(K*h)
	def f3(w):
		return cc(w)**2 + ss(w)**2

	def v_exact(x,y,t):
		return C/om*((f1(om,y)/f3(K*h)-1)*sin(om*t) - f2(om,y)/f3(K*h)*cos(om*t))


	class v_analytical(Expression):
		def __init__(self,t,h=2,om=2*pi,C=10):
			self.t, self.h, self.om, self.C = t,h,om,C

		def eval(self,values,x):
			values[1] = 0
			values[0] = v_exact(0,x[1],(self.t))
		def value_shape(self):
			return (2,)




	'''
	plt.ion()

	for t in range(0,200):
		plt.plot(y,v_exact(0,y,0.04*t))
		plt.axis([-h, h, -2, 2])
		plt.draw()
		plt.show()
		plt.clf()
	sys.exit()
	'''
	noslip = Constant((0,0))
	ds = Measure('ds')[bnd]
	bcs = [DirichletBC(VP.sub(0),noslip,bnd,1)]


	ufile = File('results_channel/v.pvd')
	pfile = File('results_channel/p.pvd')
	dt = 1e-6#0.2*2**-j
	VP_ = Function(VP)
	BE = True
	if BE:
		u1 = interpolate(v_analytical(0,h=h,om=om,C=C),V)
	else:
		u1 = interpolate(v_analytical(dt,h=h,om=om,C=C),V)
		u_1 = interpolate(v_analytical(0,h=h,om=om,C=C),V)#Function(V)
	u0 = interpolate(v_analytical(0,h=h,om=om,C=C),V)
	k = Constant(dt)



	n = FacetNormal(mesh)
	def epsilon(u):
		return sym(grad(u))

	eps = mesh.hmin()*0.05
		# WeakForm

	n = FacetNormal(mesh)
	if BE:
		a = rho_f*1/k*inner(u,v)*dx \
			+ rho_f*inner(grad(u)*u0,v)*dx \
			+ mu_f*inner(grad(u)+grad(u).T, grad(v))*dx \
			- inner(div(v),p)*dx \
			- inner(div(u),q)*dx \
			- mu_f*inner(grad(u).T*n,v)*ds(2) \
		    + eps**-2*(inner(u, v) - inner(dot(u, n), dot(v, n)))*ds(2)

		L = rho_f/k*inner(u1,v)*dx  - inner(pressure*n,v)*ds(2)
		
		t = dt
		T = dt
	else:

		a = rho_f/k*inner(u,v)*dx \
			+ 0.5*rho_f*inner(grad(u)*(3./2*u1-0.5*u_1),v)*dx \
			+ mu_f*inner(epsilon(u), grad(v))*dx \
			- inner(div(v), p)*dx \
			- inner(div(u), q)*dx \
			- mu_f*inner(grad(u).T*n,v)*ds(2) \
			+ eps**-2*(inner(u, v) - inner(dot(u, n), dot(v, n)))*ds(2)

		L = rho_f/k*inner(u1,v)*dx  - inner(pressure*n,v)*ds(2) \
			- mu_f*inner(epsilon(u1), grad(v))*dx \
			- 0.5*rho_f*inner(grad(u1)*(3./2*u1-0.5*u_1),v)*dx
		t = dt
		T = dt
	
#Err = 0.0021 for BE  T=0.7

#Err = 0.065 for BE T=0.1 C =1000

	x_l = l#l/2
	y = linspace(-h,h,Ny+1)

	xa = where(mesh.coordinates()[:,0] == x_l)
	plt.ion()

	while t < T + DOLFIN_EPS:
		#if t < 2.0:
		#	pressure.amp = 1.
		pressure.t=t
		b = assemble(L)
		err = 10
		k_iter = 0
		max_iter = 8
		if BE:
			while err > 1E-10 and k_iter < max_iter:
				A = assemble(a)
				[bc.apply(A,b) for bc in bcs]
				solve(A,VP_.vector(),b,'lu')
				u_,p_ = VP_.split(True)
				err = errornorm(u_,u0,degree_rise=3)
				k_iter += 1
				u0.assign(u_)
				#print 'k: ',k_iter, 'error: %.3e' %err
		else:
			A = assemble(a)
			[bc.apply(A,b) for bc in bcs]
			solve(A,VP_.vector(),b,'lu')
			u_,p_ = VP_.split(True)

		ufile << u_
		pfile << p_
		u1.assign(u_)
		#print 't=%.4f'%t
	
		u_val = u_.compute_vertex_values()
		'''
		plt.plot(y,u_val[xa])
		plt.plot(y,v_exact(x_l,y,t))
		plt.plot(y,v_exact(x_l,y,dt))
		plt.legend(['comp','exact', 't=dt'])
		plt.axis([-h,h,-4,4])
		plt.draw()
		plt.clf()
		'''
		t+=dt

	u_e = interpolate(v_analytical(t-dt),W)
	
	#E_sum = sqrt(1./Nx*sum((v_exact(xa,y,t-dt) - u_val[xa])**2))
	#print 'ERROR:',E_sum
	H1 = errornorm(u_,u_e,'h1',degree_rise=3)
	L2 = errornorm(u_,u_e,degree_rise=3)
	H.append(mesh.hmin())
	E2.append(L2)
	E1.append(H1)
	R2 = log(E2[-2]/E2[-1])/log(H[-2]/H[-1])
	R1 = log(E1[-2]/E1[-1])/log(H[-2]/H[-1])
	dim = VP.dim()
	print '%6.2d & %6d & %.2e & %.3f & %.2e & %.3f'%(Nx,dim,L2,R2,H1,R1) +r' \\' + ' \hline' 

print '    \hline' + '\n' + '  \end{tabular}' +'\n' +'\end{center}'


A = assemble(a)
b = assemble(L)
up = Function(VP)
[bc.apply(A,b) for bc in bcs]
solve(A,up.vector(),b)
u_,p_ = up.split(True)
plot(p_)
interactive()
ufile << u_
pfile << p_

