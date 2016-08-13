from sympy import *

x, y, t = symbols('x y t')
w01 = symbols('w01')
rho, nu = symbols('rho nu')

w02 = -2*cos(4*pi*t)*x*(x-1) 
init_printing(use_unicode=True)

#y1 = - 1/(2*pi)*sin(4*pi*t)*x*(x-1)
#w1 = -2*cos(4*pi*t)*x*(x-1)
p = 2 - x

u1 = y*(1-y)
u2 = 0

px = diff(p, x)
py = diff(p, y)
u1x = diff(u1, x)
u2x = diff(u2, x)
u1y = diff(u1, y)
u2y = diff(u2, y)
u1t = diff(u1, t)
u2t = diff(u2, t)
u1xx = diff(u1, x, x)
u2xx = diff(u2, x, x)
u1yy = diff(u1, y, y)
u2yy = diff(u2, y, y)

f1 = rho*u1t + px + rho*( (u1 - w01)*u1x + (u2 - w02)*u1y ) - rho*nu*(u1xx + u1yy)
f2 = rho*u2t + py + rho*( (u1 - w01)*u2x + (u2 - w02)*u2y ) - rho*nu*(u2xx + u2yy)

print "f1 = {}".format(f1)
print "f2 = {}".format(f2)