from sympy import *

x, y, t = symbols('x y t')

init_printing(use_unicode=True)

rho = 1.0
mu = 1.0/8.0

u1 = 1.0/(2*mu) * y * (1-y)
u2 = 0.0
p = 1 - x

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

f1 = rho*u1t + ( u1*u1x + u2*u1y ) - mu*(u1xx + u1yy) + px
f2 = rho*u2t + ( u1*u2x + u2*u2y ) - mu*(u2xx + u2yy) + py

print "f1 = {}".format(f1)
print "f2 = {}".format(f2)