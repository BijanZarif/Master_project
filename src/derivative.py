from sympy import *

x, y, t = symbols('x y t')
w01, w02 = symbols('w01 w02')

init_printing(use_unicode=True)

rho = 1.0

y1 = - 1/(2*pi)*sin(4*pi*t)*x*(x-1)
w1 = -2*cos(4*pi*t)*x*(x-1)
p = 0.5 - y

u1 = (y - y1)*(y - y1 - 1)
u2 = w1

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

f1 = u1t + px + ( (u1 - w01)*u1x + (u2 - w02)*u1y ) + (u1xx + u1yy)
f2 = u2t + py + ( (u1 - w01)*u2x + (u2 - w02)*u2y ) + (u2xx + u2yy)

print "f1 = {}".format(f1)
print "f2 = {}".format(f2)

