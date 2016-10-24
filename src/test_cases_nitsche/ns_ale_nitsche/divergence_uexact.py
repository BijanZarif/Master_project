from sympy import *

#x = symbols('x')
y = symbols('y')
#x, y = symbols('x y')
x = 0.0
#x = 1.0
#y = 1.0

u1 = 0.5 * (x-2)**2 * ((1-2*y)*sin(2*pi*y) + y*(1-y)*2*pi*cos(2*pi*y))
u2 = y*(1-y)*sin(2*pi*y)*(2-x)

#u1x = diff(u1, x)
#u2y = diff(u2, y)

#divu = u1x + u2y


print "(u1,u2)={}".format((u1,u2))
#print "d(u1)/dx = {}".format(u1x)
#print "d(u2)/dy = {}".format(u2y)
#print "div(u) = {}".format(divu)