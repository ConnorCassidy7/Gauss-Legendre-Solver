import cmath as cm                          # math library (complex)
import math as m                            # math library
import numpy as np                          # basic functions, linear algebra, etc.
import scipy.special as sp                  # special functions
import numpy.random as rn                   # random numbers
import matplotlib.pyplot as plt             # plotting library
from scipy import integrate                 # library for integration
from mpl_toolkits import mplot3d            # for 3d plotting 
from matplotlib.colors import hsv_to_rgb    # convert the color from HSV coordinates to RGB coordinates
from colorsys import hls_to_rgb             # convert the color from HLS coordinates to RGB coordinates    

n = 4
a = 0.1
b = 1
epsilon = 1e-3
m = 1

meshpoints = np.polynomial.legendre.leggauss(n)[0]
weights = np.polynomial.legendre.leggauss(n)[1]

def w(k):
    return np.sqrt(m**2 + k**2)

def kallenlambda(x,y,z):
    return x**2 + y**2 + z**2 - 2*(x*y + y*z + z*x)

def scmpair(E,k):
    return (E-w(k))**2 - k**2

def J(x):
    return np.where(
        x <= 0, 0, 
        np.where(x >= 1, 1, np.exp((-1/x)*np.exp(-1/(1-x))))
    )

def kfun(E, Ecmpair):
    return (1/(2*E)) * cm.sqrt(kallenlambda(E**2, Ecmpair**2, m**2))

def H(p,k, E):
    return J(scmpair(E,p) / (4*m**2)) * J(scmpair(E,k) / (4*m**2))
    
def alpha(p,k,E):
    return (E - w(k) - w(p))**2 - p**2 - k**2 - m**2

def G(p,k, E):
    return -(H(p,k, E)/(4*p*k)) * np.log((alpha(p,k, E) - 2*p*k + 1j*epsilon)/(alpha(p,k, E) + 2*p*k + 1j*epsilon))


def M2(k, E):
    return (16.0 * np.pi *cm.sqrt(scmpair(E, k))) / (-1.0 / (E/m)  - 1j * cm.sqrt(scmpair(E, k)/4 - m**2) )

M2_vectorized = np.vectorize(M2)

def sb(E):
    return 4*(m**2 - (1/(E/m))**2)

def smallg(E):
    return 8 * np.sqrt(2*np.pi *np.sqrt(sb(E)) * (E/m))

def q(E):
    return 1/(2*E) * np.sqrt(kallenlambda(E**2, sb(E), m**2))

def Mphib(globeE):

    def g(t):
        return -G(t, q(globeE), globeE)

    def K(t,s):
        return -(s**2 / ((2*np.pi)**2 * w(s))) * G(t,s, globeE) * M2_vectorized(s,globeE)

    def intconvert(p):
        return ((b-a)/2)*p + ((b+a)/2)

    def weightconvert(w):
        return ((b-a)/2)*w

    def vectorize(g, points):
        i = 0
        N = len(points) - 1
        list = []
        while i <= N:
            list = list + [g(intconvert(points[i]))]
            i = i + 1
        return list

    def matrixize(K, points):
        ti = 0
        N = len(points) - 1
        matrix = np.empty((0, N+1))
        while ti <= N:
            row = []
            si = 0
            while si <= N:
                row = row + [weightconvert(weights[si])*K(intconvert(points[ti]), intconvert(points[si]))]
                si = si + 1
            print('row:', row)
            matrix = np.append(matrix, np.array([row]), axis=0)
            ti = ti + 1
        return matrix

    def fsolve(gvec, Kmat):
        M = np.identity(n) - (Kmat)
        inverse = np.linalg.inv(M)
        return np.dot(inverse, gvec)

    def integrate(fvec, points):
        sum = 0
        i = 0
        N = len(points) - 1
        while i <= N:
            sum = sum + fvec[i]*weightconvert(weights[i])
            i = i + 1
        return sum

    gvec = vectorize(g, meshpoints)
    kmat = matrixize(K, meshpoints)
    fvec = fsolve(gvec, kmat)

    def f(t):
        sum = g(t)
        i = 0
        N = len(fvec) - 1
        while i <= N:
            sum = sum + fvec[i]*K(t, intconvert(meshpoints[i]))*weightconvert(weights[i])
            i = i + 1
        return sum
    
    return (smallg(globeE))**2  * f(q(globeE))

xdata = np.linspace(1, 10, 10)
ydata = Mphib(xdata)
#plt.plot(points, fvec)
plt.plot(xdata,np.real(ydata))
#plt.plot(xdata,np.imag(ydata))
plt.show()

#print(matrixize(K, meshpoints))