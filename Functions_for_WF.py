import math
import sympy
import numpy as np


r, phi, theta = sympy.symbols("r phi theta", real=True)  # spherical coordinates = r, theta, phi
a = sympy.Symbol('a', real = True)
a_val = 5.2946541*(10**(-11))                            #reduced Bohr radius


'''
INPUTS

n = principal quantum number
l = azimuthal quantum number
m = magnetic quantum number
'''

n,l,m = 3,2,1


#___________________________________Defining the Functions__________________________________


''' 
Naming convention 

func(r,y) = return func_ry

var = function that contains 'r','psi', 'theta'
const = constant, i.e. anything that does not have r
        (can have internat function variables like l,m,n,A)

(-1)^m = Condon-Shortly phase
'''

#=================================================================================================
# Legendre Polynomial P(r) in terms of m,l: m>=0
def P(func,l,m):
    const1 = 1/((2**l)*math.factorial(l))
    var1 = (1-(r**2))**(m/2)
    
    #differential Term
    f = ((r**2)-1)**l
    df = sympy.Symbol.diff(f,r,l+m).simplify()
    
    P_lm = ((-1)**m)*const1*var1*df
   
    return P_lm.subs(r, func)
#=================================================================================================

#=================================================================================================
#Angular Functions

# Laplace spherical harmonics Y(theta,psi) in terms of m,l; m>=0 
                            #--------Complex Part----------
def Y(l,m):
    const1 = ((2*l)+1)/(4*math.pi)
    const2 = (math.factorial(l-m))/(math.factorial(l+m))
    var1 = sympy.exp(complex(0,1)*m*phi)
    
    Y_ml = ((-1)**m)*math.sqrt(const1*const2)*P(sympy.cos(theta),l,m).trigsimp()*var1
    return Y_ml

def conj_Y(l,m):
    const1 = ((2*l)+1)/(4*math.pi)
    const2 = (math.factorial(l-m))/(math.factorial(l+m))
    var1 = sympy.exp(-complex(0,1)*m*phi)
    
    c_Y_ml = ((-1)**m)*math.sqrt(const1*const2)*P(sympy.cos(theta),l,m).trigsimp()*var1
    return c_Y_ml

def Y2(l,m):
    return Y(l, m)*conj_Y(l, m)

                            #----------Real Part---------
def S(l,m):
    if m>0:
        return (math.sqrt(1/2)*(Y(l,-m) + Y(l,m))).simplify()
    elif m<0:
        return (complex(0,1)*math.sqrt(1/2)*(Y(l,-m) - Y(l,m))).simplify()
    elif m==0:
        return Y(l,m)

# Genralized Laguerre Polynomial L(r) operator form from Rodrigues form in terms of n and A
def gL(func,A,n):                     
    var1 = ((r**(-A))*sympy.exp(r))/math.factorial(n)
    
    #Ableitung Term
    f = (r**(n+A)*(sympy.exp(-r)))
    df = sympy.Symbol.diff(f,r,n)
    
    gL_An = var1*df
    return gL_An.subs(r, func).simplify()
#=================================================================================================

#=================================================================================================
#Radial Function
def R(n,l):
    rho = (2*r)/(n*a)
    const1 = math.sqrt((2/(n))**3)*(a**(-3/2))
    const2 = math.sqrt((math.factorial(n-l-1))/((2*n)*math.factorial(n+l)))   #works
    const3 = sympy.exp(-rho/2)*(rho**l)                                       #works
    
    R_nl = const1*const2*const3*gL(rho, (2*l)+1,n-l-1)
    
    return R_nl
#=================================================================================================

#=================================================================================================





#==============Wave function for H-Atom=================== #Assumption a = 5.2946541*(10**(-11))metre 
 
#Wave functions in normalized positions in spherical coordinates r, theta, phi
def psi(n,l,m):
    psi_nlm = Y(l,m)*R(n,l)
    return psi_nlm

#conjugate from psi
def conj_psi(n,l,m):
    c_psi_nlm = conj_Y(l,m)*R(n,l)
    return c_psi_nlm

#psi*psi
def psi_quad(n,l,m):
    return psi(n,l,m)*conj_psi(n, l, m)


#integrating psi^2 in the real space
def normalising_psi(n,l,m):
    int1 = sympy.integrate((r**2)*psi_quad(n,l,m), (r, 0, np.Infinity)).subs(a,a_val)
    int2 = sympy.integrate(sympy.sin(theta)*int1, (theta, 0, np.pi))
    int3 = sympy.integrate(int2,(phi,0,2*np.pi))
    return int3  
#=================================================================================================

#______________________________________Graphing the Functions__________________________________
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10,10))
psi_quad_n = sympy.lambdify([r,theta,phi], psi_quad(n,l,m).subs(a,a_val))
R_n = sympy.lambdify([r], R(n,l).subs(a,a_val))
Y_n = sympy.lambdify([theta,phi], Y(l,m))
Y2_n = sympy.lambdify([theta,phi], Y2(l,m))

lim_val = 30
R_o = lim_val*a_val

r_val = np.arange(0,R_o,0.001*a_val)
theta_val = np.linspace(0,np.pi,100)


pixels = 1000
x = np.tile(np.linspace(-R_o,R_o,pixels),pixels)
y = np.repeat(np.linspace(-R_o,R_o,pixels),pixels)


theta_val = np.arctan2(y,x)
radius = np.sqrt(x**2+y**2)

#=============Probability density function================
plt.matshow(psi_quad_n(radius,theta_val,0).reshape(pixels,pixels))
plt.colorbar()

plt.show()


'''
The Code below generates a csv file with coordinates in spherical system
based on the probability density function of the wavefunction to input them in Blender.

                                     STILL IN DEVELOPMENT
'''

#========================Random number===================
# rng = np.random.default_rng(12345)

# '''Need N and n as Integers
# and
# sum of 'n' = N  '''
# dnsty_const = 1*a_val

# Koordinates = []
# max_psi_quad = psi_quad_n(r_val,np.pi/2,0).max()


# def Rej_pdf(rad,thet,azi,prob_chance):
#     real_prob = psi_quad_n(rad,thet,azi)/max_psi_quad
#     if real_prob > prob_chance:
#         return True
#     else:
#         return False
    

# for i in np.linspace(0,R_o,100):                         #Radius

#     U_kgl = 4*np.pi*(i**2)                      # Area of the sphere
#     N = U_kgl/((dnsty_const**2)*np.pi)          # N = total balls in the sphere
        
#     for j in np.linspace(0,np.pi,100):                   #Theta (North-South)
#         proportion = np.sin(j)                  # What proportion of N should be in a circle of this sphere
#         n = math.floor(proportion*N)            # n spheres in this circle
#         if n == 0:
#             pass
#         else:
#             del_phi = int(np.ceil(2*np.pi/n))
#             for k in np.linspace(0,2*np.pi,del_phi):        #Phi   (Equitorial)
#                 prob_chance = rng.uniform(0,1)
#                 phase = rng.uniform(0,2*np.pi)
#                 if Rej_pdf(i,j,k,prob_chance):
#                     Koordinates.append([i/a_val,j,k])
#                 else:
#                     pass

# # print(Koordinates)

# np.savetxt('QNfirst.csv', Koordinates, delimiter=',')












