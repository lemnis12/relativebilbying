import lal
import lalsimulation as lalsim
import numpy as np
from time import time
from math import factorial
from numba import jit

@jit(nopython=True)
def func2_21(c, s):
    return 2*c*s**3

@jit(nopython=True)
def func2_22(c, s):
    return s**4

@jit(nopython=True)
def func2_11(c, s):
    return s**2*(3*c**2-s**2)

@jit(nopython=True)
def func2_12(c, s):
    return 2*c*s**3

@jit(nopython=True)
def func201(c, s):
    return np.sqrt(6)*(c**3*s-c*s**3)

@jit(nopython=True)
def func202(c, s):
    return np.sqrt(6)*c**2*s**2

@jit(nopython=True)
def func211(c, s):
    return c**2*(c**2-3*s**2)

@jit(nopython=True)
def func212(c, s):
    return 2.*c**3*s

@jit(nopython=True)
def func221(c, s):
    return -2.*c**3*s

@jit(nopython=True)
def func222(c, s):
    return c**4


@jit(nopython=True)
def func3_31(c,s):
    return 0

@jit(nopython=True)
def func3_32(c,s):
    return np.sqrt(6)*c*s**5

@jit(nopython=True)
def func3_33(c,s):
    return s**6

@jit(nopython=True)
def func3_21(c,s):
    return 0

@jit(nopython=True)
def func3_22(c,s):
    return s**4*(5*c**2-s**2)

@jit(nopython=True)
def func3_23(c,s):
    return np.sqrt(6)*c*s**5

@jit(nopython=True)
def func3_11(c,s):
    return 0

@jit(nopython=True)
def func3_12(c,s):
    return np.sqrt(10)*s**3*(2*c**3-c*s**2)

@jit(nopython=True)
def func3_13(c,s):
    return np.sqrt(15)*c**2*s**4

@jit(nopython=True)
def func301(c,s):
    return 0

@jit(nopython=True)
def func302(c,s):
    return np.sqrt(30)*c**2*s**2*(c**2-s**2)

@jit(nopython=True)
def func303(c,s):
    return 2.*np.sqrt(5)*c**3*s**3

@jit(nopython=True)
def func311(c,s):
    return 0

@jit(nopython=True)
def func312(c,s):
    return np.sqrt(10)*c**3*(c**2*s-2*s**3)

@jit(nopython=True)
def func313(c,s):
    return np.sqrt(15)*c**4*s**2

@jit(nopython=True)
def func321(c,s):
    return 0

@jit(nopython=True)
def func322(c,s):
    return c**4*(c**2-5*s**2)

@jit(nopython=True)
def func323(c,s):
    return np.sqrt(6)*c**6*s

@jit(nopython=True)
def func331(c,s):
    return 0

@jit(nopython=True)
def func332(c,s):
    return -np.sqrt(6)*c**5*s

@jit(nopython=True)
def func333(c,s):
    return c**6

from numba import jit

@jit(nopython=True)
def func4_41(c, s):
    return 0.

@jit(nopython=True)
def func4_42(c, s):
    return 0.

@jit(nopython=True)
def func4_43(c, s):
    return 0.

@jit(nopython=True)
def func4_44(c, s):
    return s**8

@jit(nopython=True)
def func4_31(c, s):
    return 0.

@jit(nopython=True)
def func4_32(c, s):
    return 0.

@jit(nopython=True)
def func4_33(c, s):
    return 0.

@jit(nopython=True)
def func4_34(c, s):
    return 2.*np.sqrt(2)*c*s**7

@jit(nopython=True)
def func4_21(c, s):
    return 0.

@jit(nopython=True)
def func4_22(c, s):
    return 0.

@jit(nopython=True)
def func4_23(c, s):
    return 0.

@jit(nopython=True)
def func4_24(c, s):
    return 2*np.sqrt(7)*c**2*s**6

@jit(nopython=True)
def func4_11(c, s):
    return 0.

@jit(nopython=True)
def func4_12(c, s):
    return 0.

@jit(nopython=True)
def func4_13(c, s):
    return 0.

@jit(nopython=True)
def func4_14(c, s):
    return 2*np.sqrt(14)*c**3*s**5

from numba import jit

@jit(nopython=True)
def func401(c,s):
    return 0

@jit(nopython=True)
def func402(c,s):
    return 0

@jit(nopython=True)
def func403(c,s):
    return 0

@jit(nopython=True)
def func404(c,s):
    return np.sqrt(70)*c**4*s**4

@jit(nopython=True)
def func411(c,s):
    return 0

@jit(nopython=True)
def func412(c,s):
    return 0

@jit(nopython=True)
def func413(c,s):
    return 0

@jit(nopython=True)
def func414(c,s):
    return 2*np.sqrt(14)*c**5*s**3

@jit(nopython=True)
def func421(c,s):
    return 0

@jit(nopython=True)
def func422(c,s):
    return 0

@jit(nopython=True)
def func423(c,s):
    return 0

@jit(nopython=True)
def func424(c,s):
    return 2.*np.sqrt(7)*c**6*s**2

@jit(nopython=True)
def func431(c,s):
    return 0

@jit(nopython=True)
def func432(c,s):
    return 0

@jit(nopython=True)
def func433(c,s):
    return 0

@jit(nopython=True)
def func434(c,s):
    return 2.*np.sqrt(2)*c**7*s

@jit(nopython=True)
def func441(c,s):
    return 0

@jit(nopython=True)
def func442(c,s):
    return 0

@jit(nopython=True)
def func443(c,s):
    return 0

@jit(nopython=True)
def func444(c,s):
    return c**8




'''
(2, -2, 1): lambda c, s: 2*c*s**3,
        (2, -2, 2): lambda c, s: s**4,
        (2, -1, 1): lambda c, s: s**2*(3*c**2-s**2),
        (2, -1, 2): lambda c, s: 2*c*s**3,
        (2, 0, 1): lambda c, s: np.sqrt(6)*(c**3*s-c*s**3),
        (2, 0, 2): lambda c, s: np.sqrt(6)*c**2*s**2,
        (2, 1, 1): lambda c, s: c**2*(c**2-3*s**2),
        (2, 1 ,2): lambda c, s: 2.*c**3*s,
        (2, 2, 1): lambda c, s: -2.*c**3*s,
        (2, 2, 2): lambda c, s: c**4,
        (3, -3, 1): lambda c, s: 0.,
        (3, -3, 2): lambda c, s: np.sqrt(6)*c*s**5,
        (3, -3, 3): lambda c, s: s**6,
        (3, -2, 1): lambda c, s: 0.,
        (3, -2, 2): lambda c, s: s**4*(5*c**2-s**2),
        (3, -2, 3): lambda c, s: np.sqrt(6)*c*s**5,
        (3, -1, 1): lambda c, s: 0.,
        (3, -1, 2): lambda c, s: np.sqrt(10)*s**3*(2*c**3-c*s**2),
        (3, -1, 3): lambda c, s: np.sqrt(15)*c**2*s**4,
        (3, 0, 1): lambda c, s: 0.,
        (3, 0, 2): lambda c, s: np.sqrt(30)*c**2*s**2*(c**2-s**2),
        (3, 0, 3): lambda c, s: 2.*np.sqrt(5)*c**3*s**3,
        (3, 1, 1): lambda c, s: 0.,
        (3, 1, 2): lambda c, s: np.sqrt(10)*c**3*(c**2*s-2*s**3),
        (3, 1, 3): lambda c, s: np.sqrt(15)*c**4*s**2,
        (3, 2, 1): lambda c, s: 0.,
        (3, 2, 2): lambda c, s: c**4*(c**2-5*s**2),
        (3, 2, 3): lambda c, s: np.sqrt(6)*c**6*s,
        (3, 3, 1): lambda c, s: 0.,
        (3, 3, 2): lambda c, s: -np.sqrt(6)*c**5*s,
        (3, 3, 3): lambda c, s: c**6,
        (4, -4, 1): lambda c, s: 0.,
        (4, -4, 2): lambda c, s: 0.,
        (4, -4, 3): lambda c, s: 0.,
        (4, -4, 4): lambda c, s: s**8,

        (4, -3, 1): lambda c, s: 0.,
        (4, -3, 2): lambda c, s: 0.,
        (4, -3, 3): lambda c, s: 0.,
        (4, -3, 4): lambda c, s: 2.*np.sqrt(2)*c*s**7,

        (4, -2, 1): lambda c, s: 0.,
        (4, -2, 2): lambda c, s: 0.,
        (4, -2, 3): lambda c, s: 0.,
        (4, -2, 4): lambda c, s: 2*np.sqrt(7)*c**2*s**6,

        (4, -1, 1): lambda c, s: 0.,
        (4, -1, 2): lambda c, s: 0.,
        (4, -1, 3): lambda c, s: 0.,
        (4, -1, 4): lambda c, s: 2*np.sqrt(14)*c**3*s**5,

        (4, 0, 1): lambda c, s: 0.,
        (4, 0, 2): lambda c, s: 0.,
        (4, 0, 3): lambda c, s: 0.,
        (4, 0, 4): lambda c, s: np.sqrt(70)*c**4*s**4,

        (4, 1, 1): lambda c, s: 0.,
        (4, 1, 2): lambda c, s: 0.,
        (4, 1, 3): lambda c, s: 0.,
        (4, 1, 4): lambda c, s: 2*np.sqrt(14)*c**5*s**3,

        (4, 2, 1): lambda c, s: 0.,
        (4, 2, 2): lambda c, s: 0.,
        (4, 2, 3): lambda c, s: 0.,
        (4, 2, 4): lambda c, s: 2.*np.sqrt(7)*c**6*s**2,

        (4, 3, 1): lambda c, s: 0.,
        (4, 3, 2): lambda c, s: 0.,
        (4, 3, 3): lambda c, s: 0.,
        (4, 3, 4): lambda c, s: 2.*np.sqrt(2)*c**7*s,

        (4, 4, 1): lambda c, s: 0.,
        (4, 4, 2): lambda c, s: 0.,
        (4, 4, 3): lambda c, s: 0.,
        (4, 4, 4): lambda c, s: c**8



        (4, 0, 1): lambda c, s: 0.,
        (4, 0, 2): lambda c, s: 0.,
        (4, 0, 3): lambda c, s: 0.,
        (4, 0, 4): lambda c, s: np.sqrt(70)*c**4*s**4,

        (4, 1, 1): lambda c, s: 0.,
        (4, 1, 2): lambda c, s: 0.,
        (4, 1, 3): lambda c, s: 0.,
        (4, 1, 4): lambda c, s: 2*np.sqrt(14)*c**5*s**3,

        (4, 2, 1): lambda c, s: 0.,
        (4, 2, 2): lambda c, s: 0.,
        (4, 2, 3): lambda c, s: 0.,
        (4, 2, 4): lambda c, s: 2.*np.sqrt(7)*c**6*s**2,

        (4, 3, 1): lambda c, s: 0.,
        (4, 3, 2): lambda c, s: 0.,
        (4, 3, 3): lambda c, s: 0.,
        (4, 3, 4): lambda c, s: 2.*np.sqrt(2)*c**7*s,

        (4, 4, 1): lambda c, s: 0.,
        (4, 4, 2): lambda c, s: 0.,
        (4, 4, 3): lambda c, s: 0.,
        (4, 4, 4): lambda c, s: c**8


'''


d_matrix_elements = {
        (2, -2, 1): func2_21,
        (2, -2, 2): func2_22,
        (2, -1, 1): func2_11,
        (2, -1, 2): func2_12,
        (2, 0, 1): func201,
        (2, 0, 2): func202,
        (2, 1, 1): func211,
        (2, 1 ,2): func212,
        (2, 2, 1): func221,
        (2, 2, 2): func222,

        
        (3, -3, 1): func3_31,
        (3, -3, 2): func3_32,
        (3, -3, 3): func3_33,

        (3, -2, 1): func3_21,
        (3, -2, 2): func3_22,
        (3, -2, 3): func3_23,

        (3, -1, 1): func3_11,
        (3, -1, 2): func3_12,
        (3, -1, 3): func3_13,

        (3, 0, 1): func301,
        (3, 0, 2): func302,
        (3, 0, 3): func303,

        (3, 1, 1): func311,
        (3, 1, 2): func312,
        (3, 1, 3): func313,

        (3, 2, 1): func321,
        (3, 2, 2): func322,
        (3, 2, 3): func323,

        (3, 3, 1): func331,
        (3, 3, 2): func332,
        (3, 3, 3): func333,

        (4, -4, 1): func4_41,
        (4, -4, 2): func4_42,
        (4, -4, 3): func4_43,
        (4, -4, 4): func4_44,

        (4, -3, 1): func4_31,
        (4, -3, 2): func4_32,
        (4, -3, 3): func4_33,
        (4, -3, 4): func4_34,

        (4, -2, 1): func4_21,
        (4, -2, 2): func4_22,
        (4, -2, 3): func4_23,
        (4, -2, 4): func4_24,

        (4, -1, 1): func4_11,
        (4, -1, 2): func4_12,
        (4, -1, 3): func4_13,
        (4, -1, 4): func4_14,

        (4, 0, 1): func401,
        (4, 0, 2): func402,
        (4, 0, 3): func403,
        (4, 0, 4): func404,

        (4, 1, 1): func411,
        (4, 1, 2): func412,
        (4, 1, 3): func413,
        (4, 1, 4): func414,

        (4, 2, 1): func421,
        (4, 2, 2): func422,
        (4, 2, 3): func423,
        (4, 2, 4): func424,

        (4, 3, 1): func431,
        (4, 3, 2): func432,
        (4, 3, 3): func433,
        (4, 3, 4): func434,

        (4, 4, 1): func441,
        (4, 4, 2): func442,
        (4, 4, 3): func443,
        (4, 4, 4): func444

        }


@jit(nopython=True)
def L0_func(mass_1, mass_2, chi_1z, chi_2z, reference_frequency):
    """
    Computes the magnitude of the initial angular momentum L0.
    params: parameter dictionary
    """

   
    M = mass_1 + mass_2
    eta = (mass_1*mass_2)/(M**2)
    
    Omega_o = np.pi*reference_frequency
    x = (M*Omega_o*lal.MTSUN_SI)**(2/3)

    delta = np.sqrt(1 - 4*eta)
    
    L0 = 1
    L1 = 3.0/2. + eta/6.0
    L2 = (81 + (-57 + eta)*eta)/24.
    L3 = (10935. + eta*(-62001 + 1674*eta + 7*eta**2 + 2214*np.pi**2))/1296.
    L32 = (-7*(chi_1z + chi_2z + chi_1z*delta - chi_2z*delta) + 5*(chi_1z + chi_2z)*eta)/6.
    L52 = (-1650*(chi_1z + chi_2z + chi_1z*delta - chi_2z*delta) + 1336*(chi_1z + chi_2z)*eta 
           + 511*(chi_1z - chi_2z)*delta*eta + 28*(chi_1z + chi_2z)*eta**2)/600. 

    PNexp = L0 + L1*x + L2*x**2 + L3*x**3 + L32*x**(3/2) + L52*x**(5/2)

    return eta*x**(-1/2)*PNexp

@jit(nopython=True)
def thetaJL_func(mass_1, mass_2, chi_1x, chi_1y, chi_1z, chi_2x, chi_2y, chi_2z, reference_frequency):
    """
    Computes thetaJL, the angle between the total angular momentum J and the initial angular momentum L0.
    params: parameter dictionary
    """
    
   
    M = mass_1+mass_2
   

    # Components in L0 frame
    Jx = (mass_1/M)**2*chi_1x + (mass_2/M)**2*chi_2x
    Jy = (mass_1/M)**2*chi_1y + (mass_2/M)**2*chi_2y
    Jz = (mass_1/M)**2*chi_1z + (mass_2/M)**2*chi_2z + L0_func(mass_1, mass_2, chi_1z, chi_2z, reference_frequency)
    J = np.array([Jx, Jy, Jz])

    return np.arccos(Jz/np.linalg.norm(J))

@jit(nopython=True)
def phiJL_func(mass_1, mass_2, chi_1x, chi_1y, chi_1z, chi_2x, chi_2y, chi_2z, reference_frequency):
    """
    Computes phiJL, the azimuthal angle of the total angular momentum J in the L0 frame.
    params: parameter dictionary
    """
    
    
    M = mass_1+mass_2
    

    # Components in L0 frame
    Jx = (mass_1/M)**2*chi_1x + (mass_2/M)**2*chi_2x
    Jy = (mass_1/M)**2*chi_1y+ (mass_2/M)**2*chi_2y
    Jz = (mass_1/M)**2*chi_1z + (mass_2/M)**2*chi_2z + L0_func(mass_1, mass_2, chi_1z, chi_2z, reference_frequency)
    J = np.array([Jx, Jy, Jz])

    return np.arctan2(Jy, Jx)
@jit(nopython=True)
def rotate_z(V, theta):
    """
    Rotates the vector V by an angle theta around the z-axis.
    V: 1-D array with 3 components
    theta: angle
    """
    Vx = V[0]
    Vy = V[1]
    Vz = V[2]
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([Vx*c - Vy*s, Vx*s + Vy*c, Vz])

@jit(nopython=True)
def rotate_y(V, theta):
    """
    Rotates the vector V by an angle theta around the y-axis.
    V: 1-D array with 3 components
    theta: angle
    """
    Vx = V[0]
    Vy = V[1]
    Vz = V[2]
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([Vx*c + Vz*s, Vy, -Vx*s + Vz*c])
    
@jit(nopython=True)
def zeta_polarization(mass_1, mass_2, chi_1x, chi_1y, chi_1z, chi_2x, chi_2y, chi_2z, theta_jn, kappa, reference_frequency):
    """
    Computes zeta, the polarization angle between the N frame where the x axis is aligned with L0's projection into the x-y plane and the spherical harmonic N frame where the y axis is aligned with J x N. This angle is needed to rotate the mode decomposition that we computed in the J frame into the frame aligned with L0 that convention dictates.
    params: parameter dictionary
    """
   
    
    M = mass_1 + mass_2
   

    thetaJL = thetaJL_func(mass_1, mass_2, chi_1x, chi_1y, chi_1z, chi_2x, chi_2y, chi_2z, reference_frequency)
    
    # Components in J frame
    xp_x = np.cos(theta_jn)
    xp_y = 0.
    xp_z = -np.sin(theta_jn)
    xp = np.array([xp_x, xp_y, xp_z])
    
    yp_x = 0.
    yp_y = 1.
    yp_z = 0.
    yp = np.array([yp_x, yp_y, yp_z])
    
    N_x = np.sin(theta_jn)
    N_y = 0
    N_z = np.cos(theta_jn)
    N = np.array([N_x, N_y, N_z])
    
    L0_L0 = np.array([0, 0, 1]) # direction of L0 in L0 frame
    L0 = rotate_z(rotate_y(L0_L0, -thetaJL), -kappa)
    
    x = L0 - np.dot(L0, N)*N
    
    return np.arctan2(np.dot(x, yp), np.dot(x, xp))


def m2Ylm(l, m, theta_jn):
    """
    Computes the -2 spin weighted spherical harmonic evaluated at theta = theta_jn, phi = 0.
    l: l index
    m: m index
    theta_jn: theta_jn angle
    """
    m = int(m)
    l = int(l)
    return lal.SpinWeightedSphericalHarmonic(theta_jn, 0, -2, l, m) # kappa + np.pi



def wignerd_dict(l, m, mprime, beta):
    '''
    if mprime < 0:
        c = np.cos((np.pi - beta)/2)
        s = np.sin((np.pi - beta)/2)

        npf = (-1)**(l+m)
        mprime *= -1
    else:
    '''
    npf = (-1)**(l+m)    
    c = np.cos(beta/2)
    s = np.sin(beta/2)
    pf = 1
    
    f = d_matrix_elements[(l, m, mprime)]

    return f(c, s), f(s, c)


def Atransfer_slow(l, m, mprime, alpha, beta, theta_jn):
    """
    Computes the mode-by-mode transfer function A defined in equation 3.7 of https://arxiv.org/pdf/2004.06503.pdf.
    l: l index
    m: m index
    mprime: mprime index
    alpha: Euler angle alpha
    beta: Euler angle beta
    theta_jn: angle theta_jn
    """

    #t = time()
    wd_coefficients = wignerd_dict(l, m, mprime, beta)
    nwd_coefficients = wignerd_dict(l, -m, mprime, beta)
    npf = (-1)**(l+m)
    out0 = np.exp(-1j*m*alpha)*wd_coefficients[0]*m2Ylm(l, m, theta_jn)      # m pos, mprime pos
    out1 = npf*np.exp(-1j*m*alpha)*wd_coefficients[1]*m2Ylm(l, m, theta_jn) # m pos, mprime neg


    out2 = np.exp(1j*m*alpha)*nwd_coefficients[0]*m2Ylm(l, -m, theta_jn) # m neg, mprime pos
    out3 = npf*np.exp(1j*m*alpha)*nwd_coefficients[1]*m2Ylm(l, -m, theta_jn) # m neg, mprime neg
    #print('time taken for wignerd slow and m2Ylm', time()-t, '\n')

    return out0, out1, out2, out3


def twist_factor_slow_plus_cross(l, mprime, alpha, beta, gamma, theta_jn):
    """
    Computes the coefficients that twist the L frame modes into the plus and cross polarizations without the 1/2 or 1/2i. Seen in equations 3.5, 3.6 of https://arxiv.org/pdf/2004.06503.pdf.
    l: l index
    m: m index
    mprime: mprime index
    alpha: Euler angle alpha
    beta: Euler angle beta
    gamma: Euler angle gamma
    theta_jn: angle theta_jn
    pol: string, indicates polarization, accepted values are '+' and 'x'.
    """
    #t = time()

    m_arr = np.arange(1, 1+l, 1)
    plus_summand = np.zeros(len(alpha), dtype = complex)
    cross_summand = np.zeros(len(alpha), dtype = complex)

    for m in m_arr:
        at_slow = Atransfer_slow(l, m, mprime, alpha, beta, theta_jn)
        term_1 = at_slow[1]+at_slow[3]#Atransfer_slow(l, m, -mprime, alpha, beta, theta_jn)
        term_2 = (-1)**l*np.conj(at_slow[0]+at_slow[2])#(-1)**l*np.conj(Atransfer_slow(l, m, mprime, alpha, beta, theta_jn))
        plus_summand = plus_summand+term_1+term_2
        cross_summand = cross_summand+term_1-term_2

    wd_0 = wignerd_dict(l, 0, mprime, beta)
    term_1 = (-1)**l*wd_0[1]*m2Ylm(l, 0, theta_jn)
    term_2 = (-1)**l*wd_0[0]*m2Ylm(l, 0, theta_jn)

    plus_summand += term_1+term_2
    cross_summand += term_1-term_2

    return np.exp(1j*mprime*gamma)*plus_summand, np.exp(1j*mprime*gamma)*cross_summand



def h_Lframe(params, f_seq, reference_frequency, l, m):
    """
    Computes the (l, m) L frame mode evaluated for the parameters in the dict params at the frequencies in f_seq.
    params: parameter dictionary
    f_seq: REAL8Sequence LAL object that contains a frequency array
    l: l index
    m: m index
    """
    m = int(m)
    l = int(l)
    lal_dict_L = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPhenomXPHMPrecModes(lal_dict_L,1); # 1 chooses L frame instead of J frame
    try:
        #h_posf_L, _ = lalsim.SimIMRPhenomXPHMOneModeFrequencySequence(f_seq, l, m,
        h_posf_L, _ = lalsim.SimIMRPhenomXPHMFrequencySequenceOneMode(f_seq, l, m,
                                        params['mass_1'] * lal.MSUN_SI,
                                        params['mass_2'] * lal.MSUN_SI,
                                        params['chi_1x'],
                                        params['chi_1y'],
                                        params['chi_1z'],
                                        params['chi_2x'],
                                        params['chi_2y'],
                                        params['chi_2z'],
                                        params['luminosity_distance']*3.08567758128e22,
                                        params['inclination'],
                                        params['phase'], #phiref doesn't matter
                                        reference_frequency, 
                                        lal_dict_L
                                       )
        return h_posf_L.data.data
    except RuntimeError:
        print('Could not generate wf, setting likelihood to -inf')
        return None

def euler_angles(params, f_seq, waveform_arugments, mprime):
    """
    Computes the (l, m) euler angles alpha, beta, and gamma evaluated for the parameters in the dict params at the frequencies in f_seq.
    params: parameter dictionary
    f_seq: REAL8Sequence LAL object that contains a frequency array
    mprime: mprime index
    backupNNLO: boolean, if True: uses NNLO (next to next to leading order) angles as a backup to MSA (multiple scale analysis) angles if the MSA prescription fails
                         if False: raises an error if the MSA prescription fails
    """
    mprime = int(mprime)
    fs = f_seq.data
    phiz_of_f               = lal.CreateREAL8Sequence(len(fs))
    zeta_of_f               = lal.CreateREAL8Sequence(len(fs))
    costhetaL_of_f          = lal.CreateREAL8Sequence(len(fs))
    phiz_of_f.data          = np.zeros(len(fs))
    zeta_of_f.data          = np.zeros(len(fs))
    costhetaL_of_f.data     = np.zeros(len(fs))
    lalDict_MSA = lal.CreateDict();
    hard_code_fmax = 0. ## this number is irrlevant so hard coding it
        # change flag to return an error if MSA fails
    lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(lalDict_MSA, 223)
    #print('get info')
    phiz_of_f, zeta_of_f, costhetaL_of_f, = lalsim.SimIMRPhenomXPMSAAngles(f_seq,
                               params['mass_1'] * lal.MSUN_SI,
                               params['mass_2'] * lal.MSUN_SI,
                               params['chi_1x'], params['chi_1y'], params['chi_1z'],
                               params['chi_2x'], params['chi_2y'], params['chi_2z'],
                               params['inclination'],
                               waveform_arugments['reference_frequency'], mprime,
                               lalDict_MSA);
    #waveform_arugments['minimum_frequency'], hard_code_fmax
    alpha   = phiz_of_f.data + np.pi - params['kappa']
    beta    = np.unwrap(np.arccos(costhetaL_of_f.data))
    epsilon = zeta_of_f.data # gamma = -epsilon
    #print('beta', beta)
    return alpha, beta, -epsilon



def compute_C_prefactors(params, f_seq, waveform_arugments, modelist, fs):
    """
    Computes all prefactors of the time-dependent L frame modes of the strain in the detectors in detstrings for the parameters in the dict params at the frequencies in f_seq.
    params: parameter dictionary
    f_seq: REAL8Sequence LAL object that contains a frequency array
    modelist: array of modes to use (l, m)
    backupNNLO: boolean, if True: uses NNLO (next to next to leading order) angles as a backup to MSA (multiple scale analysis) angles if the MSA prescription fails
                         if False: raises an error if the MSA prescription fails
    detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
    """
    theta_jn = params['theta_jn']
    n_freq = len(f_seq.data)
    ea1 = euler_angles(params, f_seq, waveform_arugments, 1)
    ea2 = euler_angles(params, f_seq, waveform_arugments, 2)
    ea3 = euler_angles(params, f_seq, waveform_arugments, 3)
    ea4 = euler_angles(params, f_seq, waveform_arugments, 4)

    eulerangles = np.array([ea1, ea2, ea2, ea3, ea4])#np.array([euler_angles(params, f_seq, waveform_arugments, mode[1]) for mode in modelist])
    #print('time taken to get euler angles', time()-t)
    #t = time()
    nmodes = len(modelist)
    Cplus_J = np.zeros([nmodes, len(f_seq)], dtype = complex)
    Ccross_J = np.zeros([nmodes, len(f_seq)], dtype = complex)
    hL = np.zeros([nmodes, len(fs)], dtype = complex)
    for i in range(nmodes):
        t = time()
        plus, cross = twist_factor_slow_plus_cross(modelist[i,0], modelist[i,1], eulerangles[i, 0], eulerangles[i, 1], eulerangles[i, 2], theta_jn)
        Cplus_J[i, :] = .5*plus
        Ccross_J[i, :] = 1j*.5*cross
        #print(time()-t, 'time taken to get prefactors')
        t = time()
        hL[i, :] = h_Lframe(params, fs, waveform_arugments['reference_frequency'], modelist[i,0], -modelist[i,1])
        #print(time()-t, 'time taken for l frame stuff')

    #print('\n\n')


    '''
    Cplus_J = np.array([1/2*twist_factor_slow(modelist[i,0], modelist[i,1], eulerangles[i, 0], eulerangles[i, 1], eulerangles[i, 2], theta_jn, '+') for i in range(modelist.shape[0])])
    print(time()-t, 'time taken from plus')
    t = time()
    Ccross_J = np.array([1j/2*twist_factor_slow(modelist[i,0], modelist[i,1], eulerangles[i, 0], eulerangles[i, 1], eulerangles[i, 2], theta_jn, 'x') for i in range(modelist.shape[0])])
    print(time()-t, 'time taken from cross\n\n')
    #print('time for twist factors', time()-t)
    '''
    zeta = zeta_polarization(params['mass_1'], params['mass_2'], params['chi_1x'], params['chi_1y'], params['chi_1z'], 
        params['chi_2x'], params['chi_2y'], params['chi_2z'], params['theta_jn'], 
        params['kappa'], waveform_arugments['reference_frequency'])

    Cplus = np.cos(2*zeta)*Cplus_J + np.sin(2*zeta)*Ccross_J
    Ccross = np.cos(2*zeta)*Ccross_J - np.sin(2*zeta)*Cplus_J
    
    return Cplus, Ccross, hL
    
    
def compute_td_L_frame_modes(params, f_seq, waveform_arugments, modelist):
    """
    Computes all time-dependent L frame modes of the strain in the detectors in detstrings for the parameters in the dict params at the frequencies in f_seq.
    params: parameter dictionary
    f_seq: REAL8Sequence LAL object that contains a frequency array
    modelist: array of modes to use (l, m)    
    detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
    """

    hL = np.array([h_Lframe(params, f_seq, waveform_arugments['reference_frequency'], mode[0], -mode[1]) for mode in modelist])

    return hL


@jit(nopython=True)
def theta_jn_func_op(mass_1, mass_2, chi_1x, chi_1y, chi_1z, chi_2x, chi_2y, chi_2z, iota, phase, reference_frequency):
    """
    Computes theta_jn, the angle between the total angular momentum J and the line of sight normal vector N.
    params: parameter dictionary that has the old angle parameters iota/inclination and phiref
    """
    
    

    # Components in L0 frame
    '''
    Jx = (m1/M)**2*s1x + (m2/M)**2*s2x
    Jy = (m1/M)**2*s1y + (m2/M)**2*s2y
    Jz = (m1/M)**2*s1z + (m2/M)**2*s2z + L0_func(m1, m2, chi_1z, chi_2z, reference_frequency)
    J = np.array([Jx, Jy, Jz])
    '''
    # Components in L0 frame
    M = mass_1+mass_2
    Jx = (mass_1/M)**2*chi_1x + (mass_2/M)**2*chi_2x
    Jy = (mass_1/M)**2*chi_1y+ (mass_2/M)**2*chi_2y
    Jz = (mass_1/M)**2*chi_1z + (mass_2/M)**2*chi_2z + L0_func(mass_1, mass_2, chi_1z, chi_2z, reference_frequency)
    J = np.array([Jx, Jy, Jz])

    Nx = np.sin(iota)*np.cos((np.pi/2)-phase)
    Ny = np.sin(iota)*np.sin((np.pi/2)-phase)
    Nz = np.cos(iota)
    N = np.array([Nx, Ny, Nz])

    return np.arccos(np.dot(J, N)/np.linalg.norm(J))


@jit(nopython=True)
def kappa_func_op(mass_1, mass_2, chi_1x, chi_1y, chi_1z, chi_2x, chi_2y, chi_2z, iota, phase, reference_frequency):
    """
    Computes kappa, the final Euler angle between the L0 and J frames.
    params: parameter dictionary that has the old angle parameters iota/inclination and phiref
    """
    

   

    # Components in L0 frame
    Nx = np.sin(iota)*np.cos((np.pi/2)-phase)
    Ny = np.sin(iota)*np.sin((np.pi/2)-phase)
    Nz = np.cos(iota)
    N = np.array([Nx, Ny, Nz])
    
    # Rotate to J prime frame
    '''
    mass_1 = params['mass_1']
    mass_2 = params['mass_2']
    chi_1x = params['chi_1x']
    chi_1y = params['chi_1y']
    chi_1z = params['chi_1z']
    chi_2x = params['chi_2x']
    chi_2y = params['chi_2y']
    chi_2z = params['chi_2z']
    reference_frequency = params['reference_frequency']
    '''

    phiJL = phiJL_func(mass_1, mass_2, chi_1x, chi_1y, chi_1z, chi_2x, chi_2y, chi_2z, reference_frequency)
    thetaJL = thetaJL_func(mass_1, mass_2, chi_1x, chi_1y, chi_1z, chi_2x, chi_2y, chi_2z, reference_frequency)

    N_Jp = rotate_y(rotate_z(N, -phiJL), -thetaJL)
    Nx_Jp = N_Jp[0]
    Ny_Jp = N_Jp[1]
    
    return np.arctan2(Ny_Jp, Nx_Jp)

'''

def twist_factor_fast_plus_cross(l, mprime, alpha, beta, gamma, theta_jn):
    """
    Computes the coefficients that twist the L frame modes into the plus and cross polarizations without the 1/2 or 1/2i. Seen in equations 3.5, 3.6 of https://arxiv.org/pdf/2004.06503.pdf.
    l: l index
    m: m index
    mprime: mprime index
    alpha: Euler angle alpha
    beta: Euler angle beta
    gamma: Euler angle gamma
    theta_jn: angle theta_jn
    pol: string, indicates polarization, accepted values are '+' and 'x'.
    """
    m_arr = np.arange(2*l+1)-l
    term_1 = Atransfer_slow(l, m_arr, -mprime, alpha, beta, theta_jn)
    term_2 = (-1)**l * np.conj(Atransfer_slow(l, m_arr, mprime, alpha, beta, theta_jn))
    plus_summand = np.sum(term_1 + term_2, axis=0)
    cross_summand = np.sum(term_1 - term_2, axis=0)

    return np.exp(1j*mprime*gamma) * plus_summand, np.exp(1j*mprime*gamma) * cross_summand


def iota_func_np(params):
    """
    Computes iota, the angle between initial angular momentum L0 and the line of sight normal vector N.
    params: parameter dictionary that has the new angle parameters theta_jn and kappa
    """
    if params['mass_1'] < params['mass_2']:
        temp = params['mass_1']
        params['mass_1'] = params['mass_2']
        params['mass_2'] = temp
    theta_jn = params['theta_jn']
    kappa = params['kappa']

    # Components in J frame
    Nx = np.sin(theta_jn)
    # Ny = 0
    Nz = np.cos(theta_jn)
    N = np.array([Nx, 0, Nz])

    # Rotate to L0 frame
    phiJL = phiJL_func(params)
    thetaJL = thetaJL_func(params)
    NL0 = rotate_z(rotate_y(rotate_z(N, kappa), thetaJL), phiJL)

    return np.arccos(NL0[2])

def phiref_func_np(params):
    """
    Computes phiref, the azimuthal angle of N in the L0 frame.
    params: parameter dictionary that has the new angle parameters theta_jn and kappa
    """
    if params['mass_1'] < params['mass_2']:
        temp = params['mass_1']
        params['mass_1'] = params['mass_2']
        params['mass_2'] = temp
    theta_jn = params['theta_jn']
    kappa = params['kappa']
    
    # Components in J frame
    Nx = np.sin(theta_jn)
    # Ny = 0
    Nz = np.cos(theta_jn)
    N = np.array([Nx, 0, Nz])
    
    # Rotate to L0 frame
    phiJL = phiJL_func(params)
    thetaJL = thetaJL_func(params)
    NL0 = rotate_z(rotate_y(rotate_z(N, kappa), thetaJL), phiJL)
    
    return np.pi/2 - np.arctan2(NL0[1], NL0[0])'''

'''def add_np(params):
    """
    Computes and adds the new parameters theta_jn and kappa to the parameter dictionary (given a valid parameter dictionary that has inclination (iota) and phiref)
    params: parameter dictionary that has the old angle parameters iota/inclination and phiref
    """
    if params['mass_1'] < params['mass_2']:
        temp = params['mass_1']
        params['mass_1'] = params['mass_2']
        params['mass_2'] = temp
    m1 = params['mass_1']
    m2 = params['mass_2']
    M = m1 + m2
    s1x = params['chi_1x']
    s1y = params['chi_1y']
    s1z = params['chi_1z']
    s2x = params['chi_2x']
    s2y = params['chi_2y']
    s2z = params['chi_2z']
    iota = params['inclination']
    phiref = params['phi_ref']

    # Components in L0 frame
    Jx = (m1/M)**2*s1x + (m2/M)**2*s2x
    Jy = (m1/M)**2*s1y + (m2/M)**2*s2y
    Jz = (m1/M)**2*s1z + (m2/M)**2*s2z + L0_func(params)
    J = np.array([Jx, Jy, Jz])
    

    Nx = np.sin(iota)*np.cos((np.pi/2)-phiref)
    Ny = np.sin(iota)*np.sin((np.pi/2)-phiref)
    Nz = np.cos(iota)
    N = np.array([Nx, Ny, Nz])
    
    params['theta_jn'] = np.arccos(np.dot(J, N)/np.linalg.norm(J))
    
    # Rotate N to J prime frame
    phiJL = phiJL_func(params)
    thetaJL = thetaJL_func(params)
    N_Jp = rotate_y(rotate_z(N, -phiJL), -thetaJL)
    Nx_Jp = N_Jp[0]
    Ny_Jp = N_Jp[1]
    
    params['kappa'] = np.arctan2(Ny_Jp, Nx_Jp)
    
    return params'''

'''def add_op(params):
    """
    Computes and adds the old parameters inclination (iota) and phiref to the parameter dictionary (given a valid parameter dictionary that has the theta_jn and kappa).
    params: parameter dictionary that has the new angle parameters theta_jn and kappa
    """
    if params['mass_1'] < params['mass_2']:
        temp = params['mass_1']
        params['mass_1'] = params['mass_2']
        params['mass_2'] = temp
    theta_jn = params['theta_jn']
    kappa = params['kappa']
    
    # Components in J frame
    Nx = np.sin(theta_jn)
    # Ny = 0
    Nz = np.cos(theta_jn)
    N = np.array([Nx, 0, Nz])
    
    # Rotate to L0 frame
    phiJL = phiJL_func(params)
    thetaJL = thetaJL_func(params)
    NL0 = rotate_z(rotate_y(rotate_z(N, kappa), thetaJL), phiJL)
    
    params['inclination'] = np.arccos(NL0[2])
    params['phi_ref'] = np.pi/2 - np.arctan2(NL0[1], NL0[0])
    
    return params'''

'''
def twist_factor(l, mprime, eimalpha, eibetaover2, eimprimegamma, theta_jn, pol):
    """
    Computes the coefficients that twist the L frame modes into the plus and cross polarizations without the 1/2 or 1/2i. Seen in equations 3.5, 3.6 of https://arxiv.org/pdf/2004.06503.pdf.
    l: l index
    m: m index
    mprime: mprime index
    eimalpha: exp(i*m*alpha) array, index 0 is m, ranges from 0 to max_l
    eibetaover2: exp(i*beta/2)
    eimprimegamma: exp(i*mprime*gamma)
    theta_jn: angle theta_jn
    pol: string, indicates polarization, accepted values are '+' and 'x'.
    """
    m_arr = np.arange(2*l+1)-l
    if pol == '+':
        summand = [Atransfer(l, m, -mprime, eimalpha[abs(m)], eibetaover2, theta_jn) + (-1)**l*np.conj(Atransfer(l, m, mprime, eimalpha[abs(m)], eibetaover2, theta_jn)) for m in m_arr]
    elif pol == 'x':
        summand = [Atransfer(l, m, -mprime, eimalpha[abs(m)], eibetaover2, theta_jn) - (-1)**l*np.conj(Atransfer(l, m, mprime, eimalpha[abs(m)], eibetaover2, theta_jn)) for m in m_arr]
    else:
        raise ValueError('Only + or x polarizations allowed.')
    return eimprimegamma*np.sum(summand, axis=0)

def wignerd_slow(l, m, mprime, beta):
    """
    Computes the wigner d matrix element with indices l, m, mprime, evaluated at angle beta. Only works for cases enumerated.
    l: l index
    m: m index
    mprime: mprime index
    beta: Euler angle beta
    """
    #t = time()
    if mprime < 0:
        c = np.cos((np.pi - beta)/2)
        s = np.sin((np.pi - beta)/2)
        pf = (-1)**(l+m)
        mprime *= -1
    else:
        c = np.cos(beta/2)
        s = np.sin(beta/2)
        pf = 1
    if l == 2:
        if mprime == 2:
            if m == 2:
                f = c**4
            elif m == 1:
                f = 2*c**3*s
            elif m == 0:
                f = np.sqrt(6)*c**2*s**2
            elif m == -1:
                f = 2*c*s**3
            elif m == -2:
                f = s**4
            else:
                raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
        elif mprime == 1:
            if m == 2:
                f = -2*c**3*s
            elif m == 1:
                f = c**2*(c**2-3*s**2)
            elif m == 0:
                f = np.sqrt(6)*(c**3*s-c*s**3)
            elif m == -1:
                f = s**2*(3*c**2-s**2)
            elif m == -2:
                f = 2*c*s**3
            else:
                raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
        else:
            raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
    elif l == 3:
        if mprime == 3:
            if m == 3:
                f = c**6
            elif m == 2:
                f = np.sqrt(6)*c**6*s
            elif m == 1:
                f = np.sqrt(15)*c**4*s**2
            elif m == 0:
                f = 2*np.sqrt(5)*c**3*s**3
            elif m == -1:
                f = np.sqrt(15)*c**2*s**4
            elif m == -2:
                f = np.sqrt(6)*c*s**5
            elif m == -3:
                f = s**6
            else:
                raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
        elif mprime == 2:
            if m == 3:
                f = -np.sqrt(6)*c**5*s
            elif m == 2:
                f = c**4*(c**2-5*s**2)
            elif m == 1:
                f = np.sqrt(10)*c**3*(c**2*s-2*s**3)
            elif m == 0:
                f = np.sqrt(30)*c**2*s**2*(c**2-s**2)
            elif m == -1:
                f = np.sqrt(10)*s**3*(2*c**3-c*s**2)
            elif m == -2:
                f = s**4*(5*c**2-s**2)
            elif m == -3:
                f = np.sqrt(6)*c*s**5
            else:
                raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
        elif mprime == 1:
            f = 0
        else:
            raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
    elif l == 4:
        if mprime == 4:
            if m == 4:
                f = c**8
            elif m == 3:
                f = 2*np.sqrt(2)*c**7*s
            elif m == 2:
                f = 2*np.sqrt(7)*c**6*s**2
            elif m == 1:
                f = 2*np.sqrt(14)*c**5*s**3
            elif m == 0:
                f = np.sqrt(70)*c**4*s**4
            elif m == -1:
                f = 2*np.sqrt(14)*c**3*s**5
            elif m == -2:
                f = 2*np.sqrt(7)*c**2*s**6
            elif m == -3:
                f = 2*np.sqrt(2)*c*s**7
            elif m == -4:
                f = s**8
            else:
                raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
        elif mprime == 3:
            f = 0
        elif mprime == 2:
            f = 0
        elif mprime == 1:
            f = 0
        else:
            raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
    else:
        raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)

    #print('time taken by wignerd_slow', time()-t)
    return f*pf

def wignerd(l, m, mprime, eibetaover2):
    """
    Computes the wigner d matrix element with indices l, m, mprime, evaluated at angle beta. Only works for cases enumerated.
    l: l index
    m: m index
    mprime: mprime index
    eibetaover2: exp(i*beta/2)
    """
    if mprime < 0:
        c = np.imag(eibetaover2)
        s = np.real(eibetaover2)
        pf = (-1)**(l+m)
        mprime *= -1
    else:
        c = np.real(eibetaover2)
        s = np.imag(eibetaover2)
        pf = 1
    if l == 2:
        if mprime == 2:
            if m == 2:
                f = c**4
            elif m == 1:
                f = 2*c**3*s
            elif m == 0:
                f = np.sqrt(6)*c**2*s**2
            elif m == -1:
                f = 2*c*s**3
            elif m == -2:
                f = s**4
            else:
                raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
        elif mprime == 1:
            if m == 2:
                f = -2*c**3*s
            elif m == 1:
                f = c**2*(c**2-3*s**2)
            elif m == 0:
                f = np.sqrt(6)*(c**3*s-c*s**3)
            elif m == -1:
                f = s**2*(3*c**2-s**2)
            elif m == -2:
                f = 2*c*s**3
            else:
                raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
        else:
            raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
    elif l == 3:
        if mprime == 3:
            if m == 3:
                f = c**6
            elif m == 2:
                f = np.sqrt(6)*c**6*s
            elif m == 1:
                f = np.sqrt(15)*c**4*s**2
            elif m == 0:
                f = 2*np.sqrt(5)*c**3*s**3
            elif m == -1:
                f = np.sqrt(15)*c**2*s**4
            elif m == -2:
                f = np.sqrt(6)*c*s**5
            elif m == -3:
                f = s**6
            else:
                raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
        elif mprime == 2:
            if m == 3:
                f = -np.sqrt(6)*c**5*s
            elif m == 2:
                f = c**4*(c**2-5*s**2)
            elif m == 1:
                f = np.sqrt(10)*c**3*(c**2*s-2*s**3)
            elif m == 0:
                f = np.sqrt(30)*c**2*s**2*(c**2-s**2)
            elif m == -1:
                f = np.sqrt(10)*s**3*(2*c**3-c*s**2)
            elif m == -2:
                f = s**4*(5*c**2-s**2)
            elif m == -3:
                f = np.sqrt(6)*c*s**5
            else:
                raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
        elif mprime == 1:
            f = 0
        else:
            raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
    elif l == 4:
        if mprime == 4:
            if m == 4:
                f = c**8
            elif m == 3:
                f = 2*np.sqrt(2)*c**7*s
            elif m == 2:
                f = 2*np.sqrt(7)*c**6*s**2
            elif m == 1:
                f = 2*np.sqrt(14)*c**5*s**3
            elif m == 0:
                f = np.sqrt(70)*c**4*s**4
            elif m == -1:
                f = 2*np.sqrt(14)*c**3*s**5
            elif m == -2:
                f = 2*np.sqrt(7)*c**2*s**6
            elif m == -3:
                f = 2*np.sqrt(2)*c*s**7
            elif m == -4:
                f = s**8
            else:
                raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
        elif mprime == 3:
            f = 0
        elif mprime == 2:
            f = 0
        elif mprime == 1:
            f = 0
        else:
            raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
    else:
        raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
    return f*pf

def create_eimalpha(alpha, max_l):
    eialpha = np.exp(1j*alpha)
    if max_l == 2:
        return np.array([0*alpha+1, eialpha, eialpha*eialpha])
    elif max_l == 3:
        eialpha2 = eialpha*eialpha
        return np.array([0*alpha+1, eialpha, eialpha2, eialpha2*eialpha])
    elif max_l == 4:
        eialpha2 = eialpha*eialpha
        return np.array([0*alpha+1, eialpha, eialpha2, eialpha2*eialpha, eialpha2*eialpha2])
    else:
        raise ValueError('This code currently only supports modes up to mprime=4, l=4 and requires the quadrupole mode.')

def eintheta(theta, n):
    eitheta = np.exp(1j*theta)
    if n == 0:
        return 0*eitheta + 1
    elif n == 1:
        return eitheta
    elif n == 2:
        return eitheta*eitheta
    elif n == 3:
        return eitheta*eitheta*eitheta
    elif n == 4:
        eitheta2 = eitheta*eitheta
        return eitheta2*eitheta2
    
def create_eimprimegamma(gamma, mprime_arr):
    return np.array([eintheta(gamma[i], mprime_arr[i]) for i in range(len(mprime_arr))])

def Atransfer(l, m, mprime, eiabsmalpha, eibetaover2, theta_jn):
    """
    Computes the mode-by-mode transfer function A defined in equation 3.7 of https://arxiv.org/pdf/2004.06503.pdf.
    l: l index
    m: m index
    mprime: mprime index
    eiabsmalpha: exp(i*|m|*alpha)
    eibetaover2: exp(i*beta/2)
    theta_jn: angle theta_jn

    """



    if m > 0:
        return np.conj(eiabsmalpha)*wignerd(l, m, mprime, eibetaover2)*m2Ylm(l, m, theta_jn)
    else:
        return eiabsmalpha*wignerd(l, m, mprime, eibetaover2)*m2Ylm(l, m, theta_jn)

def Atransfer(l, m, mprime, eiabsmalpha, eibetaover2, theta_jn):
    """
    Computes the mode-by-mode transfer function A defined in equation 3.7 of https://arxiv.org/pdf/2004.06503.pdf.
    l: l index
    m: m index
    mprime: mprime index
    eiabsmalpha: exp(i*|m|*alpha)
    eibetaover2: exp(i*beta/2)
    theta_jn: angle theta_jn

    """



    if m > 0:
        return np.conj(eiabsmalpha)*wignerd(l, m, mprime, eibetaover2)*m2Ylm(l, m, theta_jn)
    else:
        return eiabsmalpha*wignerd(l, m, mprime, eibetaover2)*m2Ylm(l, m, theta_jn)
'''


