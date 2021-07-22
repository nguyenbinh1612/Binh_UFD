
# import necessary modules

# numpy provides powerful multi-dimensional arrays to hold and manipulate data
import numpy as np
# matplotlib provides powerful functions for plotting figures
import matplotlib.pyplot as plt
# astropy provides unit system and constants for astronomical calculations
import astropy.units as u
import astropy.constants as const


# Code created by G. Besla & E. Patel

class NFW:

    def __init__(self, Mv):
        """ Initialize the class with the current Virial mass of the halo 
        input: virial mass in Msun (mass enclosed within Rvir, which is the radius at which the dark matter
        density is DeltaVir*avg dark matter density of the universe ). """
        
        # get the gravitational constant (the value is 4.498502151575286e-06)
        self.G = const.G.to(u.kpc**3/u.Msun/u.Gyr**2).value
        #print(self.G)
          
        # initialize the virial mass global variable    
        self.Mvir = Mv
   
     ## Cosmology Same as Patel 2020
        #self.h = 0.7 # Hubble constant at z=0 / 100 
        #self.omegaM = 0.27
        self.DelVir = 359  # default z=0 overdensity for this cosmology
        ##cosmology from Jeon 2021
        self.h = 0.71
        self.omegaM = 0.265
    

    
    def c_vir(self):
        # Concentration parameter for halo of a given virial mass
        # taken from Klypin 2011 on Bolshoi simulations, equation 10
        a = self.Mvir*self.h/ 1e12
        return 9.60*(a)**(-0.075)

    
    def delta_vir(self):
        # Overdensity to define virial radius  at z=0 (else OmegaM is a function of z)
        # delta_c taken from Bryan and Norman (1998)

        x = self.omegaM - 1.
        deltac = 18*np.pi**2 + 82*x -39*x**2
        return deltac/self.omegaM

    
    def r_vir(self):
        # virial radius. Where the density of the halo is equal to DeltaVir*AvgDensity of the universe
        # taken from van der Marel 2012, equation A1 
        #("THE M31 VELOCITY VECTOR. II. RADIAL ORBIT TOWARD THE MILKY WAY AND IMPLIED LOCAL GROUP MASS")
        a = 206./self.h
        b = self.delta_vir() * self.omegaM / 97.2
        c = self.Mvir * self.h/1e12
        return a * b**(-1./3.) * c**(1./3.)

    
    def v_vir(self):
        # Circular speed at the virial radius (in km/s)
        rvir = self.r_vir()
        return np.sqrt(self.G*self.Mvir/rvir)
    
    
    def r_s(self, c=False):
        # Scale length for the NFW profile (in kpc)
        if c:
            return self.r_vir()/c
        else: 
            c = self.c_vir()
            return self.r_vir()/c
    
    
    def f(self,x):
        a = np.log(1+x) 
        b = x/(1+x)
        return a - b
    
    
    
    
    def mass(self, r, c=False):
        """NFW mass enclosed as a function of r
        Input: r = Galactocentric distance (kpc)
        c = concentration - Can take concentration as given (cvir) or give it a value
        """
        if c:
            cvir = c
        else:
            cvir = self.c_vir()
        
        x = r/self.r_s(c=cvir)
        print(self.Mvir)
        return self.Mvir*self.f(x)/self.f(cvir)

    
    def rho(self,r, c=False):
        """NFW density profile as a function of r
        Input: r = Galactocentric distance (kpc)
        c = concentration - Can take concentration as given (cvir) or give it a value
        """
        if c:
            cvir = c
        else:
            cvir = self.c_vir()
        x = r/self.r_s(c=cvir)
        rho_s = self.Mvir/(4.*np.pi*self.r_s(c=cvir)**3.*self.f(cvir))
        rho_N = rho_s/(x*(1+x)**2.)
        return rho_N
    
    


    
    def v_max(self,c=False):
        """ Maximal circular speed (km/s);  occurs at rmax = 2.163*(r_s) 
        Input: r = Galactocentric distance (kpc)
        c = concentration - Can take concentration as given (cvir) or give it a value
        """
        if c:
            cvir = c
        else:
            cvir = self.c_vir()
            
        return 0.465*self.v_vir*np.sqrt(cvir/self.f(cvir))
    
  
      
  
