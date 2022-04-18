#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import modules
import numpy as np

# import plotting modules
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.display import Latex

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


class VelocityStuff:
    
    
    def __init__(self):
        
        """Initiate the class."""
        
        self.data = []

        
    def find_star_COM(self, radius_limit, star_mass_array, x, y, z):
        
        """This function finds the positions in x, y and z of the center of mass of the stars.
        
        Inputs:
        1) radius_limit: the radius that encompasses all the stellar mass we want to consider (in pc)
        2) star_mass_array: an array from raw data that has all the stars in the system & their masses (in Msun)
        3) x: the x component of all the stars' positions (in pc)
        4) y: the y component of all the stars' positions (in pc)
        5) z: the z component of all the stars' positions (in pc)
        
        Outputs: values for the positions of the stellar COM in x, y and z (in pc)"""
        
        r = np.sqrt(x**2 + y**2 + z**2)
        COM_x = np.sum(x[r<radius_limit] * star_mass_array[r<radius_limit]) / np.sum(star_mass_array[r<radius_limit])
        COM_y = np.sum(y[r<radius_limit] * star_mass_array[r<radius_limit]) / np.sum(star_mass_array[r<radius_limit])
        COM_z = np.sum(z[r<radius_limit] * star_mass_array[r<radius_limit]) / np.sum(star_mass_array[r<radius_limit])
        return COM_x, COM_y, COM_z
    
    
    def find_star_vCOM(self, radius_limit, star_mass_array, vx, vy, vz, x, y, z):
        
        """This function finds the positions in x, y and z of the center of mass of the stars.
        
        Inputs:
        1) radius_limit: the radius that encompasses all the stellar mass we want to consider (in pc)
        2) star_mass_array: an array from raw data that has all the stars in the system & their masses (in Msun)
        3) vx: the x component of all the stars' velocities (in km/s)
        4) vy: the y component of all the stars' velocities (in km/s)
        5) vz: the z component of all the stars' velocities (in km/s)
        6) x: the x component of all the stars' positions (in pc)
        7) y: the y component of all the stars' positions (in pc)
        8) z: the z component of all the stars' positions (in pc)  
        
        Outputs: values for the velocities of the stellar COM in x, y and z (in km/s)"""
        
        r = np.sqrt(x**2 + y**2 + z**2)
        COM_vx = np.sum(vx[r<radius_limit] * star_mass_array[r<radius_limit]) / np.sum(star_mass_array[r<radius_limit])
        COM_vy = np.sum(vy[r<radius_limit] * star_mass_array[r<radius_limit]) / np.sum(star_mass_array[r<radius_limit])
        COM_vz = np.sum(vz[r<radius_limit] * star_mass_array[r<radius_limit]) / np.sum(star_mass_array[r<radius_limit])
        return COM_vx, COM_vy, COM_vz
        
    
    def star_mass_analysis(self, bin_size, radius_array, star_distance, star_mass_array):
        
        """This function computes the stellar mass profile.
        
        Inputs:
        1) bin_size: the resolution of the simulation (in kpc)
        2) radius_array: an array of radii (in kpc) generated for the analysis. 
        The mass profile is basically the mass enclosed within each value of radius in this array.
        3) star_distance: the magnitude of the star's positions (in kpc) based on the data file.
        Calculated as star_distance = np.sqrt(x**2 + y**2 + z**2)
        4) star_mass_array: an array from the raw data that has all the stars in the system & their masses (in Msun)
        
        Outputs: the stellar mass profile (in Msun), the total mass enclosed (in Msun), the stellar half-mass (in Msun)
        and the stellar half-radius (in kpc)"""
        
        mass_prof_stars = np.zeros(np.size(radius_array))
        h = 0
        for radius_value in radius_array:
            particles = np.where((star_distance < (radius_value + bin_size)))
            masses_in_here = np.sum(star_mass_array[particles])
            mass_prof_stars[h] = masses_in_here
            h += 1
        
        M_tot = mass_prof_stars[np.size(mass_prof_stars) - 1]
        M_half = (1/2) * M_tot
        closest = (np.abs(mass_prof_stars - M_half)).argmin()
        r_half = radius_array[closest]
        
        return mass_prof_stars, M_tot, M_half, r_half
        
    
    def weighted_stdev(self, velocity, star_mass_array, bin_size, star_distance, radius_array):
        
        """This function computes the weighted velocity dispersion of all the stars in one direction (x, y or z).
        
        Inputs:
        1) velocity: an array from the data file representing the velocities of all the stars (in km/s)
        in a certain direction (x, y or z)
        2) star_mass_array: an array from the raw data that has all the stars in the system & their masses (in Msun)
        3) bin_size: the resolution of the simulation (in kpc)
        4) star_distance: the magnitude of the star's positions (in kpc) based on the data file.
        Calculated as star_distance = np.sqrt(x**2 + y**2 + z**2)
        5) radius_array: an array of radii (in kpc) generated for the analysis.
        
        Outputs: an array of the weighted velocity dispersion in a certain direction (in km/s)"""
                
        u = 0
        stdev = np.zeros(np.size(radius_array))
        
        for radius_value in radius_array:
            where = np.where((star_distance < (radius_value + bin_size)))
            velocity_array = velocity[where]
            enclosed_mass = star_mass_array[where]
            weighted_mean = np.sum(velocity_array * enclosed_mass) / np.sum(enclosed_mass)
            unsquared_weighted_stdev = (np.sum(enclosed_mass * (velocity_array - weighted_mean)**2)) / np.sum(enclosed_mass)
            weighted_stdev = np.sqrt(unsquared_weighted_stdev)
            stdev[u] = weighted_stdev
            u += 1

        return stdev
    
    
    def dynamical_mass(self, radius_array, dm_distance, bin_size):
        
        """This function finds the dynamical mass profile of the galaxy, 
        which is just the dark matter mass profile.
        
        Inputs: 
        1) radius_array: an array of radii (in kpc) generated for the analysis.
        2) dm_distance: the magnitude of the dark matter particles' positions (in kpc) based on the data file.
        Calculated as dm_distance = np.sqrt(x**2 + y**2 + z**2)
        3) bin_size: the resolution of the simulation (in kpc)
        
        Outputs: an array for the dark matter mass profile of the galaxy."""
        
        dm_mass_profile = np.zeros(np.size(radius_array))
        h = 0
        for radius in radius_array:
            dm_particles = np.where((dm_distance < (radius + bin_size)))
            how_many_particles = np.size(dm_particles)
            dm_mass_profile[h] = how_many_particles * 500 # 500 Msun is the mass per particle
            h += 1
        
        return dm_mass_profile
    
    
    def Wolf_mass_estimator(self, radius, stdev):
        
        """This function gives an ARRAY of Wolf mass estimates for the stellar half-mass of the galaxy
        depending on the radius and the velocity dispersion at that radius.
        
        Inputs:
        1) radius: the radius at which the Wolf mass estimator is applied (in kpc). 
        Can be a value or an array.
        2) stdev: the corresponding weighted velocity dispersion in a certain direction (in km/s). 
        Can be a value or an array.
        
        Outputs: either one value or one array of Wolf mass estimates."""
        
        G = 4.3009e-3 # pc*(km/s)^2 / Msun
        
        return 3 * (stdev)**2 * (radius*1000) / G
    
    
    def Wolf_mass_at_rhalf(self, velocity, star_mass_array, bin_size, star_distance, dm_distance, radius_array):
        
        """This function gives a Wolf mass estimate for the stellar half-mass of the galaxy
        given the half-light radius (r_half) and the velocity dispersion at that radius.
        
        Inputs:
        1) velocity: an array from the data file representing the velocities of all the stars (in km/s)
        in a certain direction (x, y or z)
        2) star_mass_array: an array from the raw data that has all the stars in the system & their masses (in Msun)
        3) bin_size: the resolution of the simulation (in kpc)
        4) star_distance: the magnitude of the star's positions (in kpc) based on the data file.
        Calculated as star_distance = np.sqrt(x**2 + y**2 + z**2)
        5) dm_distance: the magnitude of the dark matter particles' positions (in kpc) based on the data file.
        Calculated as dm_distance = np.sqrt(x**2 + y**2 + z**2)
        6) radius_array: an array of radii (in kpc) generated for the analysis.
        
        Outputs: the Wolf mass estimate in the x, y or z direction (in Msun)"""
        
        # First, find the dynamical mass profile of the galaxy, which is just the dark matter mass profile.
        
        dm_mass_profile = self.dynamical_mass(radius_array, dm_distance, bin_size)
        
        # Next, this finds the Wolf mass estimator and takes the ratio of that over the dynamical mass at r_half.
        
        G = 4.3009e-3 # pc*(km/s)^2 / Msun

        mass_prof_stars, M_tot, M_half, r_half = self.star_mass_analysis(bin_size,                                                                          radius_array, star_distance, star_mass_array)
        stdev = self.weighted_stdev(velocity, star_mass_array, bin_size, star_distance, radius_array)
        closest = (np.abs(mass_prof_stars - M_half)).argmin()        
        stdev_at_rhalf = stdev[closest]
        M_real = dm_mass_profile[closest]
        
        M_wolf = self.Wolf_mass_estimator(stdev_at_rhalf, r_half)
        
        return stdev_at_rhalf, M_real, M_wolf


# In[3]:


# initiate the analysis by calling the entire class

analyze = VelocityStuff()

