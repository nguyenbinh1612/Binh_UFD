## Reading the observational data: r1/2 ======================

import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('r05_ufd.txt',dtype='str')
r05_ufd = np.array(data[:,2], dtype='float32')
sm_ufd = np.array(data[:,3], dtype='float32')
plt.scatter(np.log10(sm_ufd*1e6), np.log10(r05_ufd), marker='^', s=130, edgecolor='black', 
            color='blue', alpha=0.6, label='MW UFDs')

data = np.genfromtxt('r05_dsph.txt',dtype='str')
r05_dsph = np.array(data[:,2], dtype='float32')
sm_dsph = np.array(data[:,3], dtype='float32')
plt.scatter(np.log10(sm_dsph*1e6), np.log10(r05_dsph), marker='^', s=130, edgecolor='black', 
            color='black', alpha=0.6, label='MW dSphs')


## Reading the observational data: dispersion ======================

m_UFD = np.log10([0.0037e6, 0.0041e6, 0.0079e6, 0.014e6, 0.029e6, 0.037e6, 0.23e6])
disp_UFD = [4.0, 6.46, 3.88, 7.93, 5.16, 5.49, 7.67]
m_dSph = np.log10([0.29e6, 0.29e6, 0.38e6, 0.44e6, 0.74e6, 2.3e6, 3.5e6, 5.5e6, 20e6])
disp_dSph = [9.05, 9.02, 6.44, 7.10, 6.76, 8.79, 7.68, 8.99, 10.59]
plt.scatter(m_UFD, disp_UFD, marker='^', s=150, edgecolor='black', alpha=0.6, color='blue', label='MW UFDs')
plt.scatter(m_dSph, disp_dSph, marker='^', s=150, edgecolor='black', alpha=0.6, color='black', label='MW dSphs')



## Reading the observational data: metallicity ======================

aaa = np.loadtxt('gal_dsphs.txt')
Mstar_g_dsphs = aaa[:,0]
Mstar_e_dsphs = aaa[:,1]
metal_g_dsphs = aaa[:,2]
metal_e_dsphs = aaa[:,3]

aaa = np.loadtxt('gal_ufds.txt')
Mstar_g_ufds = aaa[:,0]
Mstar_e_ufds = aaa[:,1]
metal_g_ufds = aaa[:,2]
metal_e_ufds = aaa[:,3]


aaa = np.loadtxt('gal_m31.txt')
Mstar_31 = aaa[:,0]
Mstar_e_31 = aaa[:,1]
metal_31 = aaa[:,2]
metal_e_31 = aaa[:,3]

aaa = np.loadtxt('gal_dlrr.txt')
Mstar_dlrr = aaa[:,0]
Mstar_e_dlrr = aaa[:,1]
metal_dlrr = aaa[:,2]
metal_e_dlrr = aaa[:,3]

aaa = np.loadtxt('Simon.txt')
Mstar_Simon = np.abs(aaa[:,0])
fe_h_Simon = aaa[:,1]

plt.errorbar(Mstar_g_ufds, metal_g_ufds, xerr=Mstar_e_ufds, yerr=metal_e_ufds, marker='^', ms=12, mec='blue', alpha=0.6, ls='none', 
             capsize=0, color='blue', label='MW UFDs')
plt.errorbar(Mstar_g_dsphs, metal_g_dsphs, xerr=Mstar_e_dsphs, yerr=metal_e_dsphs, marker='^', ms=12, mec='blue', alpha=0.6, ls='none', 
             capsize=0, color='black', label='MW dSphs')
plt.errorbar(Mstar_31, metal_31, xerr=Mstar_e_31, yerr=metal_e_31, marker='h', ms=10, mec='yellowgreen', alpha=0.5, ls='none', 
             capsize=0, color='yellowgreen', label='M31 dSph')
plt.errorbar(Mstar_dlrr, metal_dlrr, xerr=Mstar_e_dlrr, yerr=metal_e_dlrr, marker='o', ms=10, mec='dodgerblue', alpha=0.5, ls='none', 
             capsize=0, color='dodgerblue', label='LG dIrr')
plt.scatter(Mstar_Simon, fe_h_Simon, marker='^', s=120, edgecolor='black', color='blue', alpha=0.6)
