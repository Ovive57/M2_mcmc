import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
import getdist
from getdist import plots, MCSamples


def bruit(psd, sigma):
    """
    Crée du bruit coloré à partir de sa densité spectrale de puissance.

    Args:
        psd (array): densité spectrale de puissance du bruit coloré dans l'espace de Fourier.
        sigma (int): est la déviation standard du bruit

    Returns:
        array : le bruit coloré dans l'espace réel
    
    """
    bruit_blanc = np.random.normal(0,sigma,len(psd))
    fourier_blanc = np.fft.fft(bruit_blanc)

    fourier_color = fourier_blanc*np.sqrt(psd)
    bruit_color = np.real(np.fft.ifft(fourier_color))

    return bruit_color

def covariance(psd, rayons, n_realis, sigma):
    """
    Calcule l'inverse de la matrice de covariance de 1000 réalisations du bruit en 1 s

    Args:
        psd (array): densité spectrale de puissance du bruit coloré
        rayons (array): abscisse des données
        n_realis (int): est le nombre de réalisations de bruits 

    Returns:
        array : Matrice de covariance (Cn) et son inverse (cov)
    """
    n_donnees = len(psd)
    Cn = np.zeros((n_donnees,n_donnees))
    for i in range(n_realis):
        bruit_color = bruit(psd, sigma)
        Cn += np.dot(np.transpose(bruit_color[None, 0:n_donnees]), bruit_color[None, 0:n_donnees])
    Cn /= n_realis
    cov = np.linalg.inv(Cn)
    return cov, Cn
 
def modele_init(r, rho_0, r_p ):
    a=1.1
    b=5.5
    c=0.31
    return rho_0 /((r/r_p)**c*(1+(r/r_p)**a)**((b-a)/c))
    
def modele(rayons, amp, mu, sigma, rho_0, r_p ):
    mod = modele_init(rayons, rho_0, r_p )
    mod += amp*stats.norm.pdf(rayons, mu,sigma)
    return mod
    







np.random.seed(45)
    
n_realis = 1000

sigma_bruit = 10**(-3) #cf ennoncé


rayons = np.load('r.npy')
densite = np.load('y.npy')
psd = np.load('psd.npy')
freq_psd = np.load('f.npy')

n_realis = 10000



# Matrice de covariance associée:
    
    
    
cov, cn = covariance(psd, rayons, n_realis,  sigma_bruit)


# Plot du modèle initial:
"""
plt.plot(modele_init(rayons, 100, 0.2 ))
plt.show()
"""

# On trace le modèle avec une gaussienne rajoutée:

sigma = 140
mu = 1500
amp = 10.5
rho_0 = 0.010
r_p = 900

mode = modele(rayons, amp, mu, sigma, rho_0, r_p)

""" Plot du modèle avec notre 1er jeu de paramètres:
plt.plot(rayons, mode)
plt.plot(rayons, densite)
plt.xscale('log')
plt.show()
"""

# Tableau de valeurs qui semblent adaptés pour la minisation du chi 2
tab = np.array([amp, mu, sigma, rho_0, r_p])



# Estimation des meileures valeurs des paramètres avec curve_fit:

param, pcov = curve_fit(modele, rayons, densite, tab)


#print("paramètres trouvés:", param) # à print mieux plus tard

#print(np.shape(pcov))

mode_fit = modele(rayons, *param)

""" Plot du modèle avec un premier fit de paramètres:
plt.plot(rayons, mode_fit)
plt.plot(rayons, densite)
plt.xscale('log')
plt.show()
"""

# A faire : plusieurs etudes de la valeurs de chi2 en fonction du nombre de realisation du bruit
chi2 = np.dot(densite-mode_fit, np.dot(cov, densite-mode_fit))


#print(chi2)

multi_norm = np.random.multivariate_normal(param, pcov)



samples = MCSamples(samples=multi_norm)

g = plots.get_subplot_plotter()
g.triangle_plot(samples, filled=True)





