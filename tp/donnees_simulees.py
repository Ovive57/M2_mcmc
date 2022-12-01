import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
import getdist
from getdist import plots, MCSamples
import emcee


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
    return cov
 
def modele_init(r, rho_0, r_p ):
    a=1.1
    b=5.5
    c=0.31
    return rho_0 /(((r/r_p)**c)*(1+(r/r_p)**a)**((b-c)/a))
    
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
    
    
    
cov = covariance(psd, rayons, n_realis,  sigma_bruit)


# Plot du modèle initial:
"""
plt.plot(modele_init(rayons, 100, 0.2 ))
plt.show()
"""

# On trace le modèle avec une gaussienne rajoutée:

amp = 10.5
mu = 1500
sigma = 140
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
etat_init = np.array([amp, mu, sigma, rho_0, r_p])



# Estimation des meileures valeurs des paramètres avec curve_fit:

param, pcov = curve_fit(modele, rayons, densite, etat_init)

print(param)

# où param = [amp, mu, sigma, rho_0, r_p]


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
""" Calcul de chi2, de-commenter après, ça prends du temps """
chi2 = np.dot(densite-mode_fit, np.dot(cov, densite-mode_fit))


#print(chi2)

npoints = 1000
multi_norm = np.random.multivariate_normal(param, pcov, npoints)



samples = MCSamples(samples=multi_norm)

# À refaire propre, labels et titre. style plot. (https://getdist.readthedocs.io/en/latest/plot_gallery.html)

"""
g = plots.get_subplot_plotter()
g.triangle_plot(samples, filled=True)

g.export('image.pdf')
"""


################# MCMC #####################

def proposition(etat_act, dev_act):
    etat_test = np.random.normal(etat_act, dev_act)
    return etat_test

# Mettre tous les parametres en un vecteur

def log_prior(amp, mu, sigma, rho_0, r_p):
    if amp > 7 and amp < 14 and mu > 1400 and mu < 1650 and sigma > 130 and sigma < 230 and rho_0 > 0.0065 and rho_0 < 0.012 and r_p > 800 and r_p < 2100:
        return 0
    else:
        return -np.inf

def log_likelihood(densite, rayons, cov, amp, mu, sigma, rho_0, r_p):
    model = modele(rayons, amp, mu, sigma, rho_0, r_p)
    #print("le model",model) #á chaque fois il veut voir ça. Il n'y a pas de nan, top
    chi2 = np.dot(densite - model, np.dot(cov, densite - model))
    
    return (-1/2)*chi2
    
    
def log_probability(densite, rayons, cov, amp, mu, sigma, rho_0, r_p): # METTRE EN VECTEUR, POUR EMCEE
    lp = log_prior(amp, mu, sigma, rho_0, r_p)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(densite, rayons, cov, amp, mu, sigma, rho_0, r_p)

def log_acceptance(densite, rayons, cov, etat_act, etat_test):
    test = log_probability(densite, rayons, cov, *etat_test) + log_prior(*etat_test)  
    act = log_probability(densite, rayons, cov, *etat_act) + log_prior(*etat_act)
    return test-act

def test_param(densite, rayons, cov, etat_act, dev_act):
    u = np.random.uniform(0.0,1.0)
    etat_test = proposition(etat_act, dev_act)
    alpha = np.exp(log_acceptance(densite, rayons, cov, etat_act, etat_test))
    
    if u < alpha :
        return etat_test # On accepte les nouveaux paramètres
    else: 
        return etat_act # On rejete les nouveaux paramètres et on garde l'état actuel

def algorithme(densite, rayons, cov, etat_act, pas, npas):
    matrice_param = []
    for i in range(npas):
        new = test_param(densite, rayons, cov, etat_act, pas)
        matrice_param.append(new)
    return np.array(matrice_param)


pas = [0.1, 10, 10, 0.001, 10]


chaine = algorithme(densite, rayons, cov, etat_init, pas, npas = 10000)

rho_0_ev = chaine[:,3]
r_p_ev = chaine[:,4]



npas = np.linspace(0,10000,10000)
plt.plot(npas, rho_0_ev)
plt.show()

plt.plot(r_p_ev, rho_0_ev)
plt.show()



############# EMCEE #########################

# nwalkers : nombre de chaines de Markov
# ndim : nombre de parametres

"""
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = [rayons, densite, cov])

state =sampler.run_mcmc(densite, 100)
"""






















