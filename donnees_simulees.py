import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
import getdist
from getdist import plots, MCSamples
import emcee
import corner

########## MAXIMUM DE VRAISEMBLANCE ##########

def bruit(psd, sigma):
    """
    Créée du bruit coloré à partir de sa densité spectrale de puissance.

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
    """
    Modèle de la partie exponentielle

    Args:
        r (array): data rayons
        rho_0 (float): amplitude
        r_p (float): rayon caractéristique

    Returns:
        array: Function exponentielle
    """
    a=1.1
    b=5.5
    c=0.31
    mod = rho_0 /(((r/r_p)**c)*(1+(r/r_p)**a)**((b-c)/a))
    #print(rho_0, r_p)
    return mod

def modele(rayons, amp, mu, sigma, rho_0, r_p):
    """Modèle avec structure en train de fusionner avec le halo principal, modelisée par une gaussienne.

    Args:
        rayons (array): data rayons
        amp (float): amplitud gaussienne
        mu (float): centre gaussienne
        sigma (float): deviation standard gaussienne
        rho_0 (float): amplitude de la fonction exponentielle
        r_p (float) : rayon caractéristique

    Returns:
        float: Function exponentielle avec una gaussienne
    """
    mod = modele_init(rayons, rho_0, r_p )
    mod += amp*stats.norm.pdf(rayons, mu,sigma)
    return mod


################## Minimisation du chi2 ####################

# Vraies données
rayons = np.load('r.npy')
densite = np.load('y.npy')
psd = np.load('psd.npy')
freq_psd = np.load('f.npy')

# Matrice de covariance associée au bruit:
n_realis_bruit = 10000
sigma_bruit = 10**(-3)
cov = covariance(psd, rayons, n_realis_bruit,  sigma_bruit)


# Premieres valeurs estimées qui semblent adaptées pour la minimisation du chi 2:
amp = 10.5
mu = 1500
sigma = 140
rho_0 = 0.010
r_p = 900
etat_est = np.array([amp, mu, sigma, rho_0, r_p])

# Estimation des meilleures valeurs des paramètres avec curve_fit:
param, pcov = curve_fit(modele, rayons, densite, etat_est) # où param = [amp, mu, sigma, rho_0, r_p]
mode_fit = modele(rayons, *param)

# Calcul de chi2 associé au meilleur ajustement :
chi2 = np.dot(densite-mode_fit, np.dot(cov, densite-mode_fit))

# On tire 1000 jeux de paramètres aléatoirement, centrés sur les mailleurs paramètres 
npoints = 1000
multi_norm = np.random.multivariate_normal(param, pcov, npoints)

# Plot contours de confiance maximum de vraisemblance en script plots.py

##################### MCMC #####################

def proposition(etat_act, dev_act):
    """Fonction de proposition

    Args:
        etat_act (5-array): paramètres initiaux
        dev_act (5-array): déviations standards associés

    Returns:
        5-array: paramètres proposés pour tester
    """
    etat_test = np.random.normal(etat_act, dev_act)
    return etat_test


def log_prior(theta):
    """Verification que les parametres sont logiques
    Args:
        theta (5-array): paramètres
    Returns:
        int/float: 0 (int) si valides -inf(float) si pas valides
    """
    if theta[0] > 0 \
    and theta[1] > 0 \
    and theta[2] > 0 \
    and theta[3] > 0 \
    and theta[4] > 0:
        return 0
    else:
        return -np.inf

def log_likelihood(theta, d, r, cov):
    """Logarithme de la fonction de vraisemblance pour des paramètres donnés

    Args:
        d (array): densité. y-axis de nos données.
        r (array): rayon. x-axis de nos données.
        cov (matrix): matrice de covariance inverse.

    Returns:
        float: logarithme de la fonction de vraisemblance # pas sure du type, vérifier
    """
    model = modele(r, *theta)
    chi2 = np.dot(d - model, np.dot(cov, d - model))
    return (-1/2)*chi2


def log_probability(theta, d, r, cov):
    """Logarithme de la distribution postérieure pour des paramètres initiaux donnés.

    Args:
        d (array): densité. y-axis de nos données.
        r (array): rayon. x-axis de nos données.
        cov (matrix): matrice de covariance inverse.

    Returns:
        float: Logarithme de la distribution postérieure # pas sure, vérifier
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, d = d, r = r, cov = cov)
    return lp + ll 

def log_acceptance(etat_act, etat_test, densite, rayons, cov):
    """Logarithme du rapport d'acceptance à partir des paramétres initiaux et proposés

    Args:
        etat_act (5-array): état actuel, initial
        etat_test (5-array): état proposé
        densite (array):  densité. y-axis de nos données.
        rayons  (array): rayon. x-axis de nos données.
        cov (matrix): matrice de covariance inverse.

    Returns:
        float: Acceptance (en log)
    """
    test = log_probability(etat_test, d = densite, r = rayons, cov = cov)
    act = log_probability(etat_act, d = densite, r = rayons, cov = cov)
    return test-act


def test_param(etat_act, dev_act, densite, rayons, cov):
    """test qui décide entre l'etat proposé ou l'etat actuel

    Args:
        etat_act (5-array): état actuel, initial
        dev_act (5-array): déviations stabdars associés
        densite (array):  densité. y-axis de nos données.
        rayons  (array): rayon. x-axis de nos données.
        cov (matrix): matrice de covariance inverse.

    Returns:
        5-array: Nouveau état actuel
    """
    u = np.random.uniform(0.0,1.0)
    etat_test = proposition(etat_act, dev_act)
    alpha = np.exp(log_acceptance(etat_act, etat_test, densite, rayons, cov))
    if u <= alpha :
        
        return etat_test # On accepte les nouveaux paramètres
    else:
        return etat_act # On rejete les nouveaux paramètres et on garde l'état actuel

def algorithme(etat_act, sig_pas, densite, rayons, cov, npas):
    """Fonction qui fait tout l'algorithme avec plusieurs pas

    Args:
        etat_act (5-array): état actuel, initial
        sig_pas (5-array): déviation standard de chaque état
        densite (array):  densité. y-axis de nos données.
        rayons  (array): rayon. x-axis de nos données.
        cov (matrix): matrice de covariance inverse.
        npas (int): nombre de pas que fait l'algorithme

    Returns:
        matrix(npas x nombre de parametres): matrice des parametres
    """
    matrice_param = []
    for i in range(npas):
        new = test_param(etat_act, sig_pas, densite, rayons, cov)
        matrice_param.append(new)
        etat_act = new 
    return np.array(matrice_param)

# Valeurs un peu plus loin pour plus tard voir qu'ils rejoinent la distribution de densité :
amp = 4.5
mu = 800
sigma = 100
rho_0 = 0.015
r_p = 900

theta_init = np.array([amp, mu, sigma, rho_0, r_p])

# Déviations standards utilisées des paramètres:
sig_pas = np.array([0.2, 50, 10,0.1e-2,50])

# Chaines de Markov
chaine = algorithme(theta_init, sig_pas, densite, rayons, cov, npas = 10000)
rho_0_ev = chaine[:,3]
r_p_ev = chaine[:,4]


############# EMCEE #########################

def position(theta_init, nwalkers):
    """Fonction qui donne differents possitions initiales possibles avec nos données

    Args:
        theta_init (5-array): état initial
        nwalkers (int) : nombre de chaînes de Markov

    Returns:
        matrix(nwalkers x nparametres): matrice des parametres initiaux pour chaque chaîne
    """
    mat_pos = theta_init
    for i in range(nwalkers-1):
        amp = np.random.uniform(4,15)
        mu = np.random.uniform(1400,1600)
        sigma = np.random.uniform(100,300)
        rho_0 = np.random.uniform(1e-3,2e-2)
        r_p = np.random.uniform(200,500)
        etat_init = np.array([amp, mu, sigma, rho_0, r_p])
        mat_pos = np.vstack((mat_pos, etat_init))
    return mat_pos


def test_convergence(chaines, index_param):
    """
    Test de convergence de Gelman-Rubin.

    Args:
        chaines (array) : les chaines de Markov
        index_param (int) : l'index du paramètre que l'on souhaite étudié
    Returns:
        Le R associé
    """
    step, nwalkers, nparam = chaines.shape

    # 1. Moyenne pour parametre amplitude de la chaine pour chaque chaine:

    m_chaine = [np.mean(chaines[:, j, index_param]) for j in range(nwalkers)]

    # 2. Moyenne pour tous les échantillons:

    m_echant = np.mean(m_chaine)

    # 3. Variance entre chaines:

    B = (1 / (nwalkers - 1)) * sum(
        [(m_chaine[j] - m_echant) ** 2 for j in range(nwalkers)])  # tend bien vers 0 si on augmente nsteps

    # 4. Moyenne des variances de chaque chaîne :

    sum1 = 0
    sum2 = 0
    for j in range(nwalkers):
        for i in range(step):
            sum1 += (chaines[i, j, index_param] - m_chaine[j]) ** 2
        sum2 += 1 / (step - 1) * sum1
    W = 1 / nwalkers * sum2

    R = ((step - 1) / step * W + (nwalkers + 1) / nwalkers * B) / W  # plus on augmente nsteps, plus R se rapproche de 1

    return (R)

nwalkers = 10 # Nombre chaines de Markov
ndim = len(theta_init) # Nombre de paramètres
mat_pos = position(theta_init,nwalkers)
step = 8000
step_burnin = int(0.1*step)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = [densite, rayons, cov])
sampler.run_mcmc(mat_pos,step)
chaines = sampler.get_chain(discard=step_burnin) # chaines[a,b,c] où a = échantillons(1000), b = chaines(10), c = paramétres(5)


### On fait le test de Gelman pour chaque paramètre:

step, nwalkers, nparam = chaines.shape

R_amp = test_convergence(chaines, 0)  
R_mu = test_convergence(chaines, 1)
R_sigma = test_convergence(chaines, 2)
R_rho0 = test_convergence(chaines, 3)
R_rp = test_convergence(chaines, 4)


# Fonction d'autocorrelation de chaque chaine pour chaque parametre:

""" Vous pouvez le trouver pas commenté dans le script plots.py vu que c'est là bas qu'on l'utilise"""
"""
for k in range(nparam):
    for j in range(nwalkers):
        f_auto = emcee.autocorr.function_1d(chaines[:,j,k]) # fonction d'autocorrelation
        x = np.linspace(1, step, step)

"""
# Temps d'autocorrelation: ( temps entre 2 points des chaines pour qu'ils soient independants )
tau = sampler.get_autocorr_time() # Il donne un array de 5 valeurs (1 par parametre) avec le nombre de pas dont la chaine a besoin pour "oublier où elle a commencé."

# Maintenant on reprend les chaines en faisant le burning et temps d'autocorrelation + on applatit pour avoir tout

ndiscard = int(np.max(tau))*4
nthin = 50
flat_chaines = sampler.get_chain(discard=ndiscard, thin=nthin, flat=True) # où flat_samples[a,b] :  a = nb de pas qu'on prend finalement et b = nb de parametres


# OIVIA: J'ai commencé une liste avec les trucs qu'il reste à faire, on peut la remplire et la vider selon on avance:


# TO DO : 

# Temps
# Emission Tracker code carbon
# Rapport













