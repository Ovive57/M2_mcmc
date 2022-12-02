import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
import getdist
from getdist import plots, MCSamples
import emcee


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



#np.random.seed(45)

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
# Mettre plus loin mais pas trop
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

# Tableau de valeurs qui semblent adaptées pour la minimisation du chi 2
etat_init = np.array([amp, mu, sigma, rho_0, r_p])



# Estimation des meilleures valeurs des paramètres avec curve_fit:

param, pcov = curve_fit(modele, rayons, densite, etat_init)

print(param)

# où param = [amp, mu, sigma, rho_0, r_p]


#print("paramètres trouvés:", param) # à print mieux plus tard

#print(np.shape(pcov))

mode_fit = modele(rayons, *param)
#print("model:", mode_fit)


"""
#Plot du modèle avec un premier fit de paramètres:
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
    """Fonction de proposition

    Args:
        etat_act (5-array): paramètres initiaux
        dev_act (5-array): déviations standards associés

    Returns:
        5-array: paramètres proposés pour tester
    """
    etat_test = np.random.normal(etat_act, dev_act)
    return etat_test

# Mettre tous les parametres en un vecteur
theta = [amp, mu, sigma, rho_0, r_p]

#dev_act = np.array([0.1e-2,50,0.2,50,10])
#prop = proposition(theta, dev_act)


def log_prior(theta):
    """Verification que les parametres sont logiques

    Returns:
        int/float: 0 (int) si valide -inf(float) si pas valide
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
    """
    plt.plot(r, model)
    plt.plot(r, d)
    plt.title('Model likelihood')
    plt.xscale('log')
    plt.show()
    """
    #print("model:", model)
    #print("le model",model) #á chaque fois il veut voir ça. Il n'y a pas de nan, top
    chi2 = np.dot(d - model, np.dot(cov, d - model))
    #print("chi2:", (-1/2)*chi2)
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
    #print(ll)
    return lp + ll # si on ne specifie pas d =  et r = ça reconnais pas, c'est pour ça que j'ai mis des noms plus courts aussi (d et r en lieu de densité et rayons)



# Exemple
#a = log_probability(*theta, d = densite, r = rayons, cov = cov)
#print("test proba", a)


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
    #print("test", test)
    act = log_probability(etat_act, d = densite, r = rayons, cov = cov)
    #print("act", act)
    return test-act

#a = log_acceptance(theta, prop, densite, rayons, cov)
#print("theta", theta)
#print("prop",prop)
#print("alpha", np.exp(a))


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
    #print("etat_test", etat_test)
    alpha = np.exp(log_acceptance(etat_act, etat_test, densite, rayons, cov))
    #print("alpha", alpha) # alpha est toujours 0
    if u <= alpha :
        
        return etat_test # On accepte les nouveaux paramètres
    else:
        return etat_act # On rejete les nouveaux paramètres et on garde l'état actuel

def algorithme(etat_act, pas, densite, rayons, cov, npas):
    """Fonction qui fait tout l'algorithme avec plusieurs pas

    Args:
        etat_act (5-array): état actuel, initial
        pas (5-array): taille possible du pas (déviation standard de chaque état)
        densite (array):  densité. y-axis de nos données.
        rayons  (array): rayon. x-axis de nos données.
        cov (matrix): matrice de covariance inverse.
        npas (int): nombre de pas dont on fait l'algorithme

    Returns:
        _type_: _description_
    """
    matrice_param = []
    for i in range(npas):
        new = test_param(etat_act, pas, densite, rayons, cov)
        matrice_param.append(new)
        etat_act = new # Avant on faisait pas ça, on partait tout le temps de l'etat initial, du coup on bougeait jamais.   
    return np.array(matrice_param)



pas = np.array([0.2, 50, 10,0.1e-2,50])

"""
chaine = algorithme(theta, pas, densite, rayons, cov, npas = 10000)


rho_0_ev = chaine[:,3]
r_p_ev = chaine[:,4]



npas = np.linspace(0,10000,10000)


plt.plot(npas, rho_0_ev)
plt.show()

plt.plot(r_p_ev, rho_0_ev)
plt.show()

"""

############# EMCEE #########################


mat_pos = etat_init
for i in range(9):
    amp = np.random.uniform(4,15)
    mu = np.random.uniform(1000,2000)
    sigma = np.random.uniform(100,300)
    rho_0 = np.random.uniform(0.,1)
    r_p = np.random.uniform(100,2500)
    etat_init = np.array([amp, mu, sigma, rho_0, r_p])
    mat_pos = np.vstack((mat_pos, etat_init))

# nwalkers : nombre de chaines de Markov
# ndim : nombre de parametres
nwalkers, ndim = mat_pos.shape


sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = [densite, rayons, cov]) # State marchait pas parce que on avait args = [rayons, densite, cov] xd

step = 1000
step_burnin = int(0.1*step)

state =sampler.run_mcmc(mat_pos, step_burnin) 

sampler.reset() #burn-in

sampler.run_mcmc(state, step, progress=True)


chaines = sampler.get_chain()
print("shape de chaines:", np.shape(chaines)) # On a 10 chaines, de 1000 échantillons avec 5 parametres chaque chaine.
print("shape échantillons",np.shape(chaines[0])) # Il y en a 1000 échantillons
print("shape premiere chaine", np.shape(chaines[:,0,:])) # 1000 échantillons 5 parametres
# chaines[a,b,c] où a = échantillons(1000), b = chaines(10), c = paramétres(5)

# On peut dessiner histograms de ces samples pour obtenir une estimation de la densité qu'on sample:

"""
plt.hist(chaines[:, 1], 100, histtype="step")

plt.hist(chaines[:, 2], 100, histtype="step")
plt.hist(chaines[:, 3], 100, histtype="step")
plt.hist(chaines[:, 4], 100, histtype="step")
plt.hist(chaines[:, 0], 100, histtype="step")

plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$p(\theta_1)$")
plt.gca().set_yticks([]);
plt.show()

"""

#state =sampler.run_mcmc(mat_pos, 100, progress=True)

#chaines = sampler.get_chain()
#print(np.shape(chaines))


# test de convergence Gelman-Rubin
# Florian m'a dit qu'il faut faire un test pour chaque parametre !
# Je vais faire pour l'amplitude

# NB : j - nombre chaines cad 10, i - len(chaines) cad 1000


"""Je pense qu'on pourrait faire deux fonctions pour calculer les moyennes selon le parametre (moyenne_chaine(parametre) et moyenne_echant(parametre) et après dans test_convergence(m_chaine, m_echant, nwalkers, step, parametre) appeler les fonctions moyennes avec le parametre"""

# 1. Moyenne pour parametre amplitude de la chaine pour chaque chaine:

#moyenne_chaine = np.array([(1/step)*np.sum(chaines[:,i,0]) for i in range(10)])
moyenne_chaine = [(1/step)*np.sum(chaines[i,j,0] for i in range(step)) for j in range(nwalkers)]

# 2. Moyenne pour tous les échantillons:

moyenne_echant = (1/nwalkers)*np.sum(moyenne_chaine)

def test_convergence(m_chaine, m_echant, nwalkers, step):
    
    # 3. Variance entre chaines:

    """B tient bien vers 0 si on augmente nsteps donc trop bien!"""
    B = (1/(nwalkers-1))*np.sum([(m_chaine[j]-m_echant)**2 for j in range(nwalkers)])


    # 4. Moyenne des variances de chaque chaîne :

    """ Je sais pas si W c'est bien, il est tard là, peut-être que j'ai fait de la merde, si tu    peux vérifier, s'il te plaît <3 """

    """Je pense que c'est pas bien parce que plus on augmente nsteps plus augmente R, ça devrait descendre à 1"""

    W = (1/nwalkers)*np.sum([(1/(step-1))*np.sum([(m_chaine[j]-m_echant)**2 for j in range(nwalkers)]) for i in range(step)])



    R = ((((step-1)/step) * W) + (((nwalkers-1)/nwalkers) * B))/W

    return(R)

R = test_convergence(moyenne_chaine, moyenne_echant, nwalkers, step)
# print(R)





