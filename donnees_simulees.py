import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
import getdist
from getdist import plots, MCSamples
import emcee
import corner

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





rayons = np.load('r.npy')
densite = np.load('y.npy')
psd = np.load('psd.npy')
freq_psd = np.load('f.npy')

n_realis = 10000
sigma_bruit = 10**(-3) #cf ennoncé


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
        etat_act = new 
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
    r_p = np.random.uniform(100,1000)
    etat_init = np.array([amp, mu, sigma, rho_0, r_p])
    mat_pos = np.vstack((mat_pos, etat_init))

# nwalkers : nombre de chaines de Markov
# ndim : nombre de parametres
nwalkers, ndim = mat_pos.shape


sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = [densite, rayons, cov]) 

step = 15000
step_burnin = int(0.1*step)

state =sampler.run_mcmc(mat_pos, step_burnin) 

sampler.reset() #burn-in

sampler.run_mcmc(state, step, progress=True)


chaines = sampler.get_chain()

print("shape de chaines:", np.shape(chaines)) # On a 10 chaines, de 1000 échantillons avec 5 parametres chaque chaine.
print("shape échantillons",np.shape(chaines[0])) # Il y a 1000 échantillons
print("shape premiere chaine", np.shape(chaines[:,0,:])) # 1000 échantillons 5 parametres
# chaines[a,b,c] où a = échantillons(1000), b = chaines(10), c = paramétres(5)

# On peut dessiner des histogrammes de ces samples pour obtenir une estimation de la densité qu'on sample:

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

# NB : j - nombre chaines cad 10, i - len(chaines) échantillons cad 1000 



### J'ai commencé à changer à partir de là:
### J'ai mis le test de Gelman dans une seule fonction : le R est pas encore bon je pense, j'ai tout testé pourtant dans la fonction, l'erreur vient surement de nos chaines

### OLIVIA RÉPONDS : JE PENSE QUE JE SUIS D'ACCORD, J'AI CHANGÉ UN PEU LES CHOSES POUR LES VOIR PLUS PROPREMENT, ET JE PENSE QUE C'EST BON... 

### OLIVIA: J'ai separé les moyennes en une autre fonction to keep it simple, normalement c'etail une seule chose par fonction...

### OLIVIA: JE PENSE QUE AVEC CETTE DÉFINITION POUR LES MOYENNES EST PLUS PROPRE, TU ME DIRAS CE QUE TU EN PENSES, SINON ON RÉCUPERE CELLE D'AVANT:

def moyennes(chaines, index_param, step, nwalkers, nparam):

    # 1. Moyenne pour parametre de la chaine pour chaque chaine:
    
    # Pour la chaine 0:
    m_chaine = [np.mean(chaines[:,j,index_param]) for j in range(nwalkers)]
    # C'est un vecteur avec les 10 moyennes de chaque chaine sur toutes les échantillons
    
    # 2. Moyenne pour tous les échantillons:
    
    m_echant = np.mean(m_chaine)
    # C'est juste une valeur avec la moyenne des 10 chaines sur les 1000 échantillons
    
    return m_chaine, m_echant


#step, nwalkers, nparam = chaines.shape
#a = moyennes(chaines, 0, step, nwalkers, nparam)


### OLIVIA: Ici j'ai défini la fonction différement, je sais pas c'est quoi plus claire, je laisse les deux et tu décides si tu veux, je voulais juste voir chaque somme que je faisais:

def test_convergence(chaines, index_param, step, nwalkers, nparam):
    """
    Test de convergence de Gelman-Rubin.
    
    Args:
        les chaines de Markov et l'index du paramètre que l'on souhaite étudié
    Returns:
        Le R associé
    """
    print("step:", step, "nwalkers:", nwalkers)
    B = 0
    W = 0
    
    m_chaine, m_echant = moyennes(chaines,index_param, step, nwalkers, nparam)
    
    # 3. Variance entre chaines:
    
    B = 1/(nwalkers-1)*np.sum([(m_chaine[j]-m_echant)**2 for j in range(nwalkers)])
    print("B=", B)
    
    # 4. Moyenne des variances de chaque chaîne :
    sum1 = 0
    sum2 = 0
    
    for j in range(nwalkers):
        for i in range(step):
            sum1 += (chaines[i,j,index_param] - m_chaine[j])**2
        sum2 += 1/(step-1)*sum1
        
    W = 1/nwalkers*sum2
    
    print("W=", W)
    
    # 5. Quantité R:
    
    R = ((step-1)/step * W + (nwalkers+1)/nwalkers * B)/W
    
    return R
    

#a = test_convergence(chaines, 0, step, nwalkers, nparam)
#print("R=", a)


"""

def test_convergence(chaines, index_param, step, nwalkers, nparam):
   
    Test de convergence de Gelman-Rubin.
    
    Args:
        les chaines de Markov et l'index du paramètre que l'on souhaite étudié
    Returns:
        Le R associé
    
    print("step:", step, "nwalkers:", nwalkers)
    
    m_chaine, m_echant = moyennes(chaines,index_param, step, nwalkers, nparam)
    
    # 3. Variance entre chaines:

    #B tend bien vers 0 si on augmente nsteps donc trop bien!
    B = (1/(nwalkers-1))*np.sum([(m_chaine[j]-m_echant)**2 for j in range(nwalkers)])
    print("B:", B)
    
    # 4. Moyenne des variances de chaque chaîne :
    
    var_chaine = np.array([np.sum((chaines[i, j, index_param]-m_chaine[j])**2 for i in range(step))/(step-1) for j in range(nwalkers)])
    
    W = (1/nwalkers)*np.sum(var_chaine)
    print("W:", W)
    #plus on augmente nsteps, plus R se rapproche de 1
    R = ((step-1)/step * W + (nwalkers+1)/nwalkers * B)/W

    return(R)
"""

### On teste pour chaque paramètre:
step, nwalkers, nparam = chaines.shape
R_amp = test_convergence(chaines, 0, step, nwalkers, nparam)  # Pour l'amplitude
"""
R_mu = test_convergence(chaines, 1)
R_sigma = test_convergence(chaines, 2)
R_rho0 = test_convergence(chaines, 3)
R_p = test_convergence(chaines, 4)
"""
print("On souhaite que R< 1.03 : ")

print("R de l'amplitude:",R_amp)
"""
print("R de mu:",R_mu)
print("R de sigma:",R_sigma)
print("R de rho 0:",R_rho0)
print("R de r_p:",R_p)
"""

### Certaines chaines ca va mais la dernière par exemple est bizarre

# Fonction d'autocorrelation de chaque chaine pour chaque parametre:
# OLIVIA: Ici j'ai fait la boucle pour faire toutes les parametres:


for k in range(nparam):
    for j in range(nwalkers):
        f_auto = emcee.autocorr.function_1d(chaines[:,j,k]) # fonction d'autocorrelation
        x = np.linspace(1, step, step)
    
        #plt.plot(x, f_auto)
        #plt.xscale('log')
        #plt.title(f"Param {k}")
    #plt.show() # en sortant le plt.show() de la boucle tous les graphes se voit en mêeme temps 

""" EXAMPLE DE PLOT POUR PLUS TARD
fig, ax = P.subplots()  # Création d'une figure contenant un seul système d'axes
ax.plot(x, N.sin(x), c='b', ls='-', label="Sinus")    # Courbe y = sin(x)
ax.plot(x, N.cos(x), c='r', ls=':', label="Cosinus")  # Courbe y = cos(x)
ax.set_xlabel("x [rad]")          # Nom de l'axe des x
ax.set_ylabel("y")                # Nom de l'axe des y
ax.set_title("Sinus et Cosinus")  # Titre de la figure
ax.legend() 

"""

# OLIVIA: Normalement on doit voir tous les graphes en même temps pour voir où deux poins des chaînes de Markov ne sont plus correlés ! On va faire des figures prochainement dans le script plots

# Temps d'autocorrelation:

# OLIVIA: Le tau marche pas à cause de que nos chaines convergent pas... c'est pour ça commenté, j'ai continué en inventant un tau plus tard.
""" 
tau = sampler.get_autocorr_time() # Il donne un array de 5 valeurs (1 par parametre) avec le nombre de pas dont la chaine a besoin pour "oublier où elle a commencé."
print(tau) 
"""
# Quand j'ai run tau avec 3000 pas il m'a dit: The chain is shorter than 50 times the integrated autocorrelation time for 5 parameter(s). Use this estimate with caution and run a longer chain! N/50 = 60; tau: [249.76784442 249.27214463 270.99923593 245.43068162 157.99109379] 
# avec ces données du coup je pense que la chaine doit être de minimum 250*50 = 12500. Alors 13000 ça semblerait bien à mon avis
# Plus j'augmente le numero de steps, plus ça augmente le temps de correlation et du coup plus de steps me demande d'ajouter. Je pense qu'il faut réparer le truc d'avant...


#ndiscard = (np.max(tau))*3
#nthin =(np.max(tau))/2
ndiscard = 100
nthin = 15

flat_samples = sampler.get_chain(discard=ndiscard, thin = nthin, flat = True)
print("(nb points pris, nb parametres) = ", np.shape(flat_samples)) #(a,b) où a = nb de pas qu'il dessine finalement et b = nb de parametres

# OLIVIA: et ça c'est le plot, ça marche pas encore, mais on peut le laisser pour la fin avec le script plots, ça se fait pas comme ça en fait, il faut voir ce qu'on a fait avant.

labels = ["amp","mu","sigma"," rho_0","r_p"] #parametres
fig = corner.corner(flat_samples, labels=labels, truths=etat_init)


# OIVIA: J'ai commencé une liste avec les trucs qu'il reste à faire, on peut la remplire et la vider selon on avance:


# TO DO : 

# Temps
# Emission Tracker code carbon
# Rapport
# Plots













