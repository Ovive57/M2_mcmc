import numpy as np
import emcee
import matplotlib.pyplot as plt


""" BUT """
# On veut dessiner des samples de la densité gaussienne multivariable suivante:
# p(a) prop exp[-0.5(x-nu)^T/sigma (x-nu)]

# nu: N-dim moyenne
# sigma : carré N-par-N de la matrice de covariance.

def log_prob(x,mu,cov):
	diff = x-mu
	return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))
	# linalg.solve(a,b) resoudre le système ax = b. Returns x
	# dans notre cas cov x = diff --> x = cov/diff
	# c'est l'inverse de la racine carré de l'exponente d'une loi normale
	# la racine carré de l'inverse de X^2.

# x : position of a single "walker" (N-dimensional numpy array)
# mu et cov, constantes à chaque fois qu'on appelle la fonction. Les valeurs viennent des arguments de notre EnsembleSampler (on verra plus tard)

# On va mettre des valeurs specifiques pour ces parametres (mu et cov) en 5D:

ndim = 5

np.random.seed(42)
means = np.random.rand(ndim) # Vecteur 5-dim
#print(means)

cov = 0.5 - np.random.rand(ndim**2).reshape((ndim, ndim)) # Matrice 5x5
#print(cov)
cov = np.triu(cov) # Enleve les valeurs en dessous de la diagonale
#print(cov)
cov += cov.T -np.diag(cov.diagonal()) # Crée une matrice symetrique
#print(cov)
cov = np.dot(cov,cov) # cov ^ 2. C'est aussi une matrice symétrique
#print(cov)

# En résumé on a un 5-vecteur avec 5 moyennes et une matrice symètrique, matrice de covariance au carré.

# On va utiliser 32 "walkers".
# On a besoin de un point de start pour chaque walker (32 points de start)

# Comme on est en 5D on doit donner une position en 5D pour chaque walker. 
# On va avoir une matrice "position" 32x5

nwalkers = 10
p0 = np.random.rand(nwalkers,ndim)

# Grace à emcee on peut avoir notre sample : 

# log_prob: C'est une fonction qui prends un vecteur dans l'espace de parametres comme input et returns le logarithme naturel de la probablité posterieure pour cette position.

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args =[means,cov])

# Sampler c'est une class à partir de laquelle on peut calculer differents choses:

# compute_log_prob(coords) : Calcule le vecteur de probabilité (en log) pour les walkers. Coords c'est le vecteur position dans l'espace de parametres où la probabilité doit être calculée.

# get_autocorr_time(**kwargs) : Compute une estimation du temps de autocorrelation pour chaque parametre

# See more : https://emcee.readthedocs.io/en/stable/user/sampler/

# Quand on a mis our sampler avec le args argument on a dit que la fonction de probabilité doit être appele comme log:prob(p0[0],means,cov)

""" BURN-IN steps """

# Dans la class EnsembleSampler, il y a une fontion qui iterate sample() for nsteps iterations and return the result: run_mcmc(initial_state, nsteps)

# Il y a une autre fonction qui reset les parametres comptabilités:


state = sampler.run_mcmc(p0, 100)

sampler.reset()

# On a gardé la position finale des walkers après 100 pas dans la variable state.
# reset() dégage tous les parametres importants comptabilités dans le sample pour pouvoir avoir un nouveau départ (fresh start). Ça dégage aussi la position actuelle des walkers, c'est pour ça qu'on l'a gardé dans state.

""" Production """
#Maintenant on fait notre production pour 10000 pas:
sampler.run_mcmc(state, 10000)

# On peut acceder au sample avec la fonction get_chain(**kwargs): obtient les chains de MCMC samples qui sont stored.
# Elle va donner un array avec une shape (10000, 32, 5) avec les valeurs des parametres pour chaque walker et chaque pas dans la chaîne.

samples = sampler.get_chain()
print("shape de samples:", np.shape(samples))

# On peut dessiner histograms de ces samples pour obtenir une estimation de la densité qu'on sample:

plt.hist(samples[:, 1], 100, histtype="step")
"""
plt.hist(samples[:, 2], 100, histtype="step")
plt.hist(samples[:, 3], 100, histtype="step")
plt.hist(samples[:, 4], 100, histtype="step")
plt.hist(samples[:, 0], 100, histtype="step")
"""
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$p(\theta_1)$")
plt.gca().set_yticks([]);
plt.show()

# Une autre façon de tester si le sample c'est bien passé c'est de chequer la fraction d'acceptance de la moyenne de l'ensemble avec la fonction acceptance_frction()

print(
    "Mean acceptance fraction: {0:.3f}".format(
        np.mean(sampler.acceptance_fraction)
    )
   )

# Et le temps d'autocorrélation intégré (https://emcee.readthedocs.io/en/stable/tutorials/autocorr/#autocorr)

print(
    "Mean autocorrelation time: {0:.3f} steps".format(
        np.mean(sampler.get_autocorr_time())
    )
)














