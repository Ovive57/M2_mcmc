import donnees_simulees
import style
import donnees_simulees as m
import matplotlib.pyplot as plt
import getdist
from getdist import plots, MCSamples
import numpy as np


# Outils :

plt.style.use(style.style1)
path2plot = 'C:/Users/Olivia/m2cosmo/mcmc/plots/'

# Prints :

print("Première estimation paramètres : [amp, mu, sigma, rho_0, r_p] = ", m.etat_est)
print("Paramètres un peu plus loin : [amp, mu, sigma, rho_0, r_p] = ", m.theta_init)
print("Paramètres ajustés avec curve_fit : [amp, mu, sigma, rho_0, r_p] = ", m.param)

print("Le calcul de chi_2 avec le meilleur ajustement: chi_2 = ", m.chi2, "qui effectivemment est de l'ordre de 300 (nb dl)")

### Modèle sans structure en train de fusionner ###

fig, ax = plt.subplots()
ax.plot(m.rayons, m.modele_init(m.rayons, m.etat_est[3], m.etat_est[4]), c='b', ls='-', label="Profil amas")
plt.xscale('log')
ax.set_xlabel("r[kpc]")
ax.set_ylabel("Amplitude")
ax.set_title("Profil amas sans structure qui fusionne")
plotnom = path2plot + 'modele_sans_structure.pdf'
ax.legend()
plt.savefig(plotnom)

### Modèle avec structure en train de fusionner avec paramètres de premières valeurs estimées###

fig, ax = plt.subplots()
ax.plot(m.rayons, m.densite, c='r', ls='-', label="Vraies données")
ax.plot(m.rayons, m.modele(m.rayons, *m.etat_est), c='b', ls='-.', label="Modele")


plt.xscale('log')
ax.set_xlabel("r[kpc]")
ax.set_ylabel("Amplitude")
ax.set_title("Profil amas modele vs vraies données")
plotnom = path2plot + 'modele_données.pdf'
ax.legend()
plt.savefig(plotnom)

### Modèle avec structure en train de fusionner avec paramètres fités ###

fig, ax = plt.subplots()
ax.plot(m.rayons, m.densite, c='r', ls='-', label="Vraies données")
ax.plot(m.rayons, m.modele(m.rayons, *m.param), c='b', ls='-.', label="Fit")

plt.xscale('log')
ax.set_xlabel("r[kpc]")
ax.set_ylabel("Amplitude")
ax.set_title("Profil amas curve_fit vs vraies données")
plotnom = path2plot + 'fit_données.pdf'
ax.legend()
plt.savefig(plotnom)

### Contours de confiance associés à notre ajustement de ces 5 paramètres ###

names = ["amp", "\mu", "\sigma" , "\\rho_0", "r_p"]
labels = ["amp", "\mu", "\sigma" , "\\rho_0", "r_p"]
samples_vraisem = MCSamples(samples=m.multi_norm, names=names, labels = labels, label='Vraisemblance')

g = plots.get_subplot_plotter()
g.triangle_plot(samples_vraisem, filled=True)

plotnom = path2plot + 'contours_vraisemblance.pdf'
g.export(plotnom)


# Chaines de Markov :

# rho_0 en fonction du pas :

npas = np.linspace(0,10000,10000)

fig, ax = plt.subplots()
ax.plot(npas, m.rho_0_ev, c='b', ls='-')

ax.set_xlabel("numero de pas")
ax.set_ylabel("$\\rho_0$")
ax.set_title("Variations de $\\rho_0$")
plotnom = path2plot + 'rho_pas.pdf'
plt.savefig(plotnom)

# Variations de rho_0 en fontion de celles de r_p :
fig, ax = plt.subplots()
ax.plot(m.r_p_ev, m.rho_0_ev, c='b', ls='-')

ax.set_xlabel("$r_p$")
ax.set_ylabel("$\\rho_0$")
ax.set_title("$\\rho_0$ en fonction de $r_p$")
plotnom = path2plot + 'rho_rp.pdf' # On a la banane !
plt.savefig(plotnom)