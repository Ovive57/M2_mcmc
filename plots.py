import style
import inference_bay as m
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import numpy as np
import emcee
import os


# Outils :

plt.style.use(style.style1)
path2plot = 'plots/'

### Some prints :

print("Première estimation paramètres : [amp, mu, sigma, rho_0, r_p] = ", m.etat_est)
print("Paramètres un peu plus loin : [amp, mu, sigma, rho_0, r_p] = ", m.theta_init)
print("Paramètres ajustés avec curve_fit : [amp, mu, sigma, rho_0, r_p] = ", m.param)

print("Le calcul de chi_2 avec le meilleur ajustement: chi_2 = ", m.chi2, "qui effectivemment est de l'ordre de 300 (nb dl)")


# Test Gelman-Rubin pour chaque paramètre :

print("On considérera que les chaînes ont convergé si R < 1.03, avec ",m.step, " pas :")

print("R de l'amplitude:",m.R_amp)
print("R de mu:", m.R_mu)
print("R de sigma:", m.R_sigma)
print("R de rho 0:", m.R_rho0)
print("R de r_p:", m.R_rp)

print("temps d'autocorrelation pour chaque paramètre:", m.tau)

print("Le temps d'éxecution de notre MCMC : ", m.t_mcmc_10, "s, et de la librairie emcee : ", m.t_emcee, "s.")

mcmc_emissions = np.loadtxt('emissions.csv', skiprows=1, max_rows = 1, usecols=(4), unpack=True, delimiter=',')
emcee_emissions = np.loadtxt('emissions.csv', skiprows=2,max_rows = 1, usecols=(4), unpack=True, delimiter=',')

mcmc_emissions_rate = np.loadtxt('emissions.csv', skiprows=1, max_rows = 1, usecols=(5), unpack=True, delimiter=',')
emcee_emissions_rate = np.loadtxt('emissions.csv', skiprows=2, max_rows = 1, usecols=(5), unpack=True, delimiter=',')

print("Émissions de notre MCMC : ", mcmc_emissions*10, ", avec un taux d'émission : ", mcmc_emissions_rate) # Fois 10 parce que 10 chaînes
print("Émissions de emcee : ", emcee_emissions, ", avec un taux d'émission : ", emcee_emissions_rate)

os.remove('emissions.csv')

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
samples_vraisem = MCSamples(samples=m.multi_norm, names=names, labels = labels, label="Vraisemblance")

g = plots.get_subplot_plotter()
g.triangle_plot(samples_vraisem, filled=True)

plotnom = path2plot + 'contours_vraisemblance.pdf'
g.export(plotnom)


### Chaines de Markov ###

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

# Histogrammes de ces samples pour toutes les chaines :

fig, axes = plt.subplots(2,3, figsize=(15,11))
axes[1][2].set_visible(False)
axes[1][0].set_position([0.24,0.125,0.228,0.343])
axes[1][1].set_position([0.55,0.125,0.228,0.343])

axes[0][0].hist(m.chaines[:, :, 0], 100, histtype="step")
axes[0][0].set_xlabel("Amplitude")

axes[0][1].hist(m.chaines[:, :, 1], 100, histtype="step")
axes[0][1].set_xlabel("$\mu$")

axes[0][2].hist(m.chaines[:, :, 2], 100, histtype="step")
axes[0][2].set_xlabel("$\sigma$")

axes[1][0].hist(m.chaines[:, :, 3], 100, histtype="step")
axes[1][0].set_xlabel("$\\rho_0$")

axes[1][1].hist(m.chaines[:, :, 4], 100, histtype="step")
axes[1][1].set_xlabel("$r_p$")

axes[0][0].set_yticks([])
axes[0][1].set_yticks([])
axes[0][2].set_yticks([])
axes[1][0].set_yticks([])
axes[1][1].set_yticks([])

axes[0][1].set_title("Histogrammes chaines")
plotnom = path2plot + 'histogrammes.pdf'
plt.savefig(plotnom)


# Fonction d'autocorrelation

parametre = ["amp", "$\mu$", "$\sigma$" , "$\\rho_0$", "$r_p$"]
for k in range(m.nparam):
    fig, ax = plt.subplots()
    for j in range(m.nwalkers):
        f_auto = emcee.autocorr.function_1d(m.chaines[:,j,k]) # fonction d'autocorrelation
        x = np.linspace(1, m.step, m.step)
        ax.plot(x, f_auto, ls='-')
        plt.xscale('log')
        ax.set_title(f"Autocorrelation paramètre {parametre[k]}")
    plotnom = path2plot + f'autocorrelation_{k}.pdf'
    plt.savefig(plotnom)


### Contours de confiance des 5 paramètres vraisemblance vs MCMC ###

names = ["amp", "\mu", "\sigma" , "\\rho_0", "r_p"]
labels = ["amp", "\mu", "\sigma" , "\\rho_0", "r_p"]
#samples_vraisem = MCSamples(samples=m.multi_norm, names=names, labels = labels, label = 'vraisemblance')
samples_mcmc = MCSamples(samples=m.flat_chaines, names=names, labels = labels, label='mcmc')

g = plots.get_subplot_plotter()
g.triangle_plot([samples_mcmc, samples_vraisem], filled=True)

plotnom = path2plot + 'contours_vraisemblance_mcmc.pdf'
g.export(plotnom)
