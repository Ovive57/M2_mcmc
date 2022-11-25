cm = plt.get_cmap('tab10') # Colour map to draw colours from

for iiz, zsim in enumerate(zsims):

        '''Definimos colores'''
        cols = []
        col = cm(1. * iiz / len(zsims))
        cols.append(col)
        
        
        plt.style.use(Style.style1)
        plt.plot(Mass_lista, SFR_lista, '.', label='Total SAGE z = ' + zsims[iiz] + '')
        plt.plot(ghist[ind], avSFR[ind], marker='^', linewidth=0, color='k', label='Media SAGE z = ' + zsims[iiz] + '')
        plt.errorbar(ghist[ind], avSFR[ind], yerr=ErrorSFR[ind], xerr=None, fmt='.k')
        plt.ylabel('log$_{10} \;$ (SFR $[M_{\odot} \; h^{-1}\; yr^{-1}$])')
        plt.xlabel('log$_{10} \;$(M$ \; [M_{\odot} \; h^{-1} $])')
        #plt.title('Media de la función SFR SAGE frente bines de masa de las galaxias')
        plt.xlim(8.5, 12)
        #plt.ylim(-2.5,3)
        plotnom = path2sim + 'Figuras/Definitivas/avSFR_vs_Mass_Sage_z_' + zsims[iiz] + '.png'
        plt.legend()
        plt.savefig(plotnom)

        plt.show()
