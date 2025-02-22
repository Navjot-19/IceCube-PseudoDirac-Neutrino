import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

#===============================================================================================================
# Colors
color_signal='#0065BD'            #blue
color_background='#E37222'        #pink
color_total='#808080'             #white

#===============================================================================================================
# Read data

current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "../resources/2_psi_square.txt")

(bins,
 bin_centers,
  signal_prediction,
  background_prediction,
  counts_exp)=np.genfromtxt(file_path, skip_header=True).T

#===============================================================================================================
# Error bar
error_exp=np.sqrt(counts_exp)


#===============================================================================================================
# Plot

fig,ax=plt.subplots(figsize=(8,8))

ax.hist(bin_centers,bins=bins,
        weights=signal_prediction,
        color='k', label='signal',
        histtype='step',
        lw=2,
        zorder=99,
        alpha=1
        )
ax.hist(bin_centers,bins=bins,
        weights=signal_prediction,
        color=color_signal,
        lw=2,
        histtype='stepfilled',
        alpha=0.4,
        zorder=99)
ax.hist(bin_centers,bins=bins,
        weights=background_prediction,
        color=color_background,
        label='background',
        histtype='step',
        lw=2,
        zorder=30
)
ax.hist(bin_centers,bins=bins,
        weights=background_prediction,
        color=color_background,
        histtype='stepfilled',
        lw=2,
        alpha=0.15,
        zorder=30
)
ax.hist(bin_centers,bins=bins,
        weights=signal_prediction+background_prediction,
        color=color_total, label='total',
        lw=2,
        histtype='step',
        zorder=10)
ax.errorbar(bin_centers, counts_exp, yerr=error_exp,        #yerr gives error bars
            color='k', label='data',
            linewidth=0,
            elinewidth=1.5,                                 # elinewidth= error bar thickness
            fmt='.',
            markersize=10,
            zorder=100
            )

#===============================================================================================================
# Legends and Labels 

ax.set_xlabel(r"$\hat{\psi}^2$ [deg$^2$]")
ax.set_ylabel(r"Events")

ax.legend(loc='upper right', frameon=True,
          ncol=2,
          bbox_to_anchor=(1,1))                     #bbox to anchor at 0,1 place lower left corner of legend

ax.set_xlim([0.0,bins[-1]])
ax.set_ylim([0.0, 90])
ax.set_yticks([20*i for i in range(6)])

plt.show()