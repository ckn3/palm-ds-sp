import numpy as np
import pandas as pd
import geopandas
import pysal
import seaborn as sns
import contextily as ctx
import rasterio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.cluster import DBSCAN
from pointpats import distance_statistics, QStatistic, random, PointPattern

csv_path = "vrtresults/filtered_predictions_JAMACOAQUE2_rt-1.csv"
tif_path = "images/site2/JAMACOAQUE2.tif"
data = pd.read_csv(csv_path)
data.info()

# Load TIFF to get bounds
with rasterio.open(tif_path) as src:
    bounds = src.bounds
    print(f"TIFF Bounds: {bounds}")
    img = src.read(1)  # Read the first band

extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
coordinates = data[["Longitude", "Latitude"]].values

# ##########################################################################################
# Create the KDE plot with the basemap

# Create a jointplot with scatter and histograms
joint_axes = sns.jointplot(x="Longitude", y="Latitude", data=data, kind="scatter", color="k", s=5, marginal_kws={'color': 'steelblue', 'bins': 50})

# Overlay the TIFF image in the background of the scatter plot
joint_axes.ax_joint.imshow(img, extent=extent, origin='upper', cmap='gray', aspect='auto', zorder=1)

# Add a KDE plot on top of the image but beneath the scatter plot
kde = sns.kdeplot(x="Longitude", y="Latitude", data=data, n_levels=50, fill=True, alpha=0.5, cmap="Reds", ax=joint_axes.ax_joint, zorder=2)

# Scatter plot again to ensure it is visible on top of the KDE
joint_axes.ax_joint.scatter(data['Longitude'], data['Latitude'], color='limegreen', s=0.5, zorder=3)

# Set plot limits to the bounds of the image to eliminate any white space around the image
joint_axes.ax_joint.set_xlim([bounds.left, bounds.right])
joint_axes.ax_joint.set_ylim([bounds.bottom, bounds.top])

# Ensure the axis ticks are visible and meaningful
joint_axes.ax_joint.set_xticks(np.linspace(bounds.left, bounds.right, num=5))
joint_axes.ax_joint.set_yticks(np.linspace(bounds.bottom, bounds.top, num=5))
joint_axes.ax_joint.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))  # Format for longitude
joint_axes.ax_joint.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))  # Format for latitude
joint_axes.ax_joint.tick_params(axis='both', which='both', length=0)  # Hide tick marks


# Save the combined plot
joint_axes.savefig('spatial-patterns/joint_kde_with_tif_basemap.png', dpi=300, bbox_inches='tight')
plt.show()


##########################################################################################
# Do the Q test

# meters_per_degree = 111300  # Approximate conversion factor at equator
# box_size = 100  # Size of each square box in meters
# width_m = (bounds.right - bounds.left) * meters_per_degree
# height_m = (bounds.top - bounds.bottom) * meters_per_degree
# nx = int(width_m / box_size)
# ny = int(height_m / box_size)

# qstat = QStatistic(coordinates, nx=nx, ny=ny)
# qstat.plot()

# # Set tick formats and save the plot
# plt.gca().xaxis.set_major_locator(ticker.LinearLocator(5))
# plt.gca().yaxis.set_major_locator(ticker.LinearLocator(5))
# plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
# plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
# plt.savefig('spatial-patterns/qstat_plot.png', dpi=300, bbox_inches='tight')
# plt.show()

# print("The chi-squared test statistic for the observed point pattern", qstat.chi2)
# print("The degree of freedom for the observed point pattern", qstat.df)
# print("The chi-squared test p-value is", qstat.chi2_pvalue)


##########################################################################################
# F and G functions

# Calculate the G and F functions with simulations
g_test = distance_statistics.g_test(coordinates, support=40, keep_simulations=True)
f_test = distance_statistics.f_test(coordinates, support=40, keep_simulations=True)
j_test = distance_statistics.j_test(coordinates, support=40, keep_simulations=True)

print(f"The mean p-value for the G test is {np.mean(g_test.pvalue):.3f}")
print(f"All p-values for the G test are {[f'{p:.3f}' for p in g_test.pvalue]}")
print(f"The mean p-value for the F test is {np.mean(f_test.pvalue):.3f}")
print(f"All p-values for the F test are {[f'{p:.3f}' for p in f_test.pvalue]}")

# Create a figure with subplots for G function, F function
f, ax = plt.subplots(3, 1, figsize=(6, 6), gridspec_kw=dict(height_ratios=(3, 3, 3)))

# G function plot
ax[0].plot(g_test.support * 111300, g_test.simulations.T, color="k", alpha=0.01)
ax[0].plot(g_test.support * 111300, np.median(g_test.simulations, axis=0), color="cyan", label="Simulation")
ax[0].plot(g_test.support * 111300, g_test.statistic, color="orangered", label="Observed")
for x, p, val in zip(g_test.support, g_test.pvalue, g_test.statistic):
    if p < 0.01:
        ax[0].scatter(x * 111300, val, color='red', zorder=10)
    elif p < 0.05:
        ax[0].scatter(x * 111300, val, color='orange', zorder=10)
ax[0].set_xlabel("Distance (meters)")
ax[0].set_ylabel("Cummulative Portion")
ax[0].set_xlim(0, max(g_test.support) * 111300)
ax[0].set_title(r"Ripley's $G(d)$ function")

# F function plot
ax[1].plot(f_test.support * 111300, f_test.simulations.T, color="k", alpha=0.01)
ax[1].plot(f_test.support * 111300, np.median(f_test.simulations, axis=0), color="cyan", label="Simulation")
ax[1].plot(f_test.support * 111300, f_test.statistic, color="orangered", label="Observed")
for x, p, val in zip(f_test.support, f_test.pvalue, f_test.statistic):
    if p < 0.01:
        ax[1].scatter(x * 111300, val, color='red', zorder=10)
    elif p < 0.05:
        ax[1].scatter(x * 111300, val, color='orange', zorder=10)
ax[1].set_xlabel("Distance (meters)")
ax[1].set_ylabel("Cummulative Portion")
ax[1].set_xlim(0, max(f_test.support) * 111300)
ax[1].set_title(r"Ripley's $F(d)$ function")

# J function plot using official style
ax[2].plot(j_test.support * 111300, j_test.statistic, color='orangered', label="Observed")
ax[2].axhline(1, linestyle=':', color='k', label=r"$J(d) = 1$")  # Reference line at 1
already_added = False
for x, val in zip(j_test.support, j_test.statistic):
    if val < 1 and not already_added:
        ax[2].scatter(x * 111300, val, color='red', label=r"$J(d) < 1$")
        already_added = True  # Ensure only one legend entry for J(d) < 1
    elif val < 1:
        ax[2].scatter(x * 111300, val, color='red')  # Continue plotting without adding to legend
ax[2].set_xlabel("Distance (meters)")
ax[2].set_ylabel("Ratio")
ax[2].set_title(r"Ripley's $J(d)$ function")
ax[2].set_xlim(0, max(j_test.support) * 111300)
ax[2].legend(loc='upper right')

# Adjusting legend to include new entries for clarity
handles, labels = ax[0].get_legend_handles_labels()
handles.extend([
    plt.Line2D([0], [0], marker='o', color='w', label='$p < 0.01$', markersize=10, markerfacecolor='red'),
    plt.Line2D([0], [0], marker='o', color='w', label='$0.01 \leq p < 0.05$', markersize=10, markerfacecolor='orange')
])
ax[0].legend(handles=handles, loc='lower right')
ax[1].legend(handles=handles, loc='lower right') 

f.tight_layout()
f.savefig('spatial-patterns/g_f_j_plot.png', dpi=300, bbox_inches='tight')
plt.show()