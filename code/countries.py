import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats

# Read in files
noc_total_df = pd.read_csv(
            './data/noc_total.csv', index_col=0)
host_difference = pd.read_csv(
            './data/host_difference.csv', index_col=0)

# Set style
sns.set()
title_dict = {'fontsize': 16, 'fontweight': 'bold'}

########## THE COUNTRIES #############
# Hosting - Scatter with regression 
sns.set_palette(['C1', 'C0'])
facet = sns.lmplot(data=host_difference, x='Medal_Perc', y='Host_Medal_Perc', robust=True, palette=['C0'], height=7, aspect=2)
plt.plot([0,15],[0,15], color='C1', linewidth=2, linestyle='dashed')
facet.ax.set_xticks(np.arange(0,16,2.5))
facet.ax.set_yticks(np.arange(0,36,2.5))
facet.ax.set_xticklabels(['{}%'.format(x) for x in facet.ax.get_xticks()])
facet.ax.set_yticklabels(['{}%'.format(x) for x in facet.ax.get_yticks()])
facet.ax.set_yticklabels(['{}%'.format(x) for x in facet.ax.get_yticks()])
r, _ = stats.pearsonr(host_difference['Host_Medal_Perc'], host_difference['Medal_Perc'])
facet.ax.annotate("r = {:.2f}".format(r), xy=(.5, .42), xycoords=facet.ax.transAxes, rotation=15, color='C0')
#plt.text(6,5, 'Visitor Medal Percentage = Host Medal Percentage', color='C1', rotation=10)
facet.ax.set_xlim(left=0)
facet.ax.set_ylim(bottom=0)
plt.subplots_adjust(top=0.9, left=0.08, right=0.95)
plt.xlabel('Average Percentage of Games Medals as Visitor')
plt.ylabel('Average Percentage of Games Medals as Host')
plt.title('Comparison of the average percentage of games medals won by countries who have hosted', fontdict=title_dict, pad=15)
plt.legend(['Hosting Linear Model', 'Visitor Medals = Hosting Medals',  'Host country'])
plt.savefig('./images/graph/countries_host_lmplot.png') # , bbox_inches='tight'
plt.show()


# Heatmap of stats for all countries
plt.figure(figsize=[18,12])
noc_labels = ['# Medals',  '# Entries', '# Events', '# Athletes', '# Male', '# Female', 'Medals/Event', 'Events/Athlete']
corr = noc_total_df[['Medal', 'Entries', 'Event', 'Athletes', 'Male', 'Female', 'Medal_Perc', 'Unique_Perc']].corr()
sns.heatmap(corr, annot=True, xticklabels=noc_labels, yticklabels=noc_labels, linewidths=0.5, cmap='coolwarm')
plt.title("Correlation of Partcipation Behaviour of All Countries Competing", fontdict=title_dict, pad=20)
plt.yticks(rotation = 0)
plt.xticks(rotation = 0)
plt.subplots_adjust(top=0.9, left=0.08, right=1.05)
plt.savefig('./images/graph/countries_stats_heatmap.png')
plt.show()


# 3D plot - Population, GDP and medals
sns.set_style("whitegrid")

fig = plt.figure(figsize=[18,8])
ax = fig.add_subplot(121, projection='3d')
df = noc_total_df
z =df.Medal
x =df.Population
y =df.GDP
surf = ax.scatter(x, y, z, marker='o', c=z, cmap='coolwarm')
ax.set_xlabel('Population (millions)')
ax.set_ylabel('GDP (current US$ billions)')
ax.set_zlabel('Number of Medals')
plt.title('Indicators for every country', fontdict=title_dict)
fig.colorbar(surf, shrink=0.5, aspect=15)
ax2 = fig.add_subplot(122, projection='3d')
df = df[(df['GDP'] < 2500) & (df['Population'] < 200)]
z =df.Medal
x =df.Population
y =df.GDP
surf2 = ax2.scatter(x, y, z, marker='o', c=z, cmap='coolwarm')
ax2.set_xlabel('Population (millions)')
ax2.set_ylabel('GDP (current US$ billions)')
ax2.set_zlabel('Number of Medals')
plt.title('Zoom (less than 200mil population & 2500bil GDP)', fontdict=title_dict)


fig.colorbar(surf2, shrink=0.5, aspect=15)
plt.subplots_adjust(top=0.9, left=0.01, right=1, bottom=0.05, wspace = -0.05)
plt.gcf().suptitle('Relationship between the number of medals won by a country and its population and GDP each year', fontsize=20)
plt.savefig('./images/graph/countries_pop_gdp_3d.png')
plt.show()

