import pandas as pd, seaborn as sns, scipy.stats as stats, numpy as np
from matplotlib import pyplot as plt
# Read in the data
noc_total_df = pd.read_csv(
            './data/noc_total.csv', index_col=0)
# Set style
sns.set()
title_dict = {'fontsize': 16, 'fontweight': 'bold'}
#Set the colors for top 10 countries and 'OTHER'
noc_colors = sns.color_palette("Paired", n_colors=11)
noc_colors[-1] = (0.0, 0.0, 0.0)
# SET UP LISTS FOR TOP 10 and 20 COUNTRIES
# Get the top 10 and top 20 countries
top_10 = noc_total_df[noc_total_df['Top_10'] == True]
top_20 = noc_total_df[(noc_total_df['Top_20'] == True) & (noc_total_df['Top_10'] == False)]
top_20['NOC'] = '11-20'
# Get the median of 11-20 countries
top_20_med = top_20.groupby('Year').median()
top_20_med['NOC'] = 'Rest'
top_20_med_all = pd.merge(top_10, top_20_med, how='outer')
# Sum the values of all countries not in top 10
not_top_10 = noc_total_df[noc_total_df['Top_10'] == False]
not_top_10['NOC'] = 'Rest'
not_top_10_sum = not_top_10.groupby('Year').sum().reset_index()
not_top_10_sum['NOC'] = 'Rest'
all_count = top_10.merge(not_top_10_sum, how='outer')
# Set the order of the top 10
top_summer_order = top_10.groupby(['NOC'], as_index=False)['Medal']\
        .sum().sort_values(by='Medal', ascending=False).NOC.tolist()
top_summer_order.append('Rest')

#### GRAPHS ####
# Scatterplots - Top 20 medals against athlete and events
hue_order = top_summer_order[:-1]
hue_order.append('11-20')
top_20_all = top_10.merge(top_20, how='outer')
df = top_20_all[top_20_all['Year'] != 1980]
y = 'Medal'
plt.figure(figsize=(16,8))
order = 3
plt.subplot(1,2,1)
ax2 = sns.scatterplot(data=df, y=y, x='Event', hue='NOC', 
                    hue_order=hue_order, palette=noc_colors)
ax2 = sns.regplot(data=df, y=y, x='Event', scatter=False, color='C7', order=order)
ax2.set_ylabel('Number of Medals') 
ax2.set_yticks(np.arange(0,751,50))
ax2.set_xticks(range(0,301,50))
ax2.set_xticklabels(range(0,301,50))
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=0)
ax2.set_xlabel('Number of Events') 
x2 = noc_total_df['Medal']
y2 = noc_total_df['Event']
quad = np.polyfit(x2,y2,order)
v2 = np.polyval(quad, x2)
_,_,least2,p,_ = stats.linregress(v2,y2)
ax2.annotate("r = {:.2f}".format(least2), xy=(.68, .25), 
                xycoords=ax2.transAxes, rotation=30, color='C7')
plt.title('Percentage of Medals and Number of Events', fontdict=title_dict)
ax4 = plt.subplot(1,2,2)
sns.residplot('Event', y, data=df, order=order, color='C7')
ax4.set_ylabel('Residuals') 
ax4.set_xlabel('Number of Events') 
plt.title('Residuals of Polynomial Fit', fontdict=title_dict)
plt.subplots_adjust(top=0.85, left=0.08, right=0.95)
plt.gcf().suptitle('Relationship between the percentage of Total Medals and the Number of Events', fontsize=18)
plt.savefig('./images/graph/countries_medals_resid.png')
plt.show()

# Stacked bar chart - Medal Percentage of NOC
years = noc_total_df.Year.unique().tolist()
plt.figure(figsize=[16,8])
ax = plt.subplot()
bottom = [0]*len(years)
color = 0
for noc in top_summer_order:
    country = all_count[all_count['NOC']==noc]
    noc_perc = country.Games_Medal_Perc.tolist()    
    for year in years:
        if year not in country.Year.unique():
            noc_perc.insert(years.index(year),0)
    plt.bar(years, noc_perc, bottom=bottom, color=noc_colors[color], 
                label=noc, width=2, align='center')
    bottom = [sum(i) for i in zip(bottom, noc_perc)]     
    color += 1
ax.set_xticks(years)
ax.set_yticks(range(0,101,5))
plt.xlabel('Years')
plt.ylabel('Percentage of Total Games Medals')
ax.set_yticklabels(['{}%'.format(x) for x in ax.get_yticks()])
plt.legend()
plt.subplots_adjust(top=0.9, left=0.08, right=0.95)
plt.title('The percentage of medals awarded to the top 10 countries compared to all countries', 
            fontdict=title_dict, pad=15)
plt.savefig('./images/graph/countries_medals_stacked.png')
plt.show()

#Swarmplots - Percentage of total entries
plt.figure(figsize=[16,8])
ax = plt.subplot()
sns.swarmplot(data=top_20_med_all, x='NOC', y='Games_Entries_Perc', 
                order=top_summer_order, palette=noc_colors)
plt.xlabel('The Top 20 Countries')
plt.ylabel('Percentage of Total Games Entries')
ax.set_yticks(range(1,11))
ax.set_yticklabels(['{}%'.format(x) for x in ax.get_yticks()])
plt.xlabel('The Top 20 Countries')
plt.ylabel('Percentage of total entries')
plt.title('The Percentage of Entries from the top 20 countries', 
            fontdict=title_dict, pad=15)
legend = top_summer_order[:-1]
legend.append('Top 11 to 20')
plt.legend(legend)
plt.subplots_adjust(top=0.9, left=0.08, right=0.95)
plt.savefig('./images/graph/countries_entryperc_swarm.png')
plt.show()