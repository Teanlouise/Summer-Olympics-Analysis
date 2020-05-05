import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

athlete_total_df = pd.read_csv(
            './data/athlete_total.csv', index_col=0)
games_total_df = pd.read_csv(
            './data/games_total.csv', index_col=0)
noc_total_df = pd.read_csv(
            './data/noc_total.csv', index_col=0)
games_total_before = pd.read_csv(
            './data/games_total_draft.csv', index_col=0)\
            .reset_index(drop = True)
host_difference = pd.read_csv(
            './data/host_difference.csv', index_col=0)



sns.set()
figsize = [12,8]
year_fig = [16,8]
title_dict = {'fontsize': 14, 'fontweight': 'bold'}

########## THE GAMES #############
games_var_list = ['Entries', 
                    'Athletes', 
                    'Male', 
                    'Female', 
                    'NOC', 
                    'Event', 
                    'Sport', 
                    'Medal']

# Histogram - Distribution of games before and after 1955
games_total_before = games_total_before[
                        (games_total_before['Season'] == 'Summer') 
                        & (games_total_before['Year']<1955)]

plt.figure(figsize=[12,8])
plt.gcf().suptitle('Distribution of Games Attributes for all Summer Olympics', fontdict=title_dict)     
plot = [2,4,0] 
dfs = [games_total_before, 'Before 1955'], [games_total_df, 'After 1955']
for var in games_var_list:
    plot[2] += 1
    plt.subplot(plot[0], plot[1], plot[2])        
    for df in dfs: 
        plt.hist(df[0][var], label=df[1], histtype='bar', alpha=0.7, bins=10)
    plt.xlabel('Number of {var}'.format(var=var))
    if plot[2] == 4:
        plt.legend()
plt.savefig('./images/graph/games_histogram.png')
plt.show()


########## THE ATHLETES #############

# Boxplot - athlete age
plt.figure(figsize=year_fig)
ax = plt.subplot()
sns.boxplot(x='Year' , y='Age', data=athlete_total_df, hue='Sex')
plt.title('The Age of Summer Athletes since 1956 by Gender', fontdict=title_dict, pad=15)
ax.set_yticks(range(10,76,5))
plt.subplots_adjust(top=0.9, left=0.08, right=0.95)
plt.savefig('./images/graph/athlete_age_boxplot.png')
plt.show()

# Violin plot - BMI
plt.figure(figsize=year_fig)
ax = plt.subplot()
sns.violinplot(x='Year', y='BMI', data=athlete_total_df, hue='Sex', split=True, inner='quartile')
plt.title('The BMI of Summer Athletes since 1956 by Gender', fontdict=title_dict, pad=15)
ax.set_yticks(range(5,68, 5))
plt.subplots_adjust(top=0.9, left=0.08, right=0.95)
plt.savefig('./images/graph/athlete_bmi_violinplot.png')
plt.show()

# Barplot - Number of events per athlete
plt.figure(figsize=year_fig)
ax = plt.subplot()
sns.barplot(x='Year', y='Event', data=athlete_total_df, hue='Sex')
ax.set_yticks(np.arange(1,4.1,0.25))
plt.title('The average number of events per athlete since 1956 by gender', fontdict=title_dict, pad=15)
plt.ylabel('Number of events')
plt.subplots_adjust(top=0.9, left=0.08, right=0.95)
plt.savefig('./images/graph/athlete_event_barplot.png', fontdict=title_dict, pad=15)
plt.show()

# Number of medals per athlete
plt.figure(figsize=year_fig)
ax = plt.subplot()
sns.pointplot(x='Year', y='Medal', data=athlete_total_df, hue='Sex')
ax.set_yticks(np.arange(0.1, 0.71, 0.05))
plt.title('The average number of medals per athlete since 1956 by gender', fontdict=title_dict, pad=15)
plt.ylabel('Number of medals')
plt.subplots_adjust(top=0.9, left=0.08, right=0.95)
plt.savefig('./images/graph/athlete_medal_pointplot.png')
plt.show()

# QQplot - Difference of age and BMI for medal winners
athlete_var_list = [['Age', [10, 45], '(years)'], ['BMI', [15,35], '']]
medal_athlete = athlete_total_df[athlete_total_df['Winner']]
non_medal_athlete = athlete_total_df[athlete_total_df['Winner'] == False]

plt.figure(figsize=year_fig)
plot = [1, 2, 0]
for var in athlete_var_list:
    plot[2] += 1
    ax = plt.subplot(plot[0], plot[1], plot[2])
    medal_percentile = medal_athlete[var[0]].quantile(np.arange(0,1,0.01))
    non_medal_percentile = non_medal_athlete[var[0]].quantile(np.arange(0,1,0.01))
    print(medal_percentile)

    plt.scatter(medal_percentile, non_medal_percentile, color='C1')
    plt.scatter(medal_percentile[0.49], non_medal_percentile[0.49], color='black', marker='s')
    plt.scatter(medal_percentile[0.24], non_medal_percentile[0.24], color='black', marker='+')
    plt.scatter(medal_percentile[0.74], non_medal_percentile[0.74], color='black', marker='+')
    plt.plot(var[1],var[1], color='C0', linewidth=2, linestyle='dashed')
    plt.title("Difference in {var}".format(var=var[0]))
    #plt.text(var[1][0], var[1][1], 'Medal Winners = Non Medal Athletes'.format(var=var[0]), rotation=var[1][2], color='C0', fontsize=10)
    plt.xlabel('{var} of Medal Winners {units}'.format(var=var[0], units=var[2]))
    plt.ylabel('{var} of Non Medal Athletes {units}'.format(var=var[0], units=var[2]))
    ax.set_xlim(left=var[1][0], right=var[1][1])
    ax.set_ylim(bottom=var[1][0], top=var[1][1])
    plt.legend(['Medal Winner = Non Medal Winner','Percentiles', 'Median', 'Interquartile Range'])

plt.subplots_adjust(top=0.9, left=0.08, right=0.95)
plt.gcf().suptitle('Comparison of Physical Characteristics of Medal Winners and Athletes', fontdict=title_dict)
plt.savefig('./images/graph/athlete_difference_qqplot.png')
plt.show()



########## THE COUNTRIES #############
# Get the top 10 and top 20 countries
top_10 = noc_total_df[noc_total_df['Top_10'] == True]
top_20 = noc_total_df[(noc_total_df['Top_20'] == True) & (noc_total_df['Top_10'] == False)]
top_20['NOC'] = 'Rest'
### Get the median of 11-20 countries
top_20_med = top_20.groupby('Year').median()
top_20_med['NOC'] = 'Rest'
top_20_med_all = pd.merge(top_10, top_20_med, how='outer')
#### Sum the values of all countries not in top 10
not_top_10 = noc_total_df[noc_total_df['Top_10'] == False]
not_top_10_sum = not_top_10.groupby('Year').sum().reset_index()
not_top_10_sum['NOC'] = 'Rest'
all_count = top_10.merge(not_top_10_sum, how='outer')
##### Set the order of the top 10
top_summer_order = top_10.groupby(['NOC'], as_index=False)['Medal'].sum().sort_values(by='Medal', ascending=False).NOC.tolist()
top_summer_order.append('Rest')
#### Set the colors for top 10 countries and 'OTHER'
noc_colors = sns.color_palette("Paired", n_colors=11)
noc_colors[-1] = (0.0, 0.0, 0.0)



# Plot of difference with hosting
sns.set_palette(['C1', 'C0'])
facet = sns.lmplot(data=host_difference, x='Medal_Perc', y='Host_Medal_Perc', robust=True, palette=['C0'], height=7, aspect=2)
plt.plot([0,15],[0,15], color='C1', linewidth=2, linestyle='dashed')
facet.ax.set_xticks(np.arange(0,16,2.5))
facet.ax.set_yticks(np.arange(0,36,2.5))
facet.ax.set_xticklabels(['{}%'.format(x) for x in facet.ax.get_xticks()])
facet.ax.set_yticklabels(['{}%'.format(x) for x in facet.ax.get_yticks()])
plt.text(6,5, 'Visitor Medal Percentage = Host Medal Percentage', color='C1', rotation=10)
facet.ax.set_xlim(left=0)
facet.ax.set_ylim(bottom=0)
plt.subplots_adjust(top=0.9, left=0.08, right=0.95)
plt.xlabel('Average Percentage of Games Medals as Visitor')
plt.ylabel('Average Percentage of Games Medals as Host')
plt.title('Comparison of the average percentage of games medals won by countries who have hosted', fontdict=title_dict, pad=15)
plt.savefig('./images/graph/countries_host_lmplot.png') # , bbox_inches='tight'
plt.show()

# Heatmap of stats for all countries
plt.figure(figsize=[18,12])
noc_labels = ['# Medals',  '# Entries', '# Events', '# Athletes', '# Male', '# Female', 'Medals/Event', 'Events/Athlete']
corr = noc_total_df[['Medal', 'Entries', 'Event', 'Athletes', 'Male', 'Female', 'Medal_Perc', 'Unique_Perc']].corr()
sns.heatmap(corr, annot=True, xticklabels=noc_labels, yticklabels=noc_labels, linewidths=0.5, cmap='coolwarm')
plt.title("Correlation of Partcipation Behaviour of All Countries Competing", fontdict=title_dict, pad=15)
plt.yticks(rotation = 0)
plt.xticks(rotation = 0)
plt.subplots_adjust(top=0.9, left=0.08, right=1.05)
plt.savefig('./images/graph/countries_stats_heatmap.png')
plt.show()

#Swarmplots of top 10 for games_medal_perc and games_entries_perc
plt.figure(figsize=year_fig)
ax = plt.subplot()
sns.swarmplot(data=top_20_med_all, x='NOC', y='Games_Entries_Perc', order=top_summer_order, palette=noc_colors)
plt.xlabel('The Top 20 Countries')
plt.ylabel('Percentage of Total Games Entries')
ax.set_yticks(range(1,11))
ax.set_yticklabels(['{}%'.format(x) for x in ax.get_yticks()])
plt.xlabel('The Top 20 Countries')
plt.ylabel('Percentage of total entries')
plt.title('The Percentage of Entries from the top 20 countries', fontdict=title_dict, pad=15)
legend = top_summer_order[:-1]
legend.append('Top 11 to 20')
plt.legend(legend)
plt.subplots_adjust(top=0.9, left=0.08, right=0.95)
plt.savefig('./images/graph/countries_entryperc_swarm.png')
plt.show()


# Stacked bar chart of Medal Percentage of NOC
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
    plt.bar(years, noc_perc, bottom=bottom, color=noc_colors[color], label=noc, width=2, align='center')
    bottom = [sum(i) for i in zip(bottom, noc_perc)]     
    color += 1
ax.set_xticks(years)
ax.set_yticks(range(0,101,5))
plt.xlabel('Years')
plt.ylabel('Percentage of Total Games Medals')
ax.set_yticklabels(['{}%'.format(x) for x in ax.get_yticks()])
plt.legend()
plt.subplots_adjust(top=0.9, left=0.08, right=0.95)
plt.title('The Percentage of Medals awarded to each country', fontdict=title_dict, pad=15)
plt.savefig('./images/graph/countries_medals_stacked.png')
plt.show()




# Scatterplots of top 20 medals against athlete, event and entries
top_20_all = top_10.merge(top_20, how='outer')
df = top_20_all[top_20_all['Year'] != 1980]
y = 'Games_Medal_Perc'
plt.figure(figsize=(16,10))
sns.set_style("whitegrid")
plt.subplot(2,2,1)
ax = sns.scatterplot(data=df, y=y, x='Athletes', hue='NOC', hue_order=top_summer_order, palette=noc_colors)
ax = sns.regplot(data=df, y=y, x='Athletes', order=2, scatter=False, color='C7')
ax.legend_.remove()
ax.set_yticks(np.arange(0,26,2.5))
ax.set_xticks(range(0,801,100))
ax.set_xticklabels([0, '', 100, '', 200, '', 300, '', 400, '', 500, '', 600,'', 700, '', 800])
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.set_yticklabels(['{}%'.format(x) for x in ax.get_yticks()])
ax.set_ylabel('Percentage of Total Medals') 
ax.set_xlabel('Number of Athletes') 

plt.subplot(2,2,2)
ax2 = sns.scatterplot(data=df, y=y, x='Event', hue='NOC', hue_order=top_summer_order, palette=noc_colors)
ax2.legend_.remove()
ax2 = sns.regplot(data=df, y=y, x='Event', scatter=False, color='C7', order=3)
ax2.set_ylabel('') 
ax2.set_yticks(np.arange(0,26,2.5))
ax2.set_yticklabels(['{}%'.format(x) for x in ax2.get_yticks()])
ax2.set_xticks(range(0,301,50))
ax2.set_xticklabels(range(0,301,50))
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=0)
ax2.set_xlabel('Number of Events') 

ax3 = plt.subplot(2,2,3)
sns.residplot('Athletes', y, data=df, order=2)
ax3.set_ylabel('Percentage of Total Medals') 
ax3.set_xlabel('Number of Athletes') 
ax4 = plt.subplot(2,2,4)
sns.residplot('Event', y, data=df, order=3)
ax4.set_ylabel('Percentage of Total Medals') 
ax4.set_xlabel('Number of Events') 
ax4.set_ylabel('') 
plt.subplots_adjust(top=0.9, left=0.08, right=0.95)
plt.gcf().suptitle('Relationship between the percentage of Total Medals, and the Number of Events and Athletes', fontdict=title_dict)
plt.savefig('./images/graph/countries_medals_resid.png')
plt.show()


# 3D plot of population and GDP
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=[18,8])
ax = fig.add_subplot(121, projection='3d')
df = noc_total_df
z =df.Medal
x =df.Population
y =df.GDP
ax.scatter(x, y, z, marker='o', c=z, cmap='coolwarm')
ax.set_xlabel('Population (millions)')
ax.set_ylabel('GDP (current US$ billions)')
ax.set_zlabel('Number of Medals')
plt.title('All countries since 1956')

ax2 = fig.add_subplot(122, projection='3d')
df = df[(df['GDP'] < 2500) & (df['Population'] < 200)]
z =df.Medal
x =df.Population
y =df.GDP
ax2.scatter(x, y, z, marker='o', c=z, cmap='coolwarm')
ax2.set_xlabel('Population (millions)')
ax2.set_ylabel('GDP (current US$ billions)')
ax2.set_zlabel('Number of Medals')

plt.subplots_adjust(top=0.9, left=0.08, right=0.95)
plt.gcf().suptitle('Relationship between the number of medals won by a country and its population and GDP', fontdict=title_dict)
plt.savefig('./images/graph/countries_pop_gdp_3d.png')
plt.show()



