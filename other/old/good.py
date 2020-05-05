import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm  
import numpy as np
from plots import *
from statsmodels.graphics.gofplots import qqplot_2samples


athlete_total_df = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/athlete_total.csv', index_col=0)
games_total_df = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/games_total.csv', index_col=0)
noc_total_df = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/noc_total.csv', index_col=0)

all_df = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/all_data.csv', index_col=0)

#### THE ATHLETE ########

# Athlete split up
male_athlete = athlete_total_df[athlete_total_df['Sex'] == 'M']
female_athlete = athlete_total_df[athlete_total_df['Sex'] == 'F']
medal_athlete = athlete_total_df[athlete_total_df['Winner']]
non_medal_athlete = athlete_total_df[athlete_total_df['Winner'] == False]

athlete_var_list = ['Age', 'BMI', 'Event', 'Medal']
athlete_colors ={'F': 'C6', 'M': 'C9', False: 'C4', True: 'C2'}


# DISTRIBUTION
def athlete_hist(category, *dfs): 
    plt.figure()
    plt.gcf().suptitle('Distribution of Athlete Attributes {category}'.format(category=category)) 
    plot = [1,4,0]   

    for var in athlete_var_list:
        plot_min = int(athlete_total_df[var].min())
        plot_max = int(athlete_total_df[var].max())
        #bins = int(plot_max - plot_min)
        bins = 20

        plot[2] += 1
        ax = plt.subplot(plot[0], plot[1], plot[2])
        histtype_func = lambda x: 'bar' if len(x)==1 else 'step'
        histtype = histtype_func(dfs)
        
        for df in dfs:           
            plt.hist(df[0][var], range=(plot_min, plot_max), bins=bins, label=df[1], histtype=histtype, color=df[2])
            
        ax.set_xticks(range(0, plot_max, 5))
        plt.xlabel(var)
        plt.title('Distribution of {var}'.format(var=var)) 
        plt.legend()
    plt.show()

#athlete_hist('Overall', [athlete_total_df, 'All', 'C7'])
athlete_hist('by Gender',[male_athlete, 'Male', 'C9'], [female_athlete, 'Female', 'C6'])
#athlete_hist('by Winner', [medal_athlete, 'Medal Winner', 'C2'], [non_medal_athlete, 'No Medal', 'C4'])


plot = [1,2,0]
for var in ['Age', 'BMI']:
    plot[2] += 1
    plt.subplot(plot[0],plot[1],plot[2])
    sns.distplot(male_athlete[var], kde=True, color='C9')
    sns.distplot(female_athlete[var], kde=True, color='C6')
    plt.title('The {var} of all athlete over time'.format(var=var))
plt.show()


# RELATIONSHIP
# All
# sns.pairplot(athlete_total_df, kind='scatter', diag_kind='hist', vars=athlete_var_list, plot_kws={'color':'C7'}, diag_kws={'color':'C7'})
# plt.subplots_adjust(top=0.95)
# plt.gcf().suptitle('Overall Relationship of Athlete Variables')
# plt.show()

# Each category
# athlete_hue_list = ['Sex', 'Winner'] #Remove season

# sns.pairplot(athlete_total_df, hue='Sex', kind='scatter', diag_kind='hist', vars=['Age', 'BMI','Event', 'Medal'], palette=athlete_colors)
# plt.subplots_adjust(top=0.95)
# plt.gcf().suptitle('Participation Behaviour of Athletes')
# plt.legend()
# plt.show()


# COMPARISON
# QQ plot
athlete_var_list = [['Age', [10,45]], ['BMI', [10,35]]]
def athlete_qqplot(df1, label1, df2, label2):   
    plot = [1, 2, 0]
    for var in athlete_var_list:
        plot[2] += 1
        plt.subplot(plot[0], plot[1], plot[2])
        df1_age_percentile = df1[var[0]].quantile(np.arange(0,1,0.01))
        df2_age_percentile = df2[var[0]].quantile(np.arange(0,1,0.01))
        plt.scatter(df1_age_percentile, df2_age_percentile)
        plt.plot(var[1],var[1], color='black')
        plt.title("{var} Difference".format(var=var[0]))
        plt.text(15,15, 'a=b')
        plt.xlabel(label1)
        plt.ylabel(label2)
    plt.subplots_adjust(top=0.9)
    plt.gcf().suptitle('Comparison of Athlete Variables')
    plt.show()
        
athlete_qqplot(female_athlete, 'Female', male_athlete, 'Male')
athlete_qqplot(medal_athlete, 'Medal Winner', non_medal_athlete, 'No Medal')


# Barplot for Medal and Event, Boxplot for Age and BMI
# 
# plot = [2,2,0]
# for hue in athlete_hue_list:
#     for var in ['Age','BMI']:
#         plot[2] += 1
#         plt.subplot(plot[0], plot[1], plot[2])
#         sns.boxplot(x=hue , y=var, palette=athlete_colors, data=athlete_total_df)
#     for var in ['Event', 'Medal_Perc']:
#         plot[2] += 1
#         plt.subplot(plot[0], plot[1], plot[2])
#         sns.barplot(x=hue , y=var, palette=athlete_colors, data=athlete_total_df)
#     plt.gcf().suptitle('Comparison for all athletes')
#     plt.show()
#     plot[2] = 0

# Boxplot and barplot broken down for each category
# athlete_hue_list = ['Sex', 'Winner']
# athlete_var_list = ['Age', 'BMI', 'Event', 'Medal_Perc']
# plot = [4,2,0]
# for hue in athlete_hue_list:
#     for var in athlete_var_list:   
#         plot[2] += 1
#         plt.subplot(plot[0], plot[1], plot[2]) 
#         if var in ['Age', 'BMI']:
#             sns.boxplot(x=hue , y=var, palette=athlete_colors, data=athlete_total_df)
#         else:
#             sns.barplot(x=hue , y=var, palette=athlete_colors, data=athlete_total_df)
#         for hue2 in athlete_hue_list:
#             if hue != hue2:
#                 plot[2] += 1
#                 plt.subplot(plot[0], plot[1], plot[2])
#                 if var in ['Age', 'BMI']:
#                     sns.boxplot(x=hue , y=var, hue=hue2, palette=athlete_colors, data=athlete_total_df)
#                 else:
#                     sns.barplot(x=hue , y=var, hue=hue2, palette=athlete_colors, data=athlete_total_df)
                  
#     plt.gcf().suptitle('Comparison for all athletes')
#     plt.show()
#     plot[2] = 0




# CHANGE OVER YEARS
#Overall
plot = [2,1,0]
for var in ['Event', 'Medal']:    
    plot[2] += 1
    plt.subplot(plot[0],plot[1],plot[2])
    sns.barplot(x='Year' , y=var, data=athlete_total_df)
    plt.title('The number of {var}s by each athlete over time'.format(var=var))
plt.show()
plot[2] = 0


sns.lineplot(x='Year', y='Medal', data=athlete_total_df)
plt.show()





plot[2] += 1
plt.subplot(plot[0],plot[1],plot[2])
sns.boxplot(x='Year' , y='Age', data=athlete_total_df, hue='Sex',palette=athlete_colors)
plt.title('The {var} of all athlete over time'.format(var=var))
plot[2] += 1
plt.subplot(plot[0],plot[1],plot[2])
sns.violinplot(x='Year', y='BMI', data=athlete_total_df, hue='Sex', palette=athlete_colors, split=True)
plt.title('The {var} of all athlete over time'.format(var=var))
plt.show()
plot[2] = 0





# Each category
# plot = [2,1,0]
# for hue in athlete_hue_list:
#     for var in ['Event', 'Medal']:    
#         plot[2] += 1
#         plt.subplot(plot[0],plot[1],plot[2])
#         sns.barplot(x='Year' , y=var, hue=hue, palette=athlete_colors, data=athlete_total_df)
#         plt.gcf().suptitle('The average of each Athlete Variable over time by {hue}'.format(var=var, hue=hue))
#     plt.show()
#     plot[2] = 0

#     for var in ['Age', 'BMI']:    
#         plot[2] += 1
#         plt.subplot(plot[0],plot[1],plot[2])
#         sns.boxplot(x='Year' , y=var, hue=hue, palette=athlete_colors, data=athlete_total_df)
#         plt.gcf().suptitle('The spread of each Athlete Variale over time by {hue}'.format(var=var, hue=hue))
#     plt.show()
#     plot[2] = 0


# corr = athlete_total_df[['Medal', 'Sex', 'NOC', 'Age', 'BMI', 'Event', 'Medal_Perc']].corr()
# sns.heatmap(corr, annot=True) #linewidths=0.5, cmap='coolwarm'
# plt.show()








games_var_list = ['Entries', 'Athletes', 'Male', 'Female', 'NOC', 'Event', 'Sport', 'Medal']

# DISTRIBUTION
games_total_before = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/games_total_draft.csv', index_col=0).reset_index(drop = True)
games_total_before = games_total_before[(games_total_before['Season'] == 'Summer') & (games_total_before['Year']<1955)]



def games_hist(*dfs): 
    plt.figure()
    plt.gcf().suptitle('Distribution of Total Attributes for each Games')     
    plot = [2,4,0] 

    for var in games_var_list:
        plot[2] += 1
        plt.subplot(plot[0], plot[1], plot[2])        
        for df in dfs: 
            #plt.hist(df[0][var], label=df[1], color=df[2], histtype='bar', alpha=0.8, bins=10)          
            #sns.distplot(df[0][var], label=df[1], kde=True, color=df[2], bins=10)

            # ax = sns.distplot(df[0][var], kde=False)
            # ax2 = ax.twinx()
            # sns.distplot(df[0][var], ax=ax2, kde=True, hist=False)
            # ax2.set_yticks([])
            sns.distplot(df[0][var], label=df[1], bins=15, kde=True, norm_hist=False)
        plt.xlabel('Number of {var}'.format(var=var))
        

        #plt.title('Distribution of {var} at each Games'.format(var=var)) 
    plt.legend(loc='right', bbox_to_anchor=(0, 0), ncol=1)
    plt.show()

#games_hist('Overall', [games_total_df, 'All', 'C7'])
games_hist([games_total_before, 'Before 1955'], [games_total_df, 'After 1955'])





######### THE COUNTRIES ###############

country_var_list = ['Entries', 'Athletes', 'Male', 'Female', 'Event', 'Medal']


#### Host Medal Percentage vs. Average Percentage ####
# Create host percentage and regular percentage df
medals_all = all_df.groupby(['Year', 'NOC', 'Country']).Medal.count().reset_index() #remove season
medals_all.columns=['Year', 'NOC', 'Country', 'Total_Medal'] #remove season
medals_all = medals_all.merge(games_total_df[['Year', 'Medal']], how='outer')
medals_all['Medal_Perc'] = round((medals_all.Total_Medal / medals_all.Medal)*100, 2)
medals_all = round(medals_all.groupby(['NOC', 'Country']).Medal_Perc.mean(),2).reset_index() #remove season
medals_all.columns=['NOC', 'Country', 'Medal_Perc'] # remove season
host_medals = games_total_df[['Year', 'Host_NOC', 'Host_Medal_Perc']] #remove season, games
host_medals.columns=['Year', 'NOC', 'Host_Medal_Perc'] #remove season, games
host_difference = pd.merge(host_medals, medals_all, how='left')

print(host_difference)
print(noc_total_df)



# Plot of difference with hosting
facet = sns.lmplot(data=host_difference, x='Medal_Perc', y='Host_Medal_Perc', robust=True, palette=['C1'])
plt.plot([0,15],[0,15], 'black', linewidth=2, linestyle='dashed')
facet.ax.set_xticks(np.arange(0,15,2.5))
facet.ax.set_yticks(np.arange(0,36,2.5))
plt.text(8,7, 'x=y')
facet.ax.ticklabel_format(useOffset=False)
facet.ax.set_xlim(left=0)
facet.ax.set_ylim(bottom=0)
plt.title('The difference in percentage of medals won by host countries')
plt.show()


# Get the top 20 countries
noc_colors = sns.color_palette("Paired", n_colors=11)
noc_colors[-1] = (0.0, 0.0, 0.0)

top_10 = noc_total_df[noc_total_df['Top_10'] == True]
top_20 = noc_total_df[(noc_total_df['Top_20'] == True) & (noc_total_df['Top_10'] == False)]
top_20_med = top_20.groupby('Year').median()
top_20_med['NOC'] = 'ALL'
#top_20['NOC'] = 'ALL'
top_20_all = pd.merge(top_10, top_20_med, how='outer')

not_top_10 = noc_total_df[noc_total_df['Top_10'] == False]
not_top_10_sum = not_top_10.groupby('Year').sum().reset_index()
not_top_10_sum['NOC'] = 'ALL'
all_count = top_10.merge(not_top_10_sum, how='outer')
years = noc_total_df.Year.unique().tolist()

# Get data of all not in top 10 and set NOC to ALL
top_20_not_med = noc_total_df[(noc_total_df['Top_20'] == True) & (noc_total_df['Top_10'] == False)]
top_20_not_med['NOC'] = 'ALL'
top_20_not_med_count = top_10.merge(top_20_not_med, how='outer')

# Set the order of the top 10
top_summer_order = top_10.groupby(['NOC'], as_index=False)['Medal'].sum().sort_values(by='Medal', ascending=False).NOC.tolist()
top_summer_order.append('ALL')




# Heatmap of all
corr = noc_total_df[['Games_Medal_Perc', 'Medal', 'Games_Entries_Perc', 'Entries', 'Event', 'Athletes', 'Male', 'Female', 'Medal_Perc', 'Unique_Perc']].corr()
sns.heatmap(corr, annot=True) #linewidths=0.5, cmap='coolwarm'
plt.title("All")
plt.yticks(rotation = 0)
plt.xticks(rotation = 0)
plt.show()


# Stacked bar chart of Medal Percentage of NOC
sns.set_style("ticks")
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

ax.set_xticks(range(1956,2017,4))
ax.set_yticks(range(0,101,5))
plt.xlabel('Years')
plt.ylabel('Percentage of medals')
plt.legend()
plt.title('The Percentage of Medals awarded to each country')
plt.show()



#Swarmplots of top 10 for games_medal_perc and games_entries_perc
ax = plt.subplot(2,1,1)
sns.swarmplot(data=top_20_all, x='NOC', y='Games_Entries_Perc', order=top_summer_order, palette=noc_colors)
plt.xlabel('The Top 20 Countries')
plt.ylabel('Percentage of Total Games Entries')
facet.ax.set_xticklabels(['{}%'.format(x) for x in facet.ax.get_xticks()])
facet.ax.set_yticklabels(['{}%'.format(x) for x in facet.ax.get_yticks()])

ax = plt.subplot(2,1,2)
sns.swarmplot(data=top_20_all, x='NOC', y='Medal_Perc', order=top_summer_order, palette=noc_colors)
plt.xlabel('The Top 20 Countries')
plt.ylabel('Medals per Entry')
facet.ax.set_xticklabels(['{}%'.format(x) for x in facet.ax.get_xticks()])
facet.ax.set_yticklabels(['{}%'.format(x) for x in facet.ax.get_yticks()])
plt.show()


# Scatterplots of top 20 medals against athlete, event and entries
plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
plt.subplot(2,2,1)
ax = sns.scatterplot(data=top_20_all, y='Medal', x='Athletes', hue='NOC', hue_order=top_summer_order, palette=noc_colors)
ax = sns.regplot(data=top_20_not_med_count, y='Medal', x='Athletes', order=2, scatter=False, color='C7')
ax.legend_.remove()
ax.set_yticks(range(0,751,50))
ax.set_xticks(range(0,801,100))
ax.set_xticklabels([0, '', 100, '', 200, '', 300, '', 400, '', 500, '', 600,'', 700, '', 800])
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.set_ylabel('Number of Medals') 
ax.set_xlabel('Number of Unique Athletes') 

plt.subplot(2,2,2)
ax2 = sns.scatterplot(data=top_20_all, y='Medal', x='Event', hue='NOC', hue_order=top_summer_order, palette=noc_colors)
ax2.legend_.remove()
ax2 = sns.regplot(data=top_20_not_med_count, y='Medal', x='Event', order=3, scatter=False, color='C7')
ax2.set_ylabel('') 
ax2.set_yticks(range(0,751,50))
ax2.set_xticks(range(0,301,50))
ax2.set_xticklabels(range(0,301,50))
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=0)
ax2.set_xlabel('Number of Unique Events') 

plt.subplot(2,2,3)
sns.residplot('Medal', 'Athletes', data=top_20_not_med_count, order=2)
plt.subplot(2,2,4)
sns.residplot('Medal', 'Events', data=top_20_not_med_count, order=3)


# plt.subplot(1,3,3)
# ax3 = sns.scatterplot(data=top_20_all, y='Medal', x='Entries', hue='NOC', hue_order=top_summer_order, palette=noc_colors)
# ax3 = sns.regplot(data=top_20_not_med_count, y='Medal', x='Entries', scatter=False, color='C7')
# ax3.set_ylabel('')  
# ax3.set_yticks(range(0,751,50))
# ax3.set_xticks(range(0,2251,50))
# ax3.set_xticklabels([0, '', '', '', '', 250, '', '', '', '', 500,'', '', '', '', 750,'', '', '', '', 1000, '', '', '', '', 1250, '', '', '', '', 1500,'', '', '', '', 1750,'', '', '', '', 2000])
# ax3.set_xlim(left=0)
# ax3.set_ylim(bottom=0)
# ax3.legend(loc='right', bbox_to_anchor=(1.25, 0.5), ncol=1)
# ax3.set_xlabel('Number of Entries') 

plt.subplots_adjust(wspace = 0.1, top=0.9)
plt.gcf().suptitle('Participation behaviour of Top 20 countries each Year')
plt.show()













# Distribution of Events per athlete (before and after 1950)
# noc_total_before_1950 = noc_total_df[noc_total_df['Year']<1950]
# noc_total_1950 = noc_total_df[noc_total_df['Year']>1950]
# sns.distplot(noc_total_1950['Unique_Perc'], bins=40, label='After 1950')
# sns.distplot(noc_total_before_1950['Unique_Perc'], bins=40, label='Before 1950')
# plt.title('Distribution of Unique Percentage')
# plt.legend()
# plt.show()



sns.pairplot(top_20_all, kind='scatter', vars=['Athletes', 'Entries', 'Event', 'Medal'], hue='NOC', palette=noc_colors)
plt.show()

g = sns.PairGrid(top_20_all, palette=noc_colors)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=3, legend=False)
plt.show()

# sns.pairplot(top_20_all, kind='scatter', vars=['Athletes', 'Entries', 'Event', 'Medal'], plot_kws={'color':'C7'}, diag_kws={'color':'C7'})
# plt.show()











# Scatterplots - Top 20 medals against athlete and events
order = top_summer_order[:-1]
order.append('11-20')
print(order)
top_20_all = top_10.merge(top_20, how='outer')
df = top_20_all[top_20_all['Year'] != 1980]
y = 'Games_Medal_Perc'
plt.figure(figsize=(18,12))

plt.subplot(2,2,1)
ax = sns.scatterplot(data=df, y=y, x='Athletes', hue='NOC', hue_order=order, palette=noc_colors)
ax = sns.regplot(data=df, y=y, x='Athletes', order=2, scatter=False, color='C7')
#ax.legend_.remove()
ax.set_yticks(np.arange(0,26,2.5))
ax.set_xticks(range(0,801,100))
ax.set_xticklabels([0, '', 100, '', 200, '', 300, '', 400, '', 500, '', 600,'', 700, '', 800])
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.set_xlabel('Number of Athletes') 
ax.set_yticklabels(['{}%'.format(x) for x in ax.get_yticks()])
plt.setp(ax.get_legend().get_texts(), fontsize='8')
x1 = noc_total_df['Athletes']
y1 = noc_total_df['Medal']
quad = np.polyfit(x1,y1,2)
v1 = np.polyval(quad, x1)
_,_,least,p,_ = stats.linregress(v1,y1)
ax.annotate("r = {:.2f}".format(least), xy=(.8, .6), xycoords=ax.transAxes, rotation=30, color='C7')
ax.set_ylabel('Percentage of Total Medals') 
plt.title('Percentage of Medals and Number of Athletes', fontdict=title_dict)

plt.subplot(2,2,2)
ax2 = sns.scatterplot(data=df, y=y, x='Event', hue='NOC', hue_order=order, palette=noc_colors)
#ax2.legend_.remove()
ax2 = sns.regplot(data=df, y=y, x='Event', scatter=False, color='C7', order=3)
ax2.set_ylabel('Percentage of Total Medals') 
ax2.set_yticks(np.arange(0,26,2.5))
ax2.set_yticklabels(['{}%'.format(x) for x in ax2.get_yticks()])
ax2.set_xticks(range(0,301,50))
ax2.set_xticklabels(range(0,301,50))
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=0)
ax2.set_xlabel('Number of Events') 
x2 = noc_total_df['Medal']
y2 = noc_total_df['Event']
quad = np.polyfit(x2,y2,2)
v2 = np.polyval(quad, x2)
_,_,least2,p,_ = stats.linregress(v2,y2)
ax2.annotate("r = {:.2f}".format(least2), xy=(.68, .32), xycoords=ax2.transAxes, rotation=30, color='C7')
plt.title('Percentage of Medals and Number of Events', fontdict=title_dict)

ax3 = plt.subplot(2,2,3)
sns.residplot('Athletes', y, data=df, order=2, color='C7')
ax3.set_ylabel('Percentage of Total Medals') 
ax3.set_xlabel('Number of Athletes') 
ax3.set_yticklabels(['{}%'.format(x) for x in ax3.get_yticks()])
plt.title('Residuals of Quadratic Fit', fontdict=title_dict)

ax4 = plt.subplot(2,2,4)
sns.residplot('Event', y, data=df, order=2, color='C7')
ax4.set_ylabel('Percentage of Total Medals') 
ax4.set_xlabel('Number of Events') 
ax4.set_yticklabels(['{}%'.format(x) for x in ax4.get_yticks()])
plt.title('Residuals of Quadratic Fit', fontdict=title_dict)
plt.subplots_adjust(top=0.9, left=0.08, right=0.95, hspace=0.3)
plt.gcf().suptitle('Relationship between the percentage of Total Medals, and the Number of Events and Athletes', fontsize=18)
