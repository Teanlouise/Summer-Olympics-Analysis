import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm  
import numpy as np
from plots import *
from statsmodels.graphics.gofplots import qqplot_2samples

athlete_colors ={'Winter': 'C0', 'Summer':'C1', 'F': 'C6', 'M': 'C9', False: 'C4', True: 'C2'}

# Read in data and split between summer and winter
all_df = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/all_data.csv', index_col=0).reset_index(drop = True)

## Add winner 
all_df['Winner'] = all_df.Medal.notna()

# Add BMI columns [Weight (kg) / Height^2 (m)]
all_df['BMI'] = all_df.apply(lambda x: round(x.Weight/((x.Height/100)**2), 2), axis=1)


# Split by season
winter_df = all_df[all_df.Season == 'Winter'] 
summer_df = all_df[all_df.Season == 'Summer'] 


# Games totals (Games, Year, #Athletes, #Medals, #Male, #Female, #Events)
games_total_ath = all_df.groupby(['Games']).ID.count().reset_index()
games_total_ath.columns = ['Games', 'Total_ID'] 
games_athletes = all_df.groupby(['Games']).ID.nunique().reset_index()
games_athletes.columns = ['Games', 'Unique_ID'] 
games_events = all_df.groupby(['Games']).Event.nunique().reset_index()
games_sports = all_df.groupby(['Games']).Sport.nunique().reset_index()
games_medals = all_df.groupby(['Games']).Medal.count().reset_index()
games_countries = all_df.groupby(['Games']).NOC.nunique().reset_index()
games_male = all_df[all_df['Sex'] == 'M'].groupby('Games').ID.nunique().reset_index()
games_male.columns = ['Games', 'Male']    
games_female = all_df[all_df['Sex'] == 'F'].groupby('Games').ID.nunique().reset_index()
games_female.columns = ['Games', 'Female']
games_host = all_df[all_df['NOC'] == all_df['Host_NOC']].groupby('Games').Medal.count().reset_index()
games_host.columns = ['Games', 'Host_Medal']
games_visitor = all_df[all_df['NOC'] != all_df['Host_NOC']].groupby('Games').Medal.count().reset_index()
games_visitor.columns = ['Games', 'Visitor_Medal']
games_total_df = all_df[['Games', 'Year', 'Season', 'Host_NOC']]
games_total_df = games_total_df.drop_duplicates()
games_total_df = games_total_df.merge(games_total_ath, how='outer') \
                                .merge(games_athletes, how='outer')\
                                .merge(games_events, how='outer')\
                                .merge(games_sports, how='outer')\
                                .merge(games_medals, how='outer')\
                                .merge(games_countries, how='outer')\
                                .merge(games_host, how='outer')\
                                .merge(games_visitor, how='outer')\
                                .merge(games_male, how='outer')\
                               .merge(games_female, how='outer')

games_total_df['Host_Medal_Perc'] = round((games_total_df.Host_Medal / games_total_df.Medal)*100, 2)
games_total_df['Visitor_Medal_Perc'] = round((games_total_df.Visitor_Medal / games_total_df.Medal)*100, 2)

# winter = games_total_df[games_total_df['Season']=='Winter'][['Season', 'Host_Country', 'Host_Medal_Perc']]
# print(winter)


#print(all_df.Country.unique())

#print(all_df[(all_df['Country'] == 'United States') & (all_df['Year'] == 1932)])



# Create host percentage and regular percentage df
# Add to seperate winter/summer: all_df[all_df['Season']=='Summer'].groupby.....
medals_all = all_df.groupby(['Games', 'NOC', 'Country', 'Season']).Medal.count().reset_index()
medals_all.columns=['Games', 'NOC', 'Country', 'Season', 'Total_Medal']
medals_all = medals_all.merge(games_medals, how='outer')
medals_all['Medal_Perc'] = round((medals_all.Total_Medal / medals_all.Medal)*100, 2)
medals_all = round(medals_all.groupby(['NOC', 'Country', 'Season']).Medal_Perc.mean(),2).reset_index()
medals_all.columns=['NOC', 'Country', 'Season', 'Medal_Perc']
host_medals = games_total_df[['Games', 'Season', 'Host_NOC', 'Host_Medal_Perc']]
#host_medals2 = round(host_medals.groupby(['Host_NOC', 'Season']).Host_Medal_Perc.mean(), 2).reset_index()
host_medals.columns=['Games', 'Season', 'NOC', 'Host_Medal_Perc']
host_difference = pd.merge(host_medals, medals_all, how='left')








# ax = plt.subplot()
# sns.scatterplot(data=host_difference, x='Medal_Perc', y='Host_Medal_Perc', hue='Season', ci='0.025', palette=athlete_colors)
# plt.show()


# sns.regplot(data=host_difference, x='Medal_Perc', y='Host_Medal_Perc', ci=0.025, scatter=True, fit_reg=True)
# plt.show()



# qqplot(host_difference, 
#             x='Medal_Perc', 
#             y='Host_Medal_Perc',
#             height=3, 
#             aspect=1.5, 
#             palette=athlete_colors,
#             display_kws={"identity":True,"fit":True,"reg":True,"ci":0.025})
# plt.show()


# HEREEE






# winter_all['Medal_Perc'] = round((winter_all.Medal / games_total_df.Total_Medal)*100, 2)
# print(winter_all)


# winter_all['Games_Total'] = winter_all.merge(games_total_df.Medal)
# winter_all['Medal_Perc'] = round((winter_all.Medal / 'Games_Total' )*100, 2)
# non_host2 = non_host.groupby(['NOC', 'Season']).Medal.mean()
# host2['Medal'] = host2.merge(non_host2)
# print(host2)



# Athlete Totals (Games, Year, ID, Sex, Age, BMI, Season, NOC, #Events, #Medals)
athlete_events = all_df.groupby(['Games', 'ID']).Event.count().reset_index()
athlete_medals = all_df.groupby(['Games', 'ID']).Medal.count().reset_index()
athlete_total_df = all_df[['Games', 'Year', 'ID', 'Sex', 'Age', 'BMI', 'Season', 'NOC', 'Winner']]
athlete_total_df = athlete_total_df.drop_duplicates()
athlete_total_df = athlete_total_df.merge(athlete_events, how='outer') \
                                    .merge(athlete_medals, how='outer')

# Country totals (Games, Year, #Athletes, #Medals, #Male, #Female, #Events, Host)
noc_total_ath = all_df.groupby(['Games', 'NOC']).ID.count().reset_index()
noc_total_ath.columns = ['Games', 'NOC', 'Total_ID'] 
noc_athletes = all_df.groupby(['Games', 'NOC']).ID.nunique().reset_index()
noc_athletes.columns = ['Games', 'NOC', 'Unique_ID'] 
noc_events = all_df.groupby(['Games', 'NOC']).Event.nunique().reset_index()
noc_medals = all_df.groupby(['Games', 'NOC']).Medal.count().reset_index()
noc_male = all_df[all_df['Sex'] == 'M'].groupby(['Games', 'NOC']).ID.nunique().reset_index()
noc_male.columns = ['Games', 'NOC', 'Male']    
noc_female = all_df[all_df['Sex'] == 'F'].groupby(['Games', 'NOC']).ID.nunique().reset_index()
noc_female.columns = ['Games', 'NOC', 'Female']

noc_total_df = all_df[['Games', 'Year', 'Season', 'NOC', 'Country', 'Host_Country']]
noc_total_df = noc_total_df.drop_duplicates()
noc_total_df['Host']  = noc_total_df['Country'] == noc_total_df['Host_Country']
noc_total_df = noc_total_df.merge(noc_total_ath, how='outer') \
                            .merge(noc_athletes, how='outer') \
                            .merge(noc_events, how='outer') \
                            .merge(noc_medals, how='outer') \
                            .merge(noc_male, how='outer') \
                            .merge(noc_female, how='outer')


######## THE ATHLETE #######
winter_athlete = athlete_total_df[athlete_total_df['Season'] == 'Winter']
summer_athlete = athlete_total_df[athlete_total_df['Season'] == 'Summer']
male_athlete = athlete_total_df[athlete_total_df['Sex'] == 'M']
female_athlete = athlete_total_df[athlete_total_df['Sex'] == 'F']
medal_athlete = athlete_total_df[athlete_total_df['Winner']]
non_medal_athlete = athlete_total_df[athlete_total_df['Winner'] == False]

athlete_var_list = ['Age', 'BMI', 'Event', 'Medal']

## DISTRIBUTION

# Athlete - Histogram (no hues)
# plt.figure()
# plt.gcf().suptitle('Overall Distribution of Athletes')
# plot = [1,4,0]
# for var in athlete_var_list:
#     plot[2] += 1
#     plt.subplot(plot[0], plot[1], plot[2])    
#     sns.distplot(athlete_total_df[var], kde=False)
#     plt.title('Distribution of {var}'.format(var=var))
# plt.show()


# plt.figure()
# athlete_var_list = ['Age', 'BMI', 'Medal', 'Event']
# plt.gcf().suptitle('Overall Distribution of the Athletes')
# plot = [2,4,0]
# for var in athlete_var_list:
#     plot[2] += 1
#     plt.subplot(plot[0], plot[1], plot[2])    
#     sns.distplot(winter_athlete[var], kde=False, hist_kws={'histtype':'step'}, label='Winter')
#     plt.title('Distribution of {var} in Winter'.format(var=var))
#     plt.legend()
# for var in athlete_var_list:
#     plot[2] += 1
#     plt.subplot(plot[0], plot[1], plot[2])    
#     sns.distplot(summer_athlete[var], kde=False, hist_kws={'histtype':'step'}, label='Summer')
#     plt.title('Distribution of {var} in Summer'.format(var=var))
#     plt.legend()
# plt.show()







# plt.figure()
# athlete_var_list = ['Age', 'BMI', 'Medal', 'Event']
# plt.gcf().suptitle('Distribution of Athletes Attributes Overall')
# plot = [1,4,0]
# for var in athlete_var_list:
#     plot_min = int(athlete_total_df[var].min())
#     plot_max = int(athlete_total_df[var].max())
#     bins = int(plot_max - plot_min)
#     plot[2] += 1
#     ax = plt.subplot(plot[0], plot[1], plot[2])
#     plt.hist(athlete_total_df[var], range=(plot_min, plot_max), bins=bins)
#     ax.set_xticks(range(0, plot_max, 5))
#     plt.xlabel(var)
#     plt.title('Distribution of {var}'.format(var=var)) 
# plt.show()





# Comparison









#### HOW HAVE THINGS CHANGED??

# Athlete - Boxplot - Change over years
# var_list = ['Age', 'BMI']
# hue_list = [None, 'Sex', 'Winner']
# for var in var_list:
#     for hue in hue_list:
#         plt.figure()
#         plot = [2,1,0]
#         plot[2] += 1
#         plt.subplot(plot[0], plot[1], plot[2])
#         sns.boxplot(data=winter_athlete, x='Year', y=var, hue=hue)
#         plt.title('{var} of Winter Athletes'.format(var=var))
#         plot[2] += 1
#         plt.subplot(plot[0], plot[1], plot[2])
#         sns.boxplot(data=summer_athlete, x='Year', y=var, hue=hue)
#         plt.title('{var} of Summer Athletes'.format(var=var))
#         plt.show()

# Athlete - Barplot - Change over years
# var_list = ['Age', 'BMI']
# Don't include medal or events (plots are useless)
# hue_list = [None, 'Sex', 'Winner']
# for var in var_list:
#     for hue in hue_list:
#         plt.figure()
#         plot = [2,1,0]
#         plot[2] += 1
#         plt.subplot(plot[0], plot[1], plot[2])
#         sns.barplot(data=winter_athlete, x='Year', y=var, hue=hue)
#         plt.title('{var} of Winter Athletes'.format(var=var))
#         plot[2] += 1
#         plt.subplot(plot[0], plot[1], plot[2])
#         sns.barplot(data=summer_athlete, x='Year', y=var, hue=hue)
#         plt.title('{var} of Summer Athletes'.format(var=var))
#         plt.show()




######## THE GAMES #######
winter_games = games_total_df[games_total_df['Season'] == 'Winter']
summer_games = games_total_df[games_total_df['Season'] == 'Summer']

games_var_list = ['Unique_ID', 'Total_ID', 'Male', 'Female', 'NOC', 'Event', 'Sport', 'Medal']




# Games - Histogram (no hues)
# plt.figure()
# plt.gcf().suptitle('Overall Distribution of the Games')
# plot = [2,4,0]
# for var in games_var_list:
#     plot[2] += 1
#     plt.subplot(plot[0], plot[1], plot[2])    
#     sns.distplot(games_total_df[var], kde=False)
#     plt.title('Distribution of {var}'.format(var=var))
# plt.show()

# Games - Histogram (Season hue)
# plt.figure()
# plt.gcf().suptitle('Overall Distribution of the Games')
# plot = [2,4,0]
# for var in games_var_list:
#     plot[2] += 1
#     plt.subplot(plot[0], plot[1], plot[2])    
#     sns.distplot(winter_games[var], kde=True)
#     sns.distplot(summer_games[var], kde=True)
#     plt.title('Distribution of {var} by Season'.format(var=var))
# plt.show()

# Games - Pairwise (Season)
# games_var_list = ['Male', 'Female', 'Unique_ID', 'Total_ID', 'Medal']
# sns.pairplot(games_total_df, kind='reg', diag_kind='kde', vars=games_var_list)
# plt.subplots_adjust(top=0.98)
# plt.gcf().suptitle('Games Distribution Overall')
# plt.show()

# sns.pairplot(games_total_df, hue='Season', kind='reg', diag_kind='kde', vars=games_var_list)
# plt.subplots_adjust(top=1)
# plt.gcf().suptitle('Games Distribution by Season')
# plt.show()



#### HOW HAVE THINGS CHANGED??

# var_list = ['Unique_ID', 'Total_ID', 'Male', 'Female', 'NOC', 'Event', 'Sport', 'Medal']
# for var in var_list:
#     plt.figure()
#     plot = [2,1,0]
#     plot[2] += 1
#     plt.subplot(plot[0], plot[1], plot[2])
#     sns.barplot(data=games_total_df, x='Year', y=var, hue='Season')
#     plt.title('{var} of Winter Athletes'.format(var=var))
#     plot[2] += 1
#     plt.subplot(plot[0], plot[1], plot[2])
#     sns.barplot(data=games_total_df, x='Year', y=var, hue='Season')
#     plt.title('{var} of Summer Athletes'.format(var=var))
#     plt.show()

# TODO
# Unique v. Total 
# Male v. Female
# Events v. Sports v. Medal v. Countries


athlete_hue_list = ['Season', 'Sex', 'Winner']
# for hue in athlete_hue_list:
#     for var in athlete_var_list:
#         plt.subplot(2,1,1)
#         sns.barplot(x='Year' , y=var, hue=hue, palette=athlete_colors, data=athlete_total_df)
#         plt.subplot(2,1,2)
#         sns.boxplot(x='Year', y=var, hue=hue, palette=athlete_colors, data=athlete_total_df)        
#         plt.gcf().suptitle('Change in {var} over the years by {hue}'.format(var=var, hue=hue))
#         plt.show()


# sns.set_style(style='whitegrid')


#from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x =athlete_total_df.Age
# y =athlete_total_df.BMI
# z =athlete_total_df.Medal
# ax.scatter(x, y, z, c='r', marker='o')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()


best = ['CHN', 'USA', 'FIN', 'GER', 'RUS', 'GBR', 'FRA', 'ITA', 'SWE', 'NOR', 'HUN', 'AUS']

#best_countries = noc_total_df[noc_total_df['NOC'].isin(best)]
#noc_total_df['NOC'] = np.where(~noc_total_df['NOC'].isin(best), 'ALL'), noc_total_df['NOC'])

countries_all = all_df.copy(deep=True)
countries_all.loc[~countries_all['NOC'].isin(best), 'NOC'] = 'ALL'
best_all_countries = countries_all[countries_all['NOC'].isin(best)]

countries_count = noc_total_df.copy(deep=True)
countries_count.loc[~countries_count['NOC'].isin(best), 'NOC'] = 'ALL'
best_count_countries = countries_count[countries_count['NOC'].isin(best)]

# Boxplot of difference between best and all
# sns.boxplot(x='NOC', y='Age', data=countries_all)
# plt.show()
# sns.catplot(x='NOC', y='Age', data=best_countries) # jitter=False
# plt.show()
sns.violinplot(x='NOC', y='Unique_ID', hue='Season', split=True, data=best_count_countries, palette=athlete_colors)
plt.show()


sns.violinplot(x='Year', y='Age', hue='Sex', split=True, data=athlete_total_df, palette=athlete_colors)
plt.show()


sns.scatterplot(x='Total_ID', y='Medal', hue='NOC', data=best_count_countries)
plt.show()

# sns.violinplot(x='NOC', y='Age', data=countries, inner=None) # split=True, hue='Gender'
# sns.swarmplot(x='NOC', y='Age', data=countries, color='k', alpha=0.7)
# sns.kdeplot(df1, df2)
# sns.jointplot()
# plt.ylim(0, None)
# plt.xlim(0, None)
# sns.set_style('whitegrid')
# plt.legend(bbox_to_anchor=(1, 1), loc=2)
























# plt.figure()
# row = 4
# col = 2
# pos = 1

# var = 'Age'
# plot_min = int(athlete_total_df[var].min())
# plot_max = int(athlete_total_df[var].max())
# bins = int(plot_max - plot_min)

# # All
# plt.subplot(row,col,pos)  
# plt.hist(athlete_total_df[var], range=(plot_min, plot_max), bins=bins, label='All', histtype='step')
# pos +=1
# plt.subplot(row, col, pos)
# sns.kdeplot(athlete_total_df[var], shade=True, label='All')
# pos +=1
# # Season
# ax = plt.subplot(row,col,pos) 
# olympic_hist(var, winter_athlete, summer_athlete, ax, 'Winter', 'Summer', plot_min, plot_max, bins)
# pos +=1
# plt.subplot(row, col, pos)
# sns.kdeplot(winter_athlete[var], shade=True, label='Winter')
# sns.kdeplot(summer_athlete[var], shade=True, label='Summer')
# plt.show()
# pos +=1
# # Gender
# ax = plt.subplot(row,col,pos) 
# olympic_hist(var, female_athlete, male_athlete, ax, 'Female', 'Male', plot_min, plot_max, bins)
# pos +=1
# plt.subplot(row, col, pos)
# sns.kdeplot(female_athlete[var], shade=True, label='Female')
# sns.kdeplot(male_athlete[var], shade=True, label='Male')
# pos +=1
# ax = plt.subplot(row,col,pos) 
# olympic_hist(var, medal_athlete, non_medal_athlete, ax, 'Medal', 'Non-Medal', plot_min, plot_max, bins)
# pos +=1
# plt.subplot(row, col, pos)
# sns.kdeplot(medal_athlete[var], shade=True, label='Medal')
# sns.kdeplot(non_medal_athlete[var], shade=True, label='Non-Medal')
# plt.show()


#olympic_hist(var, df1, df2, ax, label1, label2, plot_min, plot_max, bins)


  


# pd.DataFrame.hist(athlete_total_df, column='Age', by='Season')
# plt.show()




# The Athlete
# plot = [4,5,0]
# df = athlete_total_df
# plot[2] = hist_kde(df, 'Age', plot)
# plot[2] = hist_kde(df,'BMI', plot)
# plot[2] = hist_kde(df, 'Event', plot)
# hist_kde(df, 'Medal', plot)
# plt.show()

# # The Games
# plot = [8,5,0]
# df = games_total_df
# plot[2] = hist_kde(df, 'Event', plot)
# plot[2] = hist_kde(df, 'Sport', plot)
# plot[2] = hist_kde(df,'Medal', plot)
# plot[2] = hist_kde(df,'Host_Medal', plot)
# plot[2] = hist_kde(df, 'NOC', plot)
# plot[2] = hist_kde(df, 'Unique_ID', plot)
# plot[2] = hist_kde(df, 'Total_ID', plot)
# plot[2] = hist_kde(df, 'Male', plot)
# plot[2] = hist_kde(df, 'Female', plot)
# plt.show()

# # The Countries
# plot = [6,5,0]
# df = games_total_df
# plot[2] = hist_kde(df, 'Event', plot)
# plot[2] = hist_kde(df,'Medal', plot)
# plot[2] = hist_kde(df,'Host_Medal', plot)
# plot[2] = hist_kde(df, 'NOC', plot)
# plot[2] = hist_kde(df, 'Unique_ID', plot)
# plot[2] = hist_kde(df, 'Total_ID', plot)
# plt.show()


# Medals by Athlete
# row=3
# col=3
# pos=1

# plt.subplot(row, col, pos)
# sns.scatterplot(x='Medal' , y='Event', data=athlete_total_df)
# pos+=1
# plt.subplot(row, col, pos)
# sns.scatterplot(x='Medal' , y='Age', data=athlete_total_df)
# pos+=1
# plt.subplot(row, col, pos)
# sns.scatterplot(x='Medal' , y='BMI', data=athlete_total_df)
# pos+=1

# hue_list = ['Sex', 'Season']
# for hue in hue_list:
#     plt.subplot(row, col, pos)
#     sns.scatterplot(x='Medal' , y='Event', hue=hue, data=athlete_total_df)
#     pos+=1
#     plt.subplot(row, col, pos)
#     sns.scatterplot(x='Medal' , y='Age', hue=hue, data=athlete_total_df)
#     pos+=1
#     plt.subplot(row, col, pos)
#     sns.scatterplot(x='Medal' , y='BMI', hue=hue, data=athlete_total_df)
#     pos+=1
# plt.show()



# Medals by Host Country
winter_games = games_total_df[games_total_df['Season'] == 'Winter']
summer_games = games_total_df[games_total_df['Season'] == 'Summer']


# plt.figure()
# plt.gcf().suptitle('#Medals v. #Host Medals by Games')
# row, col, pos = 2,2,1
# plt.subplot(row, col, pos)
# sns.scatterplot(x='Medal', y='Host_Medal', data=games_total_df, ci='sd')
# pos+=1
# plt.subplot(row, col, pos)
# sns.scatterplot(x='Medal', y='Host_Medal', hue='Season', data=games_total_df, ci='sd', palette=athlete_colors)
# plt.gcf().suptitle('#Medals_Perc v. #Host Medals Perc by Games')
# pos+=1
# plt.subplot(row, col, pos)
# sns.scatterplot(x='Visitor_Medal_Perc', y='Host_Medal_Perc', data=winter_games, ci='sd', palette=athlete_colors)
# pos+=1
# plt.subplot(row, col, pos)
# sns.scatterplot(x='Visitor_Medal_Perc', y='Host_Medal_Perc', hue='Season', data=summer_games, ci='sd', palette=athlete_colors)
# plt.show()






# qqplot(games_total_df, 
#             x='Visitor_Medal_Perc', 
#             y='Host_Medal_Perc', 
#             hue='Season', 
#             height=3, 
#             aspect=1.5, 
#             palette=athlete_colors,
#             display_kws={"identity":False,"fit":True,"reg":True,"ci":0.025})
# plt.show()



# Medals by Country
# winter_noc = noc_total_df[noc_total_df['Season'] == 'Winter']
# summer_noc = noc_total_df[noc_total_df['Season'] == 'Summer']

# plt.figure()
# plt.gcf().suptitle('#Medals v. ??? by Country')
# row=5
# col=3
# pos=0
# y_list = ['Unique_ID', 'Total_ID', 'Event', 'Male', 'Female']
# for y in y_list:
#     pos+=1
#     plt.subplot(row, col, pos)
#     sns.scatterplot(x='Medal', y=y, hue='Season', data=noc_total_df)
#     pos+=1
#     plt.subplot(row, col, pos)
#     sns.scatterplot(x='Medal', y=y, data=summer_noc)
#     pos+=1
#     plt.subplot(row, col, pos)
#     sns.scatterplot(x='Medal', y=y, data=winter_noc)
# plt.show()