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
    'F:/TEAN/Portfolio/olympics/data/summer_data.csv', index_col=0).reset_index(drop = True)




# Games totals (Games, Year, #Athletes, #Medals, #Male, #Female, #Events)
# Summer only - Replace Games in groupby with Year
games_total_ath = all_df.groupby(['Year']).ID.count().reset_index()
games_total_ath.columns = ['Year', 'Total_ID'] 
games_athletes = all_df.groupby(['Year']).ID.nunique().reset_index()
games_athletes.columns = ['Year', 'Unique_ID'] 
games_events = all_df.groupby(['Year']).Event.nunique().reset_index()
games_sports = all_df.groupby(['Year']).Sport.nunique().reset_index()
games_medals = all_df.groupby(['Year']).Medal.count().reset_index()
games_countries = all_df.groupby(['Year']).NOC.nunique().reset_index()
games_male = all_df[all_df['Sex'] == 'M'].groupby('Year').ID.nunique().reset_index()
games_male.columns = ['Year', 'Male']    
games_female = all_df[all_df['Sex'] == 'F'].groupby('Year').ID.nunique().reset_index()
games_female.columns = ['Year', 'Female']
games_host = all_df[all_df['NOC'] == all_df['Host_NOC']].groupby('Year').Medal.count().reset_index()
games_host.columns = ['Year', 'Host_Medal']
games_visitor = all_df[all_df['NOC'] != all_df['Host_NOC']].groupby('Year').Medal.count().reset_index()
games_visitor.columns = ['Year', 'Visitor_Medal']
games_total_df = all_df[['Year', 'Host_NOC']] #Remove season, games
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
print(games_total_df)


# Create host percentage and regular percentage df
# Summer only - Replace Games in groupby with Year
medals_all = all_df.groupby(['Year', 'NOC', 'Country']).Medal.count().reset_index() #remove season
medals_all.columns=['Year', 'NOC', 'Country', 'Total_Medal'] #remove season
medals_all = medals_all.merge(games_medals, how='outer')
medals_all['Medal_Perc'] = round((medals_all.Total_Medal / medals_all.Medal)*100, 2)
medals_all = round(medals_all.groupby(['NOC', 'Country']).Medal_Perc.mean(),2).reset_index() #remove season
medals_all.columns=['NOC', 'Country', 'Medal_Perc'] # remove season
host_medals = games_total_df[['Year', 'Host_NOC', 'Host_Medal_Perc']] #remove season, games
host_medals.columns=['Year', 'NOC', 'Host_Medal_Perc'] #remove season, games
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

# winter_all['Medal_Perc'] = round((winter_all.Medal / games_total_df.Total_Medal)*100, 2)
# print(winter_all)


# winter_all['Games_Total'] = winter_all.merge(games_total_df.Medal)
# winter_all['Medal_Perc'] = round((winter_all.Medal / 'Games_Total' )*100, 2)
# non_host2 = non_host.groupby(['NOC', 'Season']).Medal.mean()
# host2['Medal'] = host2.merge(non_host2)
# print(host2)



# Athlete Totals (Games, Year, ID, Sex, Age, BMI, Season, NOC, #Events, #Medals)
# Summer only - Replace Games in groupby with Year
athlete_events = all_df.groupby(['Year', 'ID']).Event.count().reset_index()
athlete_medals = all_df.groupby(['Year', 'ID']).Medal.count().reset_index()
athlete_total_df = all_df[['Year', 'ID', 'Sex', 'Age', 'BMI', 'NOC', 'Winner']] #Remove season, games
athlete_total_df = athlete_total_df.drop_duplicates()
athlete_total_df = athlete_total_df.merge(athlete_events, how='outer') \
                                    .merge(athlete_medals, how='outer')

# Country totals (Games, Year, #Athletes, #Medals, #Male, #Female, #Events, Host)
noc_total_ath = all_df.groupby(['Year', 'NOC']).ID.count().reset_index()
noc_total_ath.columns = ['Year', 'NOC', 'Total_ID'] 
noc_athletes = all_df.groupby(['Year', 'NOC']).ID.nunique().reset_index()
noc_athletes.columns = ['Year', 'NOC', 'Unique_ID'] 
noc_events = all_df.groupby(['Year', 'NOC']).Event.nunique().reset_index()
noc_medals = all_df.groupby(['Year', 'NOC']).Medal.count().reset_index()
noc_male = all_df[all_df['Sex'] == 'M'].groupby(['Year', 'NOC']).ID.nunique().reset_index()
noc_male.columns = ['Year', 'NOC', 'Male']    
noc_female = all_df[all_df['Sex'] == 'F'].groupby(['Year', 'NOC']).ID.nunique().reset_index()
noc_female.columns = ['Year', 'NOC', 'Female']
games_medals = games_total_df[['Year', 'Medal']]
games_medals.columns = ['Year', 'Games_Medals']
games_athletes = games_total_df[['Year', 'Total_ID']]
games_athletes.columns = ['Year', 'Games_Athletes']

noc_total_df = all_df[['Year', 'NOC', 'Country', 'Host_Country']] #remove season, games
noc_total_df = noc_total_df.drop_duplicates()
noc_total_df['Host']  = noc_total_df['Country'] == noc_total_df['Host_Country']
noc_total_df = noc_total_df.merge(noc_total_ath, how='outer') \
                            .merge(noc_athletes, how='outer') \
                            .merge(noc_events, how='outer') \
                            .merge(noc_medals, how='outer') \
                            .merge(noc_male, how='outer') \
                            .merge(noc_female, how='outer') \
                            .merge(games_medals, how='outer') \
                            .merge(games_athletes, how='outer')
noc_total_df['Unique_Perc'] = round((noc_total_df.Total_ID / noc_total_df.Unique_ID), 2)
noc_total_df['Medal_Perc'] = round((noc_total_df.Medal / noc_total_df.Total_ID), 2)
noc_total_df['Games_Medal_Perc'] = round((noc_total_df.Medal / noc_total_df.Games_Medals), 2)
noc_total_df['Games_Athlete_Perc'] = round((noc_total_df.Total_ID / noc_total_df.Games_Athletes), 2)








######## THE ATHLETE #######
#winter_athlete = athlete_total_df[athlete_total_df['Season'] == 'Winter']
#summer_athlete = athlete_total_df[athlete_total_df['Season'] == 'Summer']
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
#winter_games = games_total_df[games_total_df['Season'] == 'Winter']
#summer_games = games_total_df[games_total_df['Season'] == 'Summer']

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


athlete_hue_list = ['Sex', 'Winner']
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





# Boxplot of difference between best and all
# sns.boxplot(x='NOC', y='Age', data=countries_all)
# plt.show()
# sns.catplot(x='NOC', y='Age', data=best_countries) # jitter=False
# plt.show()


# # Best NOC number of athletes comparison
# sns.violinplot(x='NOC', y='Unique_ID', hue='Season', split=True, data=best_count_countries, palette=athlete_colors)
# plt.show()
# sns.scatterplot(x='Total_ID', y='Medal', hue='NOC', data=best_count_countries)
# plt.show()


# sns.violinplot(x='Year', y='Age', hue='Sex', split=True, data=athlete_total_df, palette=athlete_colors)
# plt.show()

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
#print(sns.color_palette(n_colors=10))
noc_colors = sns.color_palette("Paired", n_colors=11)
noc_colors[-1] = (0.0, 0.0, 0.0)

noc_all_before_1950 = noc_total_df[noc_total_df['Year']<1950]
noc_total_1950 = noc_total_df[noc_total_df['Year']>1950]
#winter_noc = noc_total_1950[noc_total_1950['Season'] == 'Winter']
summer_noc = noc_total_1950

# Count who has the most medals since 1950
noc_top_medals_summer = noc_total_1950.groupby(['NOC'], as_index=False)['Medal'].sum().sort_values(by='Medal', ascending=False).head(20)
#noc_top_medals_summer = noc_total_1950[noc_total_1950['Season'] == 'Summer'].groupby(['NOC'], as_index=False)['Medal'].sum().sort_values(by='Medal', ascending=False).head(20)
#noc_top_medals_winter = noc_total_1950[noc_total_1950['Season'] == 'Winter'].groupby(['NOC'], as_index=False)['Medal'].sum().sort_values(by='Medal', ascending=False).head(20)
top_summer_25 = noc_top_medals_summer.NOC.tolist()
#top_winter_25 = noc_top_medals_winter.NOC.tolist()
top_summer_10 = noc_top_medals_summer.NOC[:10].tolist()
#top_winter_10 = noc_top_medals_winter.NOC[:10].tolist()

summer_noc['Top_20'] = summer_noc['NOC'].isin(top_summer_25)

noc_total_1950 = noc_total_1950.drop(["Country", "Host"], axis=1).reset_index()
summer_1950_10 = summer_noc[(summer_noc['NOC'].isin(top_summer_10))].reset_index()
summer_1950_all = summer_noc[(summer_noc['NOC'].isin(top_summer_25)) &  (~summer_noc['NOC'].isin(top_summer_10))].reset_index()
summer_1950_all_count = summer_1950_all.groupby(['Year']).median().reset_index()
summer_1950_all_count['NOC'] = 'ALL'
summer_1950 = pd.merge(summer_1950_10, summer_1950_all_count, how='outer')

# winter_1950_10 = winter_noc[(winter_noc['NOC'].isin(top_winter_10))].reset_index()
# winter_1950_all = winter_noc[(winter_noc['NOC'].isin(top_winter_25)) &  (~winter_noc['NOC'].isin(top_winter_10))].reset_index()
# winter_1950_all_count = winter_1950_all.groupby(['Games']).median().reset_index()
# winter_1950_all_count['NOC'] = 'ALL'
# winter_1950 = pd.merge(winter_1950_10, winter_1950_all_count, how='outer')



top_summer_order = top_summer_10.copy()
top_summer_order.append('ALL')
# top_winter_order = top_winter_10.copy()
# top_winter_order.append('ALL')







# winter_1950 = noc_total_1950[(noc_total_1950['Season']=='Winter') & (noc_total_1950['NOC'].isin(top_winter_25))].reset_index()
# winter_1950.loc[~winter_1950['NOC'].isin(top_winter_10), 'NOC'] = 'ALL'


# Top 10 countries for each season, rest to 'ALL
#top_summer = ['USA', 'RUS', 'GER', 'AUS', 'GBR', 'CHN', 'ITA', 'JPN', 'FRA', 'HUN', 'ALL']
#summer_1950 = noc_total_1950[(noc_total_1950['Season']=='Summer')].reset_index()
# summer_1950.loc[~summer_1950['NOC'].isin(top_summer), 'NOC'] = 'ALL'


#top_winter = ['RUS', 'GER', 'USA', 'CAN', 'SWE', 'FIN', 'NOR', 'AUT', 'CZE', 'SUI', 'ALL']  
#winter_1950 = noc_total_1950[(noc_total_1950['Season']=='Winter')].reset_index()
#winter_1950.loc[~winter_1950['NOC'].isin(top_winter), 'NOC'] = 'ALL'

# Top 10 countries for each only
#winter_1950_best = winter_1950[winter_1950['NOC'] != 'ALL']
#summer_1950_best = summer_1950[summer_1950['NOC'] != 'ALL']
#summer_1950_all = summer_1950[summer_1950['NOC'] == 'ALL']
#winter_1950_all = winter_1950[winter_1950['NOC'] == 'ALL']

plt.subplot(2,1,1)
sns.swarmplot(data=summer_1950, x='NOC', y='Medal', order=top_summer_order, palette=noc_colors)
plt.title('Athlete Success in Summer')
plt.subplot(2,1,2)
sns.swarmplot(data=summer_1950, x='NOC', y='Medal_Perc', order=top_summer_order, palette=noc_colors)
plt.show()


plt.subplot(3,1,1)
sns.swarmplot(data=summer_1950, x='NOC', y='Event', order=top_summer_order, palette=noc_colors)
plt.subplot(3,1,2)
sns.swarmplot(data=summer_1950, x='NOC', y='Unique_ID', order=top_summer_order, palette=noc_colors)
plt.subplot(3,1,3)
sns.swarmplot(data=summer_1950, x='NOC', y='Unique_Perc', order=top_summer_order, palette=noc_colors)
plt.show()





plt.subplot(1,2,1)
corr = summer_1950_10[['Medal_Perc', 'Medal', 'Unique_ID', 'Total_ID', 'Event', 'Male', 'Female', 'Unique_Perc', 'Games_Medal_Perc', 'Games_Athlete_Perc']].corr()
sns.heatmap(corr, annot=True) #linewidths=0.5, cmap='coolwarm'
plt.title("Top 10 Only")
plt.subplot(1,2,2)
corr = summer_noc[~summer_noc['NOC'].isin(top_summer_10)][['Medal_Perc', 'Medal', 'Unique_ID', 'Total_ID', 'Event', 'Male', 'Female', 'Unique_Perc', 'Games_Medal_Perc', 'Games_Athlete_Perc']].corr()
sns.heatmap(corr, annot=True) #linewidths=0.5, cmap='coolwarm'
plt.title("Not top 10 in Summer")
plt.show()

plt.subplot(1,2,1)
corr = summer_1950[['Medal_Perc', 'Medal', 'Unique_ID', 'Total_ID', 'Event', 'Male', 'Female', 'Unique_Perc', 'Games_Medal_Perc', 'Games_Athlete_Perc']].corr()
sns.heatmap(corr, annot=True) #linewidths=0.5, cmap='coolwarm'
plt.title("Top 20 in Summer")
plt.subplot(1,2,2)
corr = summer_noc[~summer_noc['NOC'].isin(top_summer_25)][['Medal_Perc', 'Medal', 'Unique_ID', 'Total_ID', 'Event', 'Male', 'Female', 'Unique_Perc', 'Games_Medal_Perc', 'Games_Athlete_Perc']].corr()
sns.heatmap(corr, annot=True) #linewidths=0.5, cmap='coolwarm'
plt.title("All in Summer")
plt.title("All")
plt.show()

sns.pairplot(summer_1950_10, kind='scatter', diag_kind='kde', vars=['Medal', 'Unique_ID', 'Total_ID', 'Event', 'Unique_Perc', 'Games_Medal_Perc', 'Games_Athlete_Perc'])
plt.subplots_adjust(top=0.95)
plt.gcf().suptitle('Overall Relationship of NOC Variables')
plt.show()

sns.scatterplot(data=summer_noc, x='Games_Medal_Perc', y='Games_Athlete_Perc')
print(summer_noc)
plt.show()

plt.subplot(2,1,1)
sns.barplot(data=summer_noc, x='Year', y='Games_Medal_Perc', hue='Top_20')
plt.subplot(2,1,2)
sns.barplot(data=summer_noc, x='Year', y='Games_Athlete_Perc', hue='Top_20')
plt.show()
# plt.subplot(4,1,1)
# sns.barplot(data=summer_noc, x='Year', y='Unique_ID', hue='Top_20')
# plt.title('Number of athletes')
# plt.subplot(4,1,2)
# sns.barplot(data=summer_noc, x='Year', y='Total_ID', hue='Top_20')
# plt.title('Number of entries')
# plt.subplot(4,1,3)
# sns.barplot(data=summer_noc, x='Year', y='Unique_Perc', hue='Top_20')
# plt.title('Number of events per athlete')
# plt.subplot(4,1,4)
# sns.swarmplot(data=summer_1950, x='NOC', y='Unique_Perc', order=top_summer_order, palette=noc_colors)
# plt.title('#Medals per Event since 1950 in Summer')
# plt.show()


# plt.subplot(2,1,1)
# sns.swarmplot(data=summer_1950, x='NOC', y='Unique_Perc', hue_order=top_summer)
# plt.subplot(2,1,2)
# sns.swarmplot(data=summer_1950, x='NOC', y='Medal_Perc', hue_order=top_summer)
# plt.title('Athlete Success in Summer (All)')
# plt.show()

# best = ['CHN', 'USA', 'FIN', 'GER', 'RUS', 'GBR', 'FRA', 'ITA', 'SWE', 'NOR', 'HUN', 'AUS']
# # All of the rows from all_df for just the best countries, the rest to 'ALL
# countries_all = all_df.copy(deep=True)
# countries_all.loc[~countries_all['NOC'].isin(best), 'NOC'] = 'ALL'
# best_all_countries = countries_all[countries_all['NOC'].isin(best)]
# # The noc_total_df table for NOC countries and all else to 'ALL'
# countries_count = noc_total_1950.copy(deep=True)
# countries_count.loc[~countries_count['NOC'].isin(best), 'NOC'] = 'ALL'
# best_count_countries = countries_count[countries_count['NOC'].isin(best)]
# best_count = noc_total_1950[noc_total_1950['NOC'].isin(best)].sort_values('Unique_Perc')





# plt.subplot(2,1,1)
# sns.scatterplot(data=summer_1950, x='Unique_Perc', y='Medal', palette=noc_colors)
# plt.title('Best 25 since 1950 in Summer')
# plt.subplot(2,1,2)
# sns.scatterplot(data=winter_1950, x='Unique_Perc', y='Medal', palette=noc_colors)
# plt.title('Best 25 since 1950 in Winter')
# plt.show()


#not_best = summer_1950[summer_1950['NOC']=='ALL']
#summer_1950_not = not_best.groupby(['NOC', 'Games']).median()
#print(summer_1950)
best = summer_1950[summer_1950['NOC'] !='ALL']
summer_1950_best = best.groupby(['NOC'], as_index=False).median()
print(summer_1950_10)
print(summer_1950_all_count)
print(summer_1950_best)


row,col,pos = 5,1,1
plt.subplot(row,col,pos)
sns.scatterplot(data=summer_noc, x='Medal', y='Unique_Perc')
pos+=1
plt.subplot(row,col,pos)
sns.scatterplot(data=summer_noc, x='Medal', y='Total_ID')
pos+=1
plt.subplot(row,col,pos)
sns.scatterplot(data=summer_noc, x='Medal', y='Unique_ID')
pos+=1
plt.subplot(row,col,pos)
sns.scatterplot(data=summer_noc, x='Medal', y='Event')
pos+=1
plt.subplot(row,col,pos)
sns.scatterplot(data=summer_noc, x='Medal_Perc', y='Medal')
pos+=1
plt.show()


plt.subplot(2,1,1)
sns.scatterplot(data=summer_1950_10, y='Medal_Perc', x='Unique_Perc', hue='NOC', hue_order=top_summer_order[:-1])
sns.scatterplot(data=summer_1950_all_count, y='Medal_Perc', x='Unique_Perc', color='black', marker='D')
plt.title('Best 25 in Summer split up')

plt.subplot(2,1,2)
sns.scatterplot(data=summer_1950_10, y='Medal', x='Unique_Perc', hue='NOC', hue_order=top_summer_order[:-1])
sns.scatterplot(data=summer_1950_all_count, y='Medal', x='Unique_Perc', color='black', marker='D')
plt.title('Best 25 in Summer split up')
plt.show()

print(summer_1950)
sns.scatterplot(data=summer_1950, x='Medal', y='Unique_Perc', hue='NOC', size='Year', hue_order=top_summer_order, palette=noc_colors)
plt.title('All on year')
plt.show()

plt.subplot(2,1,1)
sns.scatterplot(data=summer_1950, x='Medal', y='Unique_ID', hue='NOC', hue_order=top_summer_order, palette=noc_colors)
plt.subplot(2,1,2)
sns.scatterplot(data=summer_1950, x='Medal_Perc', y='Unique_ID', hue='NOC', hue_order=top_summer_order, palette=noc_colors)
plt.show()

plt.subplot(2,1,1)
sns.scatterplot(data=summer_1950, x='Medal', y='Event', hue='NOC', hue_order=top_summer_order, palette=noc_colors)
plt.subplot(2,1,2)
sns.scatterplot(data=summer_1950, x='Medal_Perc', y='Event', hue='NOC', hue_order=top_summer_order, palette=noc_colors)
plt.show()


def scatter_text(x, y, text_column, data, title, xlabel, ylabel):
    """Scatter plot with country codes on the x y coordinates
       Based on this answer: https://stackoverflow.com/a/54789170/2641825"""
    # Create the scatter plot
    p1 = sns.scatterplot(x, y, data=data, size = 8, legend=False)
    # Add text besides each point
    for line in range(0,data.shape[0]):
         p1.text(data[x][line]+0.01, data[y][line], 
                 data[text_column][line], horizontalalignment='left', 
                 size='medium', color='black', weight='semibold')
    # Set title and axis labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return p1

plt.figure(figsize=(20,10))
scatter_text('Unique_Perc', 'Medal_Perc', 'Year',
             data = summer_1950, 
             title = 'Iris sepals', 
             xlabel = 'Sepal Length (cm)',
             ylabel = 'Sepal Width (cm)')
plt.show()



sns.pairplot(summer_1950_10,hue='NOC', diag_kind='hist')
plt.show()


sns.pairplot(noc_total_df, kind='scatter', diag_kind='hist', vars=['Unique_ID', 'Total_ID', 'Event', 'Medal'], plot_kws={'color':'C7'}, diag_kws={'color':'C7'})
plt.show()


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













# 1. Remove art competitions
art_df = all_df[all_df['Sport'] == 'Art Competitions']
all_df = all_df.drop(art_df.index)