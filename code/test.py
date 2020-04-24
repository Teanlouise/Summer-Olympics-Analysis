import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm  
import numpy as np
from plots import *



# Read in data and split between summer and winter
all_df = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/all_data.csv', index_col=0).reset_index(drop = True)

## Add winner 
all_df['Winner'] = all_df.Medal.notna()

# Add BMI columns [Weight (kg) / Height^2 (m)]
all_df['BMI'] = all_df.apply(lambda x: round(x.Weight/((x.Height/100)**2), 2), axis=1)


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
games_host = all_df[all_df['Country'] == all_df['Host_Country']].groupby('Games').Medal.count().reset_index()
games_host.columns = ['Games', 'Host_Medal']
games_visitor = all_df[all_df['Country'] != all_df['Host_Country']].groupby('Games').Medal.count().reset_index()
games_visitor.columns = ['Games', 'Visitor_Medal']


games_total_df = all_df[['Games', 'Year', 'Season']]
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
                                
print(games_total_df)

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

# Athlete - Pairwise
# hue_list = ['Season', 'Sex', 'Winner']
# for hue in hue_list: 
#     sns.pairplot(athlete_total_df, hue=hue, kind='scatter', diag_kind='hist', vars=var_list)
#     plt.subplots_adjust(top=0.9)
#     plt.gcf().suptitle('Athlete Distribution by {hue}'.format(hue=hue))
#     plt.show()



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


## DISTRIBUTION

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
plt.figure()
plt.gcf().suptitle('#Medals v. #Host Medals by Games')
row=2
col=2
pos=1
plt.subplot(row, col, pos)
sns.scatterplot(x='Visitor_Medal', y='Host_Medal', data=games_total_df)
plt.title('All')
pos+=1
plt.subplot(row, col, pos)
sns.scatterplot(x='Visitor_Medal', y='Host_Medal', hue='Season', data=games_total_df)
pos+=1
plt.subplot(row, col, pos)
sns.scatterplot(x='Visitor_Medal', y='Host_Medal', data=summer_games)
plt.title('Summer')
pos+=1
plt.subplot(row, col, pos)
sns.scatterplot(x='Visitor_Medal', y='Host_Medal',data=winter_games)
plt.title('Winter')
plt.show()


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