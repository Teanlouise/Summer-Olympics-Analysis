import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm  
import numpy as np
from plots import *
from statsmodels.graphics.gofplots import qqplot_2samples
import scipy.stats as stats
import scipy.optimize as opt

athlete_colors ={'Winter': 'C0', 'Summer':'C1', 'F': 'C6', 'M': 'C9', False: 'C4', True: 'C2'}

# Read in data and split between summer and winter
all_df = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/summer_data.csv', index_col=0).reset_index(drop = True)
athlete_total_df = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/athlete_total.csv', index_col=0)
games_total_df = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/games_total.csv', index_col=0)
noc_total_df = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/noc_total.csv', index_col=0)
host_difference = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/host_difference.csv', index_col=0)
noc_colors = sns.color_palette("Paired", n_colors=11)
noc_colors[-1] = (0.0, 0.0, 0.0)



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




top_20_all = top_10.merge(top_20, how='outer')

# from scipy.optimize import curve_fit
# from pylab import *
# x = top_20_all['Event']
# y = top_20_all['Medal']


# def func(x, a, b, c, d):
#     return a*np.exp(-c*(x-b))+d

# popt, pcov = curve_fit(func, x, y, [100,400,0.001,0])
# print(popt)

# ax = sns.scatterplot(data=top_20_med_all, y='Medal', x='Athletes', hue='NOC', hue_order=top_summer_order, palette=noc_colors)
# x=np.linspace(0,750,50)
# #plt.plot(x,func(x,*popt))
# plt.show()




# print(hel)











sns.set()











































top_10 = noc_total_df[noc_total_df['Top_10'] == True]
top_20 = noc_total_df[(noc_total_df['Top_20'] == True) & (noc_total_df['Top_10'] == False)]
top_20_med = top_20.groupby('Year').median()
top_20_med['NOC'] = 'ALL'
#top_20['NOC'] = 'ALL'
top_20_all = pd.merge(top_10, top_20_med, how='outer')

top_summer_order = top_10.groupby(['NOC'], as_index=False)['Medal'].sum().sort_values(by='Medal', ascending=False).NOC.tolist()
top_summer_order.append('ALL')








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

# Set the order of the top 10
top_summer_order = top_10.groupby(['NOC'], as_index=False)['Medal'].sum().sort_values(by='Medal', ascending=False).NOC.tolist()
top_summer_order.append('ALL')
top_20_not_med = noc_total_df[(noc_total_df['Top_20'] == True) & (noc_total_df['Top_10'] == False)]
top_20_not_med['NOC'] = 'ALL'
top_20_not_med_count = top_10.merge(top_20_not_med, how='outer')


sns.set()
athlete_var_list = [['Age', [10, 45], '(years)'], ['BMI', [15,35], '']]
medal_athlete = athlete_total_df[athlete_total_df['Winner']]
non_medal_athlete = athlete_total_df[athlete_total_df['Winner'] == False]



        









#df1 = noc_total_df[(noc_total_df['Year'] != 1984) & (noc_total_df['Year'] != 2012) & (noc_total_df['Year'] != 2004)]
#print(df1)
import matplotlib.animation as animation
print(noc_total_df[noc_total_df['NOC'] == 'CHN'])





#df = top_20_all[(top_20_all['NOC'] != 'CHN')]
#df = noc_total_df[(noc_total_df['NOC'] != 'CHN')]
df = noc_total_df
#df = df[(df['GDP'] < 500) & (df['Population'] < 200)]





# ax.set_xlim([-1, 1])
# ax.set_ylim([-1,1])
# ax.set_zlim([-1,1])
# plot = ax.scatter(x, y, z, color='b', marker= '*',)
# def func(i):
#     x_lim = ax.set_xlim(-i,i)
#     y_lim = ax.set_ylim(-i, i)
#     z_lim = ax.set_zlim(-i, i)
#     return plot

# ani = animation.FuncAnimation(fig, func, frames=100, interval=1000, blit=True)




# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# #df = top_20_all[(top_20_all['NOC'] != 'CHN')]
# #df = noc_total_df[(noc_total_df['NOC'] != 'CHN')]
# df = noc_total_df
# #df = top_20_all
# z =df.Athletes
# x =df.Event
# y =df.Medal
# ax.scatter(x, y, z, marker='o', cmap=noc_colors)
# ax.set_xlabel('Event')
# ax.set_ylabel('Medal')
# ax.set_zlabel('Athlete')
# plt.show()


quad = np.polyfit(noc_total_df['Event'],noc_total_df['Medal'],3)
v1 = np.polyval(quad, noc_total_df['Event'])
_,_,least,p,_ = stats.linregress(v1,noc_total_df['Medal'])


not_top_10 = noc_total_df[noc_total_df['Top_10'] == False]
not_top_10['NOC'] = 'Rest'
print(not_top_10)
#covariance = np.cov(x,y)
x = noc_total_df['Event']
y = noc_total_df['Medal']
pearson, _ = stats.pearsonr(x,y)
print('Pearson: ', pearson)
spearman,_ = stats.spearmanr(x,y)
print('Spearman: ', spearman)
_,_,least,p,_ = stats.linregress(x,y)
print('Least:', least, p)


print('FIT')

x = noc_total_df['Event']
y = noc_total_df['Medal']
quad = np.polyfit(x,y,4)
x1 = np.polyval(quad, x)
_,_,least,p,_ = stats.linregress(x1,y)
print('Least:', least)

df = noc_total_df[noc_total_df['Event'] < 150]
x2 = df['Medal']
y2 = df['Event']
quad = np.polyfit(x2,y2,3)
v2 = np.polyval(quad, x2)
_,_,least2,p,_ = stats.linregress(v2,y2)
print('Least:', least2)

print(h)




top_summer_order = top_10.groupby(['NOC'], as_index=False)['Medal'].sum().sort_values(by='Medal', ascending=False).NOC.tolist()
top_summer_order.append('Rest')


# Scatterplots - Top 20 medals against athlete and events
top_20_all = top_10.merge(top_20, how='outer')
df = top_20_all[top_20_all['Year'] != 1980]
y = 'Games_Medal_Perc'
plt.figure(figsize=(16,10))

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
plt.gcf().suptitle('Relationship between the percentage of Total Medals, and the Number of Events and Athletes')
plt.savefig('./images/graph/countries_medals_resid.png')
plt.show()



# Spearman - A linear relationship between the variables is not assumed, although a monotonic relationship is assumed. This is a mathematical name for an increasing or decreasing relationship between the two variables.







print(hel)

mean = np.zeros(3)
cov = np.random.uniform(.2, .4, (3, 3))
cov += cov.T
cov[np.diag_indices(3)] = 1
data = np.random.multivariate_normal(mean, cov, 100)
df = pd.DataFrame(data, columns=["X", "Y", "Z"])

def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)

g = sns.PairGrid(df, palette=["red"])
g.map_upper(plt.scatter, s=10)
g.map_diag(sns.distplot, kde=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_lower(corrfunc)








# ax = plt.subplot()
# bottom = [0]*len(years)
# color = 0
# for noc in top_summer_order:
#     country = all_count[all_count['NOC']==noc]
#     noc_perc = country.Games_Entries_Perc.tolist()    
#     for year in years:
#         if year not in country.Year.unique():
#             noc_perc.insert(years.index(year),0)
#     print(noc)
#     print(noc_perc)
#     print(bottom)
#     plt.bar(years, noc_perc, bottom=bottom, color=noc_colors[color], label=noc,width=-1.5, align='edge')
#     bottom = [sum(i) for i in zip(bottom, noc_perc)]     
#     color += 1
# ax.set_xticks(range(1956,2017,4))
# ax.set_yticks(range(0,101,5))
# #ax.set_xlabel(years)
# plt.xlabel('Years')
# plt.ylabel('Percentage of medals')
# plt.legend()
# plt.title('The Percentage of Entries from each country')
# plt.show()


# print(hel)




# sns.swarmplot(data=top_20_all, x='NOC', y='Games_Medal_Perc', order=top_summer_order, palette=noc_colors)
# plt.title('Athlete Success in Summer')
# plt.subplot(2,1,2)

#plt.show()

# plt.subplot(2,1,1)
# sns.swarmplot(data=top_20_all, x='NOC', y='Medal', order=top_summer_order, palette=noc_colors)
# plt.title('Athlete Success in Summer')


# plt.subplot(3,1,1)
# sns.swarmplot(data=top_20_all, x='NOC', y='Event', order=top_summer_order, palette=noc_colors)
# plt.subplot(3,1,2)
# sns.swarmplot(data=top_20_all, x='NOC', y='Athletes', order=top_summer_order, palette=noc_colors)
# plt.subplot(3,1,3)
# sns.swarmplot(data=top_20_all, x='NOC', y='Unique_Perc', order=top_summer_order, palette=noc_colors)
# plt.show()







# sns.lmplot(data=top_20_all, x='Games_Medal_Perc', y='Games_Entries_Perc', hue='NOC', palette=noc_colors, fit_reg=True, ci=0.025)
# plt.show()

# sns.scatterplot(data=noc_total_df, x='Games_Medal_Perc', y='Games_Entries_Perc')
# plt.show()




# # Games totals (Games, Year, #Athletes, #Medals, #Male, #Female, #Events)
# # Summer only - Replace Games in groupby with Year
# games_total_ath = all_df.groupby(['Year']).ID.count().reset_index()
# games_total_ath.columns = ['Year', 'Total_ID'] 
# games_athletes = all_df.groupby(['Year']).ID.nunique().reset_index()
# games_athletes.columns = ['Year', 'Unique_ID'] 
# games_events = all_df.groupby(['Year']).Event.nunique().reset_index()
# games_sports = all_df.groupby(['Year']).Sport.nunique().reset_index()
# games_medals = all_df.groupby(['Year']).Medal.count().reset_index()
# games_countries = all_df.groupby(['Year']).NOC.nunique().reset_index()
# games_male = all_df[all_df['Sex'] == 'M'].groupby('Year').ID.nunique().reset_index()
# games_male.columns = ['Year', 'Male']    
# games_female = all_df[all_df['Sex'] == 'F'].groupby('Year').ID.nunique().reset_index()
# games_female.columns = ['Year', 'Female']
# games_host = all_df[all_df['NOC'] == all_df['Host_NOC']].groupby('Year').Medal.count().reset_index()
# games_host.columns = ['Year', 'Host_Medal']
# games_visitor = all_df[all_df['NOC'] != all_df['Host_NOC']].groupby('Year').Medal.count().reset_index()
# games_visitor.columns = ['Year', 'Visitor_Medal']
# games_total_df = all_df[['Year', 'Host_NOC']] #Remove season, games
# games_total_df = games_total_df.drop_duplicates()
# games_total_df = games_total_df.merge(games_total_ath, how='outer') \
#                                 .merge(games_athletes, how='outer')\
#                                 .merge(games_events, how='outer')\
#                                 .merge(games_sports, how='outer')\
#                                 .merge(games_medals, how='outer')\
#                                 .merge(games_countries, how='outer')\
#                                 .merge(games_host, how='outer')\
#                                 .merge(games_visitor, how='outer')\
#                                 .merge(games_male, how='outer')\
#                                .merge(games_female, how='outer')

# games_total_df['Host_Medal_Perc'] = round((games_total_df.Host_Medal / games_total_df.Medal)*100, 2)
# games_total_df['Visitor_Medal_Perc'] = round((games_total_df.Visitor_Medal / games_total_df.Medal)*100, 2)
# print(games_total_df)






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


#noc_all_before_1950 = noc_total_df[noc_total_df['Year']<1950]
#noc_total_1950 = noc_total_df[noc_total_df['Year']>1950]
#winter_noc = noc_total_1950[noc_total_1950['Season'] == 'Winter']
#summer_noc = noc_total_1950

# Count who has the most medals since 1950
#noc_top_medals_summer = noc_total_1950[noc_total_1950['Season'] == 'Summer'].groupby(['NOC'], as_index=False)['Medal'].sum().sort_values(by='Medal', ascending=False).head(20)
#noc_top_medals_winter = noc_total_1950[noc_total_1950['Season'] == 'Winter'].groupby(['NOC'], as_index=False)['Medal'].sum().sort_values(by='Medal', ascending=False).head(20)
#top_winter_25 = noc_top_medals_winter.NOC.tolist()
#top_winter_10 = noc_top_medals_winter.NOC[:10].tolist()





# noc_total_1950 = noc_total_1950.drop(["Country", "Host"], axis=1).reset_index()
# summer_1950_10 = summer_noc[(summer_noc['NOC'].isin(top_summer_10))].reset_index()
# summer_1950_all = summer_noc[(summer_noc['NOC'].isin(top_summer_25)) &  (~summer_noc['NOC'].isin(top_summer_10))].reset_index()
# summer_1950_all_count = summer_1950_all.groupby(['Year']).median().reset_index()
# summer_1950_all_count['NOC'] = 'ALL'
# summer_1950 = pd.merge(summer_1950_10, summer_1950_all_count, how='outer')

# winter_1950_10 = winter_noc[(winter_noc['NOC'].isin(top_winter_10))].reset_index()
# winter_1950_all = winter_noc[(winter_noc['NOC'].isin(top_winter_25)) &  (~winter_noc['NOC'].isin(top_winter_10))].reset_index()
# winter_1950_all_count = winter_1950_all.groupby(['Games']).median().reset_index()
# winter_1950_all_count['NOC'] = 'ALL'
# winter_1950 = pd.merge(winter_1950_10, winter_1950_all_count, how='outer')




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




# plt.subplot(1,2,1)
# corr = top_10[['Medal_Perc', 'Medal', 'Athletes', 'Entries', 'Event', 'Male', 'Female', 'Unique_Perc', 'Games_Medal_Perc', 'Games_Entries_Perc']].corr()
# sns.heatmap(corr, annot=True) #linewidths=0.5, cmap='coolwarm'
# plt.title("Top 10 Only")
# plt.subplot(1,2,2)
# corr = top_20[['Medal_Perc', 'Medal', 'Athletes', 'Entries', 'Event', 'Male', 'Female', 'Unique_Perc', 'Games_Medal_Perc', 'Games_Entries_Perc']].corr()
# sns.heatmap(corr, annot=True) #linewidths=0.5, cmap='coolwarm'
# plt.title("Top 11-20 in Summer")
# plt.show()

# plt.subplot(1,2,1)
# corr = top_20_all[['Medal_Perc', 'Medal', 'Athletes', 'Entries', 'Event', 'Male', 'Female', 'Unique_Perc', 'Games_Medal_Perc', 'Games_Entries_Perc']].corr()
# sns.heatmap(corr, annot=True) #linewidths=0.5, cmap='coolwarm'
# plt.title("Top 20 in Summer")
# plt.subplot(1,2,2)
# corr = noc_total_df[noc_total_df['Top_20'] == False][['Medal_Perc', 'Medal', 'Athletes', 'Entries', 'Event', 'Male', 'Female', 'Unique_Perc', 'Games_Medal_Perc', 'Games_Entries_Perc']].corr()
# sns.heatmap(corr, annot=True) #linewidths=0.5, cmap='coolwarm'
# plt.title("All in Summer")
# plt.title("All")
# plt.show()




# sns.pairplot(top_10, kind='scatter', diag_kind='kde', vars=['Medal', 'Athletes', 'Entries', 'Event', 'Unique_Perc', 'Games_Medal_Perc', 'Games_Entries_Perc'])
# plt.subplots_adjust(top=0.95)
# plt.gcf().suptitle('Overall Relationship of NOC Variables for top 10')
# plt.show()







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
# best = summer_1950[summer_1950['NOC'] !='ALL']
# summer_1950_best = best.groupby(['NOC'], as_index=False).median()
# print(summer_1950_10)
# print(summer_1950_all_count)
# print(summer_1950_best)


sns.scatterplot(data=noc_total_df, x='Games_Medal_Perc', y='Games_Entries_Perc')
plt.title('All Countries')
plt.show()



row,col,pos = 3,1,1
plt.subplot(row,col,pos)
# sns.scatterplot(data=noc_total_df, x='Medal', y='Unique_Perc')
# pos+=1
plt.subplot(row,col,pos)
sns.scatterplot(data=noc_total_df, x='Medal', y='Entries')
pos+=1
# plt.subplot(row,col,pos)
# sns.scatterplot(data=noc_total_df, x='Medal', y='Athletes')
# pos+=1
plt.subplot(row,col,pos)
sns.scatterplot(data=noc_total_df, x='Medal', y='Event')
pos+=1
# plt.subplot(row,col,pos)
# sns.scatterplot(data=noc_total_df, x='Medal_Perc', y='Medal')
# pos+=1
plt.show()


# plt.subplot(2,1,1)
# sns.scatterplot(data=top_10, y='Medal_Perc', x='Unique_Perc', hue='NOC', hue_order=top_summer_order[:-1])
# sns.scatterplot(data=top_20_med, y='Medal_Perc', x='Unique_Perc', color='black', marker='D')
# plt.title('Best 25 in Summer split up')

# plt.subplot(2,1,2)
# sns.scatterplot(data=top_10, y='Medal', x='Unique_Perc', hue='NOC', hue_order=top_summer_order[:-1])
# sns.scatterplot(data=top_20_med, y='Medal', x='Unique_Perc', color='black', marker='D')
# plt.title('Best 25 in Summer split up')
# plt.show()

# sns.scatterplot(data=top_20_all, x='Medal', y='Unique_Perc', hue='NOC', size='Year', hue_order=top_summer_order, palette=noc_colors)
# plt.title('All on year')
# plt.show()



plt.subplot(2,1,1)
sns.scatterplot(data=top_20_all, x='Medal_Perc', y='Athletes', hue='NOC', hue_order=top_summer_order, palette=noc_colors)
plt.subplot(2,1,2)
sns.scatterplot(data=top_20_all, x='Medal_Perc', y='Event', hue='NOC', hue_order=top_summer_order, palette=noc_colors)
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
             data = top_20_all, 
             title = 'Iris sepals', 
             xlabel = 'Sepal Length (cm)',
             ylabel = 'Sepal Width (cm)')
plt.show()





sns.pairplot(noc_total_df, kind='scatter', diag_kind='hist', vars=['Athletes', 'Entries', 'Event', 'Medal'], plot_kws={'color':'C7'}, diag_kws={'color':'C7'})
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









