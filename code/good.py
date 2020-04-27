import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm  
import numpy as np
from plots import *
from statsmodels.graphics.gofplots import qqplot_2samples


# Read in data and split between summer and winter
all_df = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/all_data.csv', index_col=0).reset_index(drop = True)

## Add winner 
all_df['Winner'] = all_df.Medal.notna()

# Add BMI columns [Weight (kg) / Height^2 (m)]
all_df['BMI'] = all_df.apply(lambda x: round(x.Weight/((x.Height/100)**2), 2), axis=1)




######## THE ATHLETE #######


# Athlete Totals (Games, Year, ID, Sex, Age, BMI, Season, NOC, #Events, #Medals)
athlete_events = all_df.groupby(['Games', 'ID']).Event.count().reset_index()
athlete_medals = all_df.groupby(['Games', 'ID']).Medal.count().reset_index()
athlete_total_df = all_df[['Games', 'Year', 'ID', 'Sex', 'Age', 'BMI', 'Season', 'NOC', 'Winner']]
athlete_total_df = athlete_total_df.drop_duplicates()
athlete_total_df = athlete_total_df.merge(athlete_events, how='outer') \
                                    .merge(athlete_medals, how='outer')


# Athlete split up
winter_athlete = [athlete_total_df[athlete_total_df['Season'] == 'Winter'], 'Winter', 'C0']
summer_athlete = [athlete_total_df[athlete_total_df['Season'] == 'Summer'], 'Summer', 'C1']
male_athlete = athlete_total_df[athlete_total_df['Sex'] == 'M']
female_athlete = athlete_total_df[athlete_total_df['Sex'] == 'F']
medal_athlete = athlete_total_df[athlete_total_df['Winner']]
non_medal_athlete = athlete_total_df[athlete_total_df['Winner'] == False]

athlete_var_list = ['Age', 'BMI', 'Event', 'Medal']
athlete_colors ={'Winter': 'C0', 'Summer':'C1', 'F': 'C6', 'M': 'C9', False: 'C4', True: 'C2'}

# DISTRIBUTION
def athlete_hist(category, *dfs): 
    plt.figure()
    plt.gcf().suptitle('Distribution of Athlete Attributes {category}'.format(category=category)) 
    plot = [1,4,0]   

    for var in athlete_var_list:
        plot_min = int(athlete_total_df[var].min())
        plot_max = int(athlete_total_df[var].max())
        bins = int(plot_max - plot_min)

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

athlete_hist('Overall', [athlete_total_df, 'All', 'C7'])
athlete_hist('by Season', winter_athlete, summer_athlete)
athlete_hist('by Gender', male_athlete, female_athlete)
athlete_hist('by Winner', medal_athlete, non_medal_athlete)


# RELATIONSHIP
# All
sns.pairplot(athlete_total_df, kind='scatter', diag_kind='hist', vars=athlete_var_list, plot_kws={'color':'C7'}, diag_kws={'color':'C7'})
plt.subplots_adjust(top=0.95)
plt.gcf().suptitle('Overall Relationship of Athlete Variables')
plt.show()

# Each category
hue_list = ['Season', 'Sex', 'Winner']
for hue in hue_list: 
    sns.pairplot(athlete_total_df, hue=hue, kind='scatter', diag_kind='hist', vars=athlete_var_list, palette=athlete_colors)
    plt.subplots_adjust(top=0.95)
    plt.gcf().suptitle('Relationship of Athlete Variables by {hue}'.format(hue=hue))
    plt.legend()
    plt.show()


# COMPARISON
# QQ plot
def athlete_qqplot(df1, df2):
    plot = [2, 2, 0]
    for var in athlete_var_list:
        plot[2] += 1
        ax = plt.subplot(plot[0], plot[1], plot[2])
        ax.axis(facecolor='blue')
        qqplot_2samples(df1[0][var], df2[0][var], xlabel=df1[1], ylabel=df2[1], line='45', ax=ax)
        plt.title('{var}'.format(var=var))
    plt.subplots_adjust(top=0.9)
    plt.gcf().suptitle('Comparison of Athlete Variables')
    plt.show()
athlete_qqplot(winter_athlete, summer_athlete)
athlete_qqplot(male_athlete, summer_athlete)
athlete_qqplot(medal_athlete, non_medal_athlete)

# Barplot for Medal and Event, Boxplot for Age and BMI
plot = [2,2,0]
for hue in athlete_hue_list:
    for var in ['Age','BMI']:
        plot[2] += 1
        plt.subplot(plot[0], plot[1], plot[2])
        sns.boxplot(x=hue , y=var, palette=athlete_colors, data=athlete_total_df)
    for var in ['Event', 'Medal']:
        plot[2] += 1
        plt.subplot(plot[0], plot[1], plot[2])
        sns.barplot(x=hue , y=var, palette=athlete_colors, data=athlete_total_df)
    plt.gcf().suptitle('Comparison for all athletes')
    plt.show()
    plot[2] = 0

# Boxplot and barplot broken down for each category
plot = [4,3,0]
for hue in athlete_hue_list:
    for var in athlete_var_list:   
        plot[2] += 1
        plt.subplot(plot[0], plot[1], plot[2]) 
        if var in ['Age', 'BMI']:
            sns.boxplot(x=hue , y=var, palette=athlete_colors, data=athlete_total_df)
        else:
            sns.barplot(x=hue , y=var, palette=athlete_colors, data=athlete_total_df)
        for hue2 in athlete_hue_list:
            if hue != hue2:
                plot[2] += 1
                plt.subplot(plot[0], plot[1], plot[2])
                if var in ['Age', 'BMI']:
                    sns.boxplot(x=hue , y=var, hue=hue2, palette=athlete_colors, data=athlete_total_df)
                else:
                    sns.barplot(x=hue , y=var, hue=hue2, palette=athlete_colors, data=athlete_total_df)
                  
    plt.gcf().suptitle('Comparison for all athletes')
    plt.show()
    plot[2] = 0




# CHANGE OVER YEARS
#Overall
plot = [2,1,0]
for var in ['Event', 'Medal']:    
    plot[2] += 1
    plt.subplot(plot[0],plot[1],plot[2])
    sns.barplot(x='Year' , y=var, color='C7', data=athlete_total_df)
    plt.title('The number of {var}s by each athlete over time'.format(var=var))
plt.show()
plot[2] = 0

for var in ['Age', 'BMI']:
    plot[2] += 1
    plt.subplot(plot[0],plot[1],plot[2])
    sns.boxplot(x='Year' , y=var, color='C7', data=athlete_total_df)
    plt.title('The {var} of all athlete over time'.format(var=var))
plt.show()

# Each category
plot = [2,1,0]
for hue in athlete_hue_list:
    for var in ['Event', 'Medal']:    
        plot[2] += 1
        plt.subplot(plot[0],plot[1],plot[2])
        sns.barplot(x='Year' , y=var, hue=hue, palette=athlete_colors, data=athlete_total_df)
        plt.gcf().suptitle('The average of each Athlete Variable over time by {hue}'.format(var=var, hue=hue))
    plt.show()
    plot[2] = 0

    for var in ['Age', 'BMI']:    
        plot[2] += 1
        plt.subplot(plot[0],plot[1],plot[2])
        sns.boxplot(x='Year' , y=var, hue=hue, palette=athlete_colors, data=athlete_total_df)
        plt.gcf().suptitle('The spread of each Athlete Variale over time by {hue}'.format(var=var, hue=hue))
    plt.show()
    plot[2] = 0








######## THE GAMES #######

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

# Games split up
winter_games = games_total_df[games_total_df['Season'] == 'Winter']
summer_games = games_total_df[games_total_df['Season'] == 'Summer']

games_var_list = ['Unique_ID', 'Total_ID', 'Male', 'Female', 'NOC', 'Event', 'Sport', 'Medal']

# DISTRIBUTION
def games_hist(category, *dfs): 
    plt.figure()
    plt.gcf().suptitle('Distribution of Total Attributes for each Games {category}'.format(category=category))     
    plot = [2,4,0] 

    for var in games_var_list:
        plot[2] += 1
        plt.subplot(plot[0], plot[1], plot[2])        
        for df in dfs: 
            plt.hist(df[0][var], label=df[1], color=df[2], histtype='bar', alpha=0.8)          
            #sns.distplot(df[0][var], label=df[1], kde=True, color=df[2], bins=10)
            

        plt.xlabel(var)
        plt.title('Distribution of {var} at each Games'.format(var=var)) 
        plt.legend()
    plt.show()

games_hist('Overall', [games_total_df, 'All', 'C7'])
games_hist('by Season', [summer_games, 'Summer', 'C1'], [winter_games, 'Winter', 'C0'])












######### THE COUNTRIES ###############

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


country_var_list = ['Total_ID', 'Unique_ID', 'Male', 'Female', 'Event', 'Medal']



#### Host Medal Percentage vs. Average Percentage ####
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

# Plot of difference
facet = sns.lmplot(data=host_difference, x='Medal_Perc', y='Host_Medal_Perc', hue='Season', palette=athlete_colors, robust=True)
plt.plot([0,20],[0,20], 'black', linewidth=2, linestyle='dashed')
facet.ax.set_xticks(np.arange(0,21,2.5))
facet.ax.set_yticks(np.arange(0,41,2.5))
facet.ax.ticklabel_format(useOffset=False)
facet.ax.set_xlim(left=0)
facet.ax.set_ylim(bottom=0)
plt.title('The difference in percentage of medals won by host countries')
plt.legend()
plt.show()





# Heatmap of all variables
corr = noc_total_df[['Unique_ID', 'Total_ID', 'Event', 'Medal', 'Male', 'Female']].corr()
sns.heatmap(corr, annot=True) #linewidths=0.5, cmap='coolwarm'
plt.show()