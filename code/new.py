import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm  
import numpy as np
from plots import *
from statsmodels.graphics.gofplots import qqplot_2samples


############### DATA FILES ######################

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


male_df = all_df[all_df.Sex == 'M'] 
female_df = all_df[all_df.Sex == 'F'] 

medal_df = all_df[all_df.Medal.notna()] 
non_medal_df = all_df[all_df.Medal.notna() == False] 


# Add whether athlete competed both winter and summer
all_df['Both'] = summer_df.ID.isin(winter_df.ID.unique())

# Create new dataframe with count of variables
def season_count_df(df, season):
    years = df.groupby(['Year', 'Sex'])
    athletes = years.ID.nunique().reset_index()
    events = years.Event.nunique().reset_index()
    medals = years.Medal.count().reset_index()
    countries = years.NOC.nunique().reset_index()
    male = df[df['Sex'] == 'M'].groupby('Year').ID.nunique().reset_index()
    male.columns = ['Year', 'Male']    
    female = df[df['Sex'] == 'F'].groupby('Year').ID.nunique().reset_index()
    female.columns = ['Year', 'Female']

    new_df = pd.DataFrame()
    new_df['Year'] = df.Year.unique()
    new_df = new_df.merge(athletes, how='outer')
    new_df = new_df.merge(events, how='outer')
    new_df = new_df.merge(medals, how='outer')
    new_df = new_df.merge(countries, how='outer')
    new_df = new_df.merge(male, how='outer')
    new_df = new_df.merge(female, how='outer')
    new_df['Season'] = season    
    return new_df

winter_count_df = season_count_df(winter_df, "Winter")
summer_count_df = season_count_df(summer_df, "Summer")
all_count_df = pd.concat([winter_count_df, summer_count_df]).sort_values('Year').reset_index(drop=True)




###### AGE & BMI #######

#HISTOGRAM
plt.figure()
plot= [2,3,0]
plot[2] = all_hist('Age', all_df, winter_df, summer_df, male_df, female_df, medal_df, non_medal_df, plot)
all_hist('BMI', all_df, winter_df, summer_df, male_df, female_df, medal_df, non_medal_df, plot)
plt.show()

# Histogram and KDE plots
plt.figure()
all_hist_kde('Age', all_df, winter_df, summer_df, male_df, female_df, medal_df, non_medal_df)
plt.show()
all_hist_kde('BMI', all_df, winter_df, summer_df, male_df, female_df, medal_df, non_medal_df)
plt.show()


#BOXPLOTS
boxplots_year('Age', all_df)
boxplots_hues('Age', all_df)
plt.show()
boxplots_year('BMI', all_df)
boxplots_hues('BMI', all_df)
plt.show()

#BAR PLOT
plt.subplot(2,1,1)
sns.barplot(x='Year' , y='Age', data=all_df)
plt.subplot(2,1,2)
sns.barplot(x='NOC' , y='Age', data=all_df)
plt.show()
hue_list = ['Season', 'Sex', 'Winner', 'Medal']
pos = 1
for hue in hue_list: 
    plt.subplot(4, 2, pos)
    sns.barplot(x='Year' , y='Age', hue=hue, data=all_df)
    pos+=1
    plt.subplot(4, 2, pos)
    sns.barplot(x='NOC' , y='Age', hue=hue, data=all_df)
    pos+=1
plt.show()


#PAIRWISE
hue_list = ['Season', 'Sex', 'Winner', 'Medal']
for hue in hue_list: 
    sns.pairplot(all_df, hue=hue, kind='reg', vars=['Age', 'BMI'])
    plt.show()


#QQPLOT
qqplot_2('Age', medal_df, non_medal_df, male_df, female_df, winter_df, summer_df)
plt.show()
qqplot_2('BMI', medal_df, non_medal_df, male_df, female_df, winter_df, summer_df)
plt.show()





##### SEX ####

#Distribution
plt.figure()


var = 'Age'
ax = plt.subplot()
plot_min = int(all_df[var].min())
plot_max = int(all_df[var].max())
bins = int(plot_max - plot_min)
plt.hist(all_df[all_df['Sex'] == 'M'][var], range=(plot_min, plot_max), bins=bins, label='Male', histtype='step')
plt.hist(all_df[all_df['Sex'] == 'F'][var], range=(plot_min, plot_max), bins=bins, label='Female', histtype='step')
ax.set_xticks(range(plot_min, plot_max, 5))
plt.xlabel(var)
plt.title('Distribution of {var}'.format(var=var)) 

plt.figure()
plt.subplot(3,1,1)
sns.boxplot(data=all_df, x='Games', y='Age', hue='Sex')
plt.subplot(3,1,2)
sns.boxplot(data=all_df, x='Games', y='BMI', hue='Sex')
plt.subplot(3,1,3)
sns.boxplot(data=all_df, x='Games', y='Winner', hue='Sex')
plt.show()






#Number of each sex at each year
qqplot(all_count_df, 
            x='Male', 
            y='Female', 
            hue='Season', 
            height=3, 
            aspect=1.5, 
            display_kws={"identity":True,"fit":True,"reg":True,"ci":0.025})
plt.axis([0, 7000,0,7000])
plt.show()














##### CHANGE IN AVERAGE OF VARIABLES ##########
var_list = ['BMI', 'Age']
hue_list = ['Sex', 'Season', 'Winner', 'Both']

# Plots of each variable
for var in var_list:
    single_int_plots(var, all_df, winter_df, summer_df)
    plt.show()

# Pairwise for each hue
for hue in hue_list:
    olympic_pairwise(all_df, hue, var_list)
    plt.show()

# Athletes who competed in both seasons
plt.figure()
plt.subplot(1,2,1)
both_boxplot('Age', all_df)
plt.subplot(1,2,2)
both_boxplot('BMI', all_df)
plt.show()



######### CHANGE IN NUMBER OF UNIQUE VARIABLES #########

# Athletes, Medals, Events, Countries
var_list = ['ID', 'NOC', 'Event', 'Medal']
hue_list = ['Sex', 'Season']

# Pairwise for each hue
for hue in hue_list:
    olympic_pairwise(all_count_df, hue, var_list)
plt.show()











    

# Add Title above whole figure
plt.gcf().suptitle('THIS IS A TITLE, YOU BET')


############ EXTRA ########################

# Athletes in both summer and winter

# Difference in age between those who compete in both winter and summer

# both_df = summer_df[summer_df.ID.isin(winter_df.ID.unique())]
all_df['Both'] = summer_df.ID.isin(winter_df.ID.unique())
sns.boxplot(data=all_df, x='Both', y='Age')
#plt.show()