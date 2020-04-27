import pandas as pd

# USEFUL TIPS
# Groupby and count number of value, switch data to be column headers (https://towardsdatascience.com/pandas-tips-and-tricks-33bcc8a40bb9)
# Add engine='python' to read data file properly (https://www.shanelynn.ie/pandas-csv-error-error-tokenizing-data-c-error-eof-inside-string-starting-at-line/)
# Put whole file path to help find it (didnt like athlete_events.csv)

# THOUGHTS
# avg height, weight, age = country / all / medal winners / NA medals / sport / year /
# men/women = all / country / year
# athletes = country / year
# GDP/population to number of medals = 
# which country wins medals in what
# Number of athletes, nations, events overall
# Women in Olympics - gender overall, make v female per noc, difference in medal counts
# Geographic
# Height/weight of athletes (from 1960) - height over time, weight over time, 



# Read file
all_df = pd.read_csv('F:/TEAN/Portfolio/olympics/data/athlete_events.csv', index_col=0, engine='python')
all_df.head()

# SUMMER
# Select all Summer games
summer_df = all_df[all_df['Season'] == 'Summer']


# REPRESENTATION
# Number of athletes sorted by each country (NOC region)
country_df = summer_df.groupby('NOC')['Year'].value_counts().unstack().fillna(0)
country_df.head()

# Number of male athletes for each country
male_df = summer_df[summer_df['Sex'] == 'M'] .groupby('NOC')['Year'].value_counts().unstack().fillna(0)
male_df.head()

# Number of female athletes for each country
female_df = summer_df[summer_df['Sex'] == 'F'] .groupby('NOC')['Year'].value_counts().unstack().fillna(0)
female_df.head()

# THE PERFECT OLYMPIC BODY
# Average height, weight and age for each country (compare with gender)
# Average height, weight and age for each country for medal winners


# DOES COUNTRY AFFECT CHANCE OF WINNING A MEDAL
# Number of athletes to population, GDP


# Select all with no medals
no_medals = summer_df[(olympics_df['Medal'] != 'Gold') & (olympics_df['Medal'] != 'Silver') & (olympics_df['Medal'] != 'Bronze')].groupby('NOC')['Year'].value_counts().unstack().fillna(0)
no_medals.head()

# Select all with medals
medals = summer_df[(olympics_df['Medal'] == 'Gold') | (olympics_df['Medal'] == 'Silver') | (olympics_df['Medal'] == 'Bronze')].groupby('NOC')['Year'].value_counts().unstack().fillna(0)
medals.head()






# WINTER
# Select all Winter games
winter_df = all_df[all_df['Season'] == 'Winter']




# Unique years
years = []
for year in winter_df.Year.unique():
    years.append(year)

# Years are column, NOC is rows
#summer_medals = summer_df[summer_df.Medal.notna()].groupby('Year')['NOC'].value_counts().unstack().fillna(0)
# NOC are column, Years rows
#summer_medals = summer_medals.groupby('Year').sum()
# Add total medals in rows
#summer_medals['Total'] = summer_medals.sum(axis=1)