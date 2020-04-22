import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from seaborn_qqplot import qqplot
import numpy as np

def check_item_in(df1, df2):
    item_list = []
    count = 0       
    for item in df1.unique():
        if item in df2.unique():
            item_list.append(item)
            count+=1
    return count, item_list


# Read in data and split between summer and winter
all_df = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/all_data.csv', index_col=0).reset_index(drop = True)
winter_df = all_df[all_df.Season == 'Winter']
summer_df = all_df[all_df.Season == 'Summer']


def get_pairwise(season):
    season_df = all_df[all_df.Season == season]
    years = season_df.groupby('Year')

    athletes = years.ID.nunique().reset_index()
    events = years.Event.nunique().reset_index()
    medals = years.Medal.count().reset_index()

    df = pd.DataFrame()
    df['Year'] = season_df.Year.unique()
    df = df.merge(athletes, how='outer')
    df = df.merge(events, how='outer')
    df = df.merge(medals, how='outer')
    df['Season'] = season
    df = df.drop(columns=['Year'], axis=1)
    return df


sns.kdeplot(winter_df.groupby('Year').ID.count(), shade=True)
plt.show()

winter_df = get_pairwise("Winter")
summer_df = get_pairwise("Summer")
df = pd.concat([winter_df, summer_df])
sns.pairplot(df, hue='Season')
plt.show()

# all_df['Winner'] = all_df.Medal.notna()
# all_df['Weight'] = all_df.Weight.fillna(0)
# all_df['Height'] = all_df.Height.fillna(0)


# none_df = all_df[(all_df.Weight == 0) | (all_df.Height == 0)]
# all_df = all_df.drop(none_df.index)

# #winners = all_df[all_df.Medal.notna() == True]
# #non_winners = all_df[all_df.Medal.notna() == False]

# medal_df = pd.DataFrame()
# medal_df['BMI Medal'] = all_df.apply(lambda x: get_BMI(x.Weight, x.Height/100) if x.Winner==True else 0, axis=1)
# medal_df['BMI Non-Medal'] = all_df.apply(lambda x: get_BMI(x.Weight, x.Height/100) if x.Winner==False else 0, axis=1)

# print(medal_df)

# qqplot(medal_df, x="BMI Medal", y='BMI Non-Medal', display_kws={"identity":True,"fit":True,"reg":True,"ci":0.025})
# plt.show()


# BMI Boxplot for medal and non-medal for season
# plt.figure()
# plt.subplot(1,2,1)
# sns.boxplot(data=all_df, x='Season', y='BMI', hue='Winner')
# plt.title('Overall BMI of Athletes by Season')
# plt.subplot(1,2,2)
# sns.boxplot(data=all_df, x='Season', y='BMI', hue='Medal')
# plt.title('Overall BMI of Medal Winners by Season')
# plt.show()







#count_both, list_both = check_item_in(winter_df.ID, summer_df.ID)


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