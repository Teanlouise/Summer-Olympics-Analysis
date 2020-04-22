import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from plots import olympic_hist, year_boxplot, season_boxplot

# Read in data and split between summer and winter
all_df = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/all_data.csv', index_col=0).reset_index(drop = True)
# Add Winner
all_df['Winner'] = all_df.Medal.notna()
# Split by season
winter_df = all_df[all_df.Season == 'Winter'] 
summer_df = all_df[all_df.Season == 'Summer'] 


################ HISTOGRAM ###############################
plt.figure()
olympic_hist('Age')
plt.show()


################ BOXPLOT ###############################
       
# Age of Athletes by Year, each season
plt.figure()
year_boxplot('Age', 'Winner', 'Athletes')
plt.show()

# Age of Medal Winners by Year, each season
plt.figure()
year_boxplot('Age', 'Medal', 'Medal Winners')
plt.show()

# Age of Athletes by Season & Age of Medal Winners by Season
plt.figure()
plt.subplot(1,2,1)
season_boxplot('Age', 'Winner', 'Athletes')
plt.subplot(1,2,2)
season_boxplot('Age', 'Medal', 'Medal Winners')
plt.show()






############ EXTRA ########################

# Difference in age between those who compete in both winter and summer
both_df = summer_df[summer_df.ID.isin(winter_df.ID.unique())]
all_df['Both'] = summer_df.ID.isin(winter_df.ID.unique())
sns.boxplot(data=all_df, x='Both', y='Age')
#plt.show()