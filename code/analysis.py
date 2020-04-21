import pandas as pd
from matplotlib import pyplot as plt

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

# HISTOGRAM - events, athletes, medals
plt.figure()
# The number of events per year
winter_events = winter_df.groupby('Year').Event.nunique().reset_index()
summer_events = summer_df.groupby('Year').Event.nunique().reset_index()
ax1 = plt.subplot(1,3,1)
bins = range(10, 320, 10)
plt.hist(winter_events.Event, range=(10,100), bins=bins)
plt.hist(summer_events.Event, range=(10,310), bins=bins)
ax1.set_xticks(range(0, 310, 50))
ax1.set_yticks(range(7))
plt.xlabel('Number of events per year')
plt.ylabel('Number of years')
plt.title('Number of events')

# The number of athletes
winter_athletes = winter_df.groupby('Year').ID.nunique().reset_index()
summer_athletes = summer_df.groupby('Year').ID.nunique().reset_index()
ax2 = plt.subplot(1,3,2)
bins = range(200, 12000, 500)
plt.hist(winter_athletes.ID, bins=bins)
plt.hist(summer_athletes.ID, bins=bins)
ax2.set_xticks(range(0, 12000, 2000))
ax2.set_yticks(range(8))
plt.xlabel('Number of athletes per year')
plt.ylabel('Number of years')
plt.title('Number of athletes')

# The number of medals
winter_medals = winter_df.groupby('Year').Medal.count()
summer_medals = summer_df.groupby('Year').Medal.count()
print(winter_medals)
ax3 = plt.subplot(1,3,3)
bins = range(0, 6000, 75)
plt.hist(winter_medals, bins=bins)
plt.hist(summer_medals, bins=bins)
ax3.set_xticks(range(0, 6000, 1000))
ax3.set_yticks(range(7))
plt.xlabel('Number of medals per year')
plt.ylabel('Number of years')
plt.title('Number of medals')

plt.show()





# both_df = summer_df[summer_df.ID.isin(winter_df.ID.unique())]
#count_both, list_both = check_item_in(winter_df.ID, summer_df.ID)


# Unique years
years = []
for year in winter_df.Year.unique():
    years.append(year)

# Years are column, NOC is rows
summer_medals = summer_df[summer_df.Medal.notna()].groupby('Year')['NOC'].value_counts().unstack().fillna(0)
# NOC are column, Years rows
summer_medals = summer_medals.groupby('Year').sum()
# Add total medals in rows
summer_medals['Total'] = summer_medals.sum(axis=1)