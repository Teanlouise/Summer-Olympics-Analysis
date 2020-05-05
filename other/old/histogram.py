import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Read in data and split between summer and winter
all_df = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/all_data.csv', index_col=0).reset_index(drop = True)
winter_df = all_df[all_df.Season == 'Winter']
summer_df = all_df[all_df.Season == 'Summer']


# HISTOGRAM - events, athletes, medals seperate
def get_hist(data1, data2, plot, bins, plot_range, x_ticks, y_ticks, title):    
    ax = plt.subplot(1,3,plot)
    plt.hist(data1, range=plot_range, bins=bins, label='Winter', histtype='step')
    plt.hist(data2, range=plot_range, bins=bins, label='Summer', histtype='step')
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    plt.xlabel('Number of {title} per year'.format(title=title))
    plt.title('Distribution of {title}'.format(title=title))
    plt.legend()


winter_years = winter_df.groupby('Year')
summer_years = summer_df.groupby('Year')
plt.figure()
#The number of events per year
winter_events = winter_years.Event.nunique().reset_index()
summer_events = summer_years.Event.nunique().reset_index()
get_hist(data1=winter_events.Event, 
            data2=summer_events.Event, 
            plot=1, 
            bins=range(10, 320, 10), 
            plot_range=(10,310), 
            x_ticks=range(0, 310, 50), 
            y_ticks=range(7), 
            title='events')
# The number of athletes
winter_athletes = winter_years.ID.nunique().reset_index()
summer_athletes = summer_years.ID.nunique().reset_index()
get_hist(data1=winter_athletes.ID, 
            data2=summer_athletes.ID, 
            plot=2, 
            bins=range(200, 12000, 500), 
            plot_range=(10,310), 
            x_ticks=range(0, 12000, 2000), 
            y_ticks=range(8), 
            title='athletes')
# The number of medals
winter_medals = winter_years.Medal.count().reset_index()
summer_medals = summer_years.Medal.count().reset_index()
get_hist(data1=winter_medals.Medal, 
            data2=summer_medals.Medal, 
            plot=3, 
            bins=range(0, 6000, 75), 
            plot_range=(10,310), 
            x_ticks=range(0, 6000, 1000), 
            y_ticks=range(7), 
            title='medals')
plt.show()




# HISTOGRAM - events, athletes, medals - summer and winter seperate
plt.figure()
bins=100
plot_range=(0,2500)
# Winter
ax1 = plt.subplot(1,2,1)
winter_events = winter_df.groupby('Year').Event.nunique().reset_index()
plt.hist(winter_events.Event, range=plot_range, bins=bins, label='Events')
winter_athletes = winter_df.groupby('Year').ID.nunique().reset_index()
plt.hist(winter_athletes.ID, range=plot_range, bins=bins, label='Athletes')
winter_medals = winter_df.groupby('Year').Medal.count()
plt.hist(winter_medals, range=plot_range, bins=bins, label='Medals')
plt.xlabel('Number per year')
plt.title('Winter')
plt.legend()
# Summer
plot_range=(0,12000)
ax2 = plt.subplot(1,2,2)
summer_events = summer_df.groupby('Year').Event.nunique().reset_index()
plt.hist(summer_events.Event, range=plot_range, bins=bins, label='Events')
summer_athletes = summer_df.groupby('Year').ID.nunique().reset_index()
plt.hist(summer_athletes.ID, range=plot_range, bins=bins, label='Athletes')
summer_medals = summer_df.groupby('Year').Medal.count()
plt.hist(summer_medals, range=plot_range, bins=bins, label='Medals')
plt.xlabel('Number per year')
plt.title('Summer')
plt.legend()
plt.show()