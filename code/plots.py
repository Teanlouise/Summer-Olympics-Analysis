import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sns_qqplot import qqplot

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


def olympic_hist(var):    
    ax = plt.subplot()
    plot_min = int(all_df[var].min())
    plot_max = int(all_df[var].max())
    bins = int(plot_max - plot_min)
    plt.hist(winter_df[var], range=(plot_min, plot_max), bins=bins, label='Winter', histtype='step')
    plt.hist(summer_df[var], range=(plot_min, plot_max), bins=bins, label='Summer', histtype='step')
    ax.set_xticks(range(plot_min, plot_max, 5))
    plt.xlabel(var)
    plt.title('Distribution of {var}'.format(var=var)) 



def season_boxplot(y, hue, title):   
    x = 'Season' 
    sns.boxplot(data=all_df, x=x, y=y, hue=hue)
    plt.title('{y} of All {title} by {x}'.format(y=y, x=x, title=title ))


def year_boxplot(y, hue, title):
    x = 'Year'
    plt.subplot(2,1,1)
    sns.boxplot(data=winter_df, x=x, y=y, hue=hue)
    plt.title('{y} of Winter {title} by {x}'.format(y=y, x=x, title=title ))

    plt.subplot(2,1,2)
    sns.boxplot(data=summer_df, x=x, y=y, hue=hue)
    plt.title('{y} of Summer {title} by {x}'.format(y=y, x=x, title=title ))












def season_qqplot(df, season): 
    df = df.groupby(['Year', 'Winner']).BMI.mean().reset_index()
    df = df.pivot(columns='Winner', index='Year', values='BMI').reset_index()
    df.columns = ['Year', 'Medal', 'Non-Medal']
    df['Season'] = season    
    return df

def olympic_qqplot():
    df1 = season_qqplot(winter_df, "Winter")
    df2 = season_qqplot(summer_df, "Summer")
    df = pd.concat([df1, df2]).reset_index()
    # Use seaborn_qqplot package
    # Remove [::-1] from label at bottom of code, wrong legend order
    qqplot(df, 
            x='Medal', 
            y='Non-Medal', 
            hue='Season', 
            height=3, 
            aspect=1.5, 
            display_kws={"identity":True,"fit":True,"reg":True,"ci":0.025})