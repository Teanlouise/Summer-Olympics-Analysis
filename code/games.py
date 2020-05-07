import pandas as pd, seaborn as sns, numpy as np
from matplotlib import pyplot as plt
# Read data
games_total_df = pd.read_csv(
            './data/games_total.csv', index_col=0)
games_total_before = pd.read_csv(
            './data/games_total_before.csv', index_col=0)\
            .reset_index(drop = True)
# Set style
sns.set()
title_dict = {'fontsize': 16, 'fontweight': 'bold'}

##### GRAPHS######
# Histogram - Distribution of games before and after 1955
games_var_list = ['Entries', 'Athletes', 
                    'Male', 'Female', 
                    'NOC', 'Event', 
                    'Sport', 'Medal']

games_total_before = games_total_before[
                        (games_total_before['Season'] == 'Summer') 
                        & (games_total_before['Year']<1955)]
plt.figure(figsize=[12,8])
plt.gcf().suptitle('Distribution of Games Attributes for all Summer Olympics', 
            fontdict=title_dict)     
plot = [2,4,0] 
bins = 10
dfs = [games_total_before, 'Before 1955'], \
        [games_total_df, 'After 1955'] 
for var in games_var_list:
    plot[2] += 1
    plt.subplot(plot[0], plot[1], plot[2])
    for df in dfs: 
        if (plot[2] == 4) and (df[1] =='Before 1955'):
            bins=5
        plt.hist(df[0][var], label=df[1], histtype='bar', 
                    alpha=0.7, bins=bins)
        bins=10
        if plot[2] == 4:
            plt.legend()  
    plt.xlabel('Number of {var}'.format(var=var))
    plt.ylabel('Number of occurences')  
plt.subplots_adjust(top=0.9, left=0.08, right=0.95,wspace=0.4)
plt.savefig('./images/graph/games_histogram.png')
plt.show()