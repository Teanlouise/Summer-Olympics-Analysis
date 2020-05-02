import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

athlete_total_df = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/athlete_total.csv', index_col=0)
games_total_df = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/games_total.csv', index_col=0)
noc_total_df = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/noc_total.csv', index_col=0)

########## THE GAMES #############
games_var_list = ['Entries', 'Athletes', 'Male', 'Female', 'NOC', 'Event', 'Sport', 'Medal']

# Histogram - Distribution of games before and after 1955
games_total_before = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/games_total_draft.csv', index_col=0).reset_index(drop = True)
games_total_before = games_total_before[(games_total_before['Season'] == 'Summer') & (games_total_before['Year']<1955)]

plt.figure(figsize=[12,8])
plt.gcf().suptitle('Distribution of Games Attributes for all Summer Olympics')     
plot = [2,4,0] 
dfs = [games_total_before, 'Before 1955'], [games_total_df, 'After 1955']
for var in games_var_list:
    plot[2] += 1
    plt.subplot(plot[0], plot[1], plot[2])        
    for df in dfs: 
        plt.hist(df[0][var], label=df[1], histtype='bar', alpha=0.7, bins=10)
    plt.xlabel('Number of {var}'.format(var=var))
plt.legend(loc='right', bbox_to_anchor=(0, 0), ncol=1)
plt.savefig('./images/graph/games_histogram.png')
plt.show()




########## THE ATHLETES #############

plt.figure(figsize=[18,8])
sns.boxplot(x='Year' , y='Age', data=athlete_total_df, hue='Sex')
plt.title('The Age of Summer Athletes since 1956 by Gender')
plt.legend('Male', 'Female')
plt.savefig('./images/graph/athlete_age_boxplot.png')
plt.show()

plt.figure(figsize=[18,8])
sns.violinplot(x='Year', y='BMI', data=athlete_total_df, hue='Sex', split=True)
plt.title('The BMI of Summer Athletes since 1956 by Gender')
plt.savefig('./images/graph/athlete_bmi_violinplot.png')
plt.show()





