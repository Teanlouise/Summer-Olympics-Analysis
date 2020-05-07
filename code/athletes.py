import pandas as pd, seaborn as sns, numpy as np
from matplotlib import pyplot as plt
# Read the data
athlete_total_df = pd.read_csv(
            './data/athlete_total.csv', index_col=0)
#Set the style
sns.set()
year_fig = [16,8]
title_dict = {'fontsize': 16, 'fontweight': 'bold'}

### GRAPHS ###
## Boxplot - Age
plt.figure(figsize=year_fig)
ax = plt.subplot()
sns.boxplot(x='Year' , y='Age', data=athlete_total_df, hue='Sex')
plt.title('The Age of Summer Athletes since 1956 by Gender', 
            fontdict=title_dict, pad=15)
ax.set_yticks(range(10,76,5))
plt.subplots_adjust(top=0.9, left=0.08, right=0.95)
plt.savefig('./images/graph/athlete_age_boxplot.png')
plt.show()

# Violin plot - BMI
plt.figure(figsize=year_fig)
ax = plt.subplot()
sns.violinplot(x='Year', y='BMI', data=athlete_total_df, 
                hue='Sex', split=True, inner='quartile')
plt.title('The BMI of Summer Athletes since 1956 by Gender', 
            fontdict=title_dict, pad=15)
ax.set_yticks(range(5,68, 5))
plt.subplots_adjust(top=0.9, left=0.08, right=0.95)
plt.savefig('./images/graph/athlete_bmi_violinplot.png')
plt.show()

# Barplot - Number of events per athlete
plt.figure(figsize=year_fig)
ax = plt.subplot()
sns.barplot(x='Year', y='Event', data=athlete_total_df, hue='Sex')
ax.set_yticks(np.arange(1,4.1,0.25))
plt.title('The average number of events per athlete since 1956 by gender', 
            fontdict=title_dict, pad=15)
plt.ylabel('Number of events')
plt.subplots_adjust(top=0.9, left=0.08, right=0.95)
plt.savefig('./images/graph/athlete_event_barplot.png', 
                fontdict=title_dict, pad=15)
plt.show()

# Pointplot - Number of medals per athlete
plt.figure(figsize=year_fig)
ax = plt.subplot()
sns.pointplot(x='Year', y='Medal', data=athlete_total_df, hue='Sex')
ax.set_yticks(np.arange(0.1, 0.71, 0.05))
plt.title('The average number of medals per athlete since 1956 by gender', 
            fontdict=title_dict, pad=15)
plt.ylabel('Number of medals')
plt.subplots_adjust(top=0.9, left=0.08, right=0.95)
plt.savefig('./images/graph/athlete_medal_pointplot.png')
plt.show()

# QQplot - Difference of age and BMI for medal winners
athlete_var_list = [['Age', [10, 45], '(years)'], ['BMI', [15,35], '']]
medal_athlete = athlete_total_df\
                    [athlete_total_df['Winner']]
non_medal_athlete = athlete_total_df\
                    [athlete_total_df['Winner'] == False]
plt.figure(figsize=year_fig)
plot = [1, 2, 0]
for var in athlete_var_list:
    plot[2] += 1
    ax = plt.subplot(plot[0], plot[1], plot[2])
    medal_percentile = medal_athlete[var[0]].quantile(np.arange(0,1,0.01))
    non_medal_percentile = non_medal_athlete[var[0]]\
                            .quantile(np.arange(0,1,0.01))
    plt.scatter(medal_percentile, non_medal_percentile, color='C1')
    plt.scatter(medal_percentile[0.49], non_medal_percentile[0.49], 
                color='black', marker='s')
    plt.scatter(medal_percentile[0.24], non_medal_percentile[0.24], 
                color='black', marker='+')
    plt.scatter(medal_percentile[0.74], 
                non_medal_percentile[0.74], 
                color='black', marker='+')
    plt.plot(var[1],var[1], color='C0', linewidth=2, linestyle='dashed')
    plt.title("Difference in {var}".format(var=var[0]))
    plt.xlabel('{var} of Medal Winners {units}'
                .format(var=var[0], units=var[2]))
    plt.ylabel('{var} of Non Medal Athletes {units}'
                .format(var=var[0], units=var[2]))
    ax.set_xlim(left=var[1][0], right=var[1][1])
    ax.set_ylim(bottom=var[1][0], top=var[1][1])
    plt.legend(['Medal Winner = Non Medal Winner',
                'Percentiles', 'Median', 'Interquartile Range'])
plt.subplots_adjust(top=0.9, left=0.08, right=0.95)
plt.gcf().suptitle('Comparison of Physical Characteristics of Medal Winners and Athletes', 
            fontdict=title_dict)
plt.savefig('./images/graph/athlete_difference_qqplot.png')
plt.show()