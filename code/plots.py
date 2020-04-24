import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sns_qqplot import qqplot
import statsmodels.api as sm  
from statsmodels.graphics.gofplots import qqplot_2samples


def single_int_plots(var, all_df, winter_df, summer_df):
    # HISTOGRAM    
    plt.figure()
    ax = plt.subplot(2,2,1)
    olympic_hist(var, all_df, winter_df, summer_df, ax)        

    # BOXPLOT   
    plt.subplot(2,2,3) 
    single_boxplot(all_df, 'Season', var, 'Winner', 'Athletes', 'All')
    plt.subplot(2,2,4)
    single_boxplot(all_df, 'Season', var, 'Medal', 'Medal Winners', 'All')

    # QQPLOT
    #plt.subplot(2,2,2)
    medals_qqplot(var, winter_df, summer_df)



######## HISTOGRAM ####################

def olympic_hist(var, df1, df2, ax, label1, label2, plot_min, plot_max, bins):   
    plt.hist(df1[var], range=(plot_min, plot_max), bins=bins, label=label1, histtype='step')
    plt.hist(df2[var], range=(plot_min, plot_max), bins=bins, label=label2, histtype='step')
    ax.set_xticks(range(plot_min, plot_max, 5))
    plt.xlabel(var)
    plt.title('Distribution of {var}'.format(var=var)) 
    plt.legend()

def all_hist(var, all_df, winter_df, summer_df, male_df, female_df, medal_df, non_medal_df, plot):    
    row=plot[0]
    col=plot[1]
    pos=plot[2]    
    plot_min = int(all_df[var].min())
    plot_max = int(all_df[var].max())
    bins = int(plot_max - plot_min)

    pos+=1    
    ax = plt.subplot(row, col, pos)
    olympic_hist(var, winter_df, summer_df, ax, 'Winter', 'Summer', plot_min, plot_max, bins)    
    pos+=1 
    ax = plt.subplot(row, col, pos)
    olympic_hist(var, male_df, female_df, ax, 'Male', 'Female', plot_min, plot_max, bins)
    pos+=1
    ax = plt.subplot(row, col, pos)
    olympic_hist(var, medal_df, non_medal_df, ax, 'Medal', 'Non-Medal', plot_min, plot_max, bins)
    #plt.gcf().suptitle('Distribution of {var}'.format(var=var))
    return pos


def hist_kde(df, var, plot):
    stat = df[var]
    row=plot[0]
    col=plot[1]
    pos=plot[2]
    pos+=1
    plt.subplot(row, col, pos)
    sns.distplot(stat, rug=False, kde=False)
    plt.title('Hist')
    pos+=1
    plt.subplot(row, col, pos)
    sns.distplot(stat, rug=True, kde=False)
    plt.title('Hist with Rug')
    pos+=1
    plt.subplot(row, col, pos)
    sns.distplot(stat)
    plt.title('Hist with KDE')
    pos+=1
    plt.subplot(row, col, pos)
    sns.distplot(stat, rug=True, hist=False)
    pos+=1
    plt.title('KDE with Rug')
    plt.subplot(row, col, pos)    
    sns.kdeplot(stat, shade=True, label='All')
    plt.title('KDE')
    return pos

def hist_kde_hue(df1, df2, var, label1, label2, plot):
    stat1 = df1[var]
    stat2 = df2[var]
    row=plot[0]
    col=plot[1]
    pos=plot[2]
    pos+=1
    plt.subplot(row, col, pos)
    sns.distplot(stat1, rug=False, kde=False)
    sns.distplot(stat2, rug=False, kde=False)
    pos+=1
    plt.subplot(row, col, pos)
    sns.distplot(stat1, rug=True, kde=False)
    sns.distplot(stat2, rug=True, kde=False)
    pos+=1
    plt.subplot(row, col, pos)
    sns.distplot(stat1)
    sns.distplot(stat2)
    pos+=1
    plt.subplot(row, col, pos)
    sns.distplot(stat1, rug=True, hist=False)
    sns.distplot(stat2, rug=True, hist=False)
    pos+=1
    plt.subplot(row, col, pos)
    sns.kdeplot(stat1, shade=True, label=label1)
    sns.kdeplot(stat2, shade=True, label=label2)
    return pos

def all_hist_kde(var, all_df, winter_df, summer_df, male_df, female_df, medal_df, non_medal_df, plot):
    plt.figure()
    plot = [4,5,0]    
    plot[2] = hist_kde(all_df, var, plot)
    plot[2] = hist_kde_hue(winter_df, summer_df, var, 'Winter', 'Summer', plot)
    plot[2] = hist_kde_hue(male_df, female_df, var, 'Male', 'Female', plot)
    plot[2] = hist_kde_hue(medal_df, non_medal_df, var, 'Medal', 'Non-Medal', plot)
















######### BOXPLOT ################

def both_boxplot(y, all_df):
    sns.boxplot(data=all_df, x='Both', y=y)
    plt.title('{y} of Athletes competing both seasons'.format(y=y))

def single_boxplot(df, x, y, hue, title, season):
    sns.boxplot(data=df, x=x, y=y, hue=hue)
    plt.title('{y} of {season} {title} by {x}'.format(y=y, x=x, title=title, season=season))


def medals_boxplot(y, all_df, winter_df, summer_df):
    plt.figure()
    x = 'Season'
    plt.subplot(3,2,1)
    single_boxplot(all_df, x, y, 'Winner', 'Athletes', 'All')
    plt.subplot(3,2,2)
    single_boxplot(all_df, x, y, 'Medal', 'Medal Winners', 'All')

    x = 'Year'
    hue = 'Winner'
    title = 'Athletes'
    plt.subplot(3,2,3)
    single_boxplot(winter_df, x, y, hue, title, 'Winter')
    plt.subplot(3,2,5)
    single_boxplot(summer_df, x, y, hue, title, 'Summer')

    x = 'Year'
    hue = 'Medal'
    title = 'Medal Winners'
    plt.subplot(3,2,4)
    single_boxplot(winter_df, x, y, hue, title, 'Winter')
    plt.subplot(3,2,6)
    single_boxplot(summer_df, x, y, hue, title, 'Summer')
    
    

def boxplots_year(var, df):
    y = var

    # By Year
    x = 'Year'
    hue_list = ['Season', 'Sex', 'Winner', 'Medal']

    plt.figure()
    row = 5
    col = 1
    pos = 1

    plt.subplot(row, col, pos)
    sns.boxplot(data=df, x=x, y=y)
    plt.title('{y} each {x}'.format(y=y, x=x))

    for hue in hue_list:
        pos+=1
        plt.subplot(row, col, pos)
        sns.boxplot(data=df, x=x, y=y, hue=hue)
        plt.title('{y} each {x} by {hue}'.format(y=y, x=x, hue=hue))
    
def boxplots_hues(var, df):

    # By Season, Gender, Winner
    x_list = ['Season', 'Sex', 'Winner']
    hue_list = ['Season', 'Sex', 'Winner']

    plt.figure()
    row = 3
    col = 2
    pos = 1

    for hue in hue_list:
        for x in x_list:
            if hue != x:            
                plt.subplot(row, col, pos)
                sns.boxplot(data=df, x=x, y=var, hue=hue)
                plt.title('{y} each {x} by {hue}'.format(y=var, x=x, hue=hue))
                pos+=1
    

####### QQPLOT ############

def df_qqplot(df, season, var): 
    df = df.groupby(['Year', 'Winner'])[var].mean().reset_index()
    df = df.pivot(columns='Winner', index='Year', values=var).reset_index()
    df.columns = ['Year', 'Medal', 'Non-Medal']
    df['Season'] = season    
    return df

def medals_qqplot(var, winter_df, summer_df):
    df1 = df_qqplot(winter_df, "Winter", var)
    df2 = df_qqplot(summer_df, "Summer", var)
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
    plt.subplots_adjust(top=0.9)
    plt.title('Comparison of Medal and Non-Medal Athletes by Average {var} each year'.format(var=var))    





def qqplot_2(var, medal_df, non_medal_df, male_df, female_df, winter_df, summer_df):
    qqplot_2samples(medal_df[var], non_medal_df[var], xlabel='Medal', ylabel='Non-Medal', line='45')
    plt.title('{var} for Medal v. Non-Medal'.format(var=var))

    qqplot_2samples(female_df[var], male_df[var], xlabel='Male', ylabel='Female', line='45')
    plt.title('{var} for Female v. Male'.format(var=var))

    qqplot_2samples(winter_df[var], summer_df[var], xlabel='Winter', ylabel='Summer', line='45')
    plt.title('{var} for Winter v. Summer'.format(var=var))





#### PAIRWISE #######
def olympic_pairwise(df, hue, vars):
    sns.pairplot(df, hue=hue, kind='reg', vars=vars)
    plt.subplots_adjust(top=0.9)
    plt.gcf().suptitle('Number of {vars} by {hue}'.format(hue=hue, vars=vars))



def get_games_count_df(df, season):
    years = df.groupby(['Year'])
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
# games_winter_count_df = get_games_count_df(winter_df, "Winter")
# games_summer_count_df = get_games_count_df(summer_df, "Summer")
# games_count_df = pd.concat([games_winter_count_df, games_summer_count_df]).sort_values('Year').reset_index(drop=True)