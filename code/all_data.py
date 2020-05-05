import pandas as pd
import numpy as np


# COMPARE SUBSETS OF DATA WITH MAIN AS CHANGES
def check_item_not_in(df1, df2):
    item_list = []
    count = 0
    if df1.nunique() != df2.nunique():
        for item in df1.unique():
            if item not in df2.unique():
                item_list.append(item)
                count+=1
    return item_list,count

# PRINT CHECKS OF HOW DATA CHANGING
def checkpoint(action, all, bool=False, lost=None):
    print("{action}: \n Unique NOC: {num_noc} \
        \n Unique Athletes: {num_athletes}."
        .format(action=action, 
                num_noc=all.NOC.nunique(), 
                num_athletes=all.ID.nunique()))

    if bool:
        print("\tLost NOC: {} \t Lost Athletes: {}" 
                .format(check_item_not_in(lost.NOC, all.NOC)[0],
                check_item_not_in(lost.ID, all.ID)[1]))

########### READ IN DATASETS ##########
noc_df = pd.read_csv(
    './data/noc_regions.csv')
host_df = pd.read_csv(
    './data/host_countries.csv')
athlete_df = pd.read_csv(
    './data/athlete_events.csv')
worldbank_gdp = pd.read_csv(
    './data/worldbank_gdp.csv', 
    index_col=0).reset_index(drop = True) 
worldbank_pop = pd.read_csv(
    './data/worldbank_pop.csv', 
    index_col=0).reset_index(drop = True) 

############### START ##############
all_df = athlete_df
#checkpoint('START', all_df)

# 1. Remove art competitions
art_df = all_df[all_df['Sport'] == 'Art Competitions']
all_df = all_df.drop(art_df.index)
#checkpoint('REMOVE ART', all_df, True, art_df)

# 2. Remove irrelevant columns
extra_df = all_df
all_df = all_df.drop(["Name", "Team"], axis=1)
#checkpoint('REMOVE EXTRA', all_df, True, extra_df)

# 3. Make NOC codes consistent for countries that have changed.
noc_unique = all_df.NOC.unique()
all_df.loc[(all_df.NOC == 'TCH'),'NOC'] = 'CZE'
all_df.loc[(all_df.NOC == 'SGP'),'NOC'] = 'SIN'
all_df.loc[(all_df.NOC == 'EUN') 
                | (all_df.NOC == 'URS'),'NOC'] = 'RUS'
all_df.loc[(all_df.NOC == 'FRG') 
                | (all_df.NOC == 'GDR'),'NOC'] = 'GER'
all_df.loc[(all_df.NOC == 'SCG') 
                | (all_df.NOC == 'YUG'),'NOC'] = 'SRB'
#checkpoint('UPDATE NOC', all_df)

# 4. Add Country Column to match NOC
all_df = all_df.merge(noc_df[['region', 'NOC']]
                .rename(columns={'region':'Country'})) \
                .reset_index(drop = True)
#checkpoint('ADD COUNTRY', all_df)

# 5. Update Host City names that don't match 
all_df.loc[(all_df.City == 'Athina'),'City'] = 'Athens'
all_df.loc[(all_df.City == 'Roma'),'City'] = 'Rome'
all_df.loc[(all_df.City == 'Antwerpen'),'City'] = 'Antwerp'
all_df.loc[(all_df.City == 'Moskva'),'City'] = 'Moscow'
all_df.loc[(all_df.City == 'Torino'),'City'] = 'Turin'
all_df.loc[(all_df.City == 'Sankt Moritz'),'City'] = 'St. Moritz'
#checkpoint('UPDATE HOST', all_df)

# 6. Add Host Country NOC
all_df = all_df.merge(host_df[['Host_Country', 
                                'Host_NOC', 
                                'City']]) \
                .sort_values("Year") \
                .reset_index(drop = True)
#checkpoint('ADD HOST COUNTRY', all_df)

# 7. Add BMI columns [Weight (kg) / Height^2 (m)]
all_df['BMI'] = all_df.apply(
    lambda x: round(x.Weight/((x.Height/100)**2), 2),
    axis=1)

# 8. Add boolean to mark who is a medal winner 
all_df['Winner'] = all_df.Medal.notna()


# 9. Add GDP
# Get GDP and merge with noc_total, divide by 1billion
worldbank_gdp = worldbank_gdp.drop(['Indicator Name', 
                                    'Indicator Code', 
                                    'Unnamed: 64'], 
                                    axis=1)
worldbank_gdp = worldbank_gdp.melt(id_vars="Country Code", 
                                    var_name="Year", 
                                    value_name="GDP")
worldbank_gdp.columns = (['NOC', 'Year', 'GDP'])
worldbank_gdp.sort_values('Year')
worldbank_gdp['Year'] = pd.to_numeric(worldbank_gdp.Year)
worldbank_gdp['GDP'] = round(worldbank_gdp['GDP']
                        .divide(1000000000), 2)

all_df = all_df.merge(worldbank_gdp, how='left')

# 10. Add Population
worldbank_pop = worldbank_pop.drop(['Indicator Name', 
                                    'Indicator Code', 
                                    'Unnamed: 64'], 
                                    axis=1)
worldbank_pop = worldbank_pop.melt(id_vars="Country Code", 
        var_name="Year", 
        value_name="Population")
worldbank_pop.columns = (['NOC', 'Year', 'Population'])
worldbank_pop.sort_values('Year')
worldbank_pop['Year'] = pd.to_numeric(worldbank_pop.Year)
worldbank_pop['Population'] = round(worldbank_pop['Population']
                                .divide(1000000), 2)
all_df = all_df.merge(worldbank_pop, how='left')

# Summer 1956 Olympics Equistrian events in Sweden 
# Update to reflect actual host Australia
all_df.loc[(all_df.Host_NOC == 'SWE'),'Host_NOC'] = 'AUS'
all_df.loc[(all_df.City == 'Stockholm'),'City'] = 'Melbourne'
all_df.loc[(all_df.Host_Country == 'Sweden'),
    'Host_Country'] = 'Australia'


#############TEST THE DATA###############
## LOOK AT OVERVIEW OF GAMES DATA
games_total_ath = all_df.groupby(['Games'])\
                        .ID.count().reset_index()
games_total_ath.columns = ['Games', 'Entries'] 
games_athletes = all_df.groupby(['Games'])\
                        .ID.nunique().reset_index()
games_athletes.columns = ['Games', 'Athletes'] 
games_events = all_df.groupby(['Games'])\
                    .Event.nunique().reset_index()
games_sports = all_df.groupby(['Games'])\
                    .Sport.nunique().reset_index()
games_medals = all_df.groupby(['Games'])\
                    .Medal.count().reset_index()
games_countries = all_df.groupby(['Games'])\
                    .NOC.nunique().reset_index()
games_male = all_df[all_df['Sex'] == 'M']\
                .groupby('Games')\
                .ID.nunique().reset_index()
games_male.columns = ['Games', 'Male']    
games_female = all_df[all_df['Sex'] == 'F']\
                .groupby('Games')\
                .ID.nunique().reset_index()
games_female.columns = ['Games', 'Female']
games_BMI = all_df[~all_df['BMI'].isna()]\
                .groupby('Games', as_index=False)\
                .ID.count()
games_BMI.columns = ['Games', 'Num_BMI']

games_host = all_df[all_df['NOC'] == all_df['Host_NOC']]\
                                        .groupby('Games')\
                                        .Medal.count().reset_index()
games_host.columns = ['Games', 'Host_Medal']
games_visitor = all_df[all_df['NOC'] != all_df['Host_NOC']]\
                    .groupby('Games')\
                    .Medal.count().reset_index()
games_visitor.columns = ['Games', 'Visitor_Medal']
games_total_df = all_df[['Games', 'Host_NOC', 'Season', 'Year']]
games_total_df = games_total_df.drop_duplicates()
games_total_df = games_total_df \
                    .merge(games_total_ath, how='outer') \
                    .merge(games_athletes, how='outer')\
                    .merge(games_events, how='outer')\
                    .merge(games_sports, how='outer')\
                    .merge(games_medals, how='outer')\
                    .merge(games_countries, how='outer')\
                    .merge(games_male, how='outer')\
                    .merge(games_female, how='outer')\
                    .merge(games_BMI, how='outer')
# Check the percentage of weight and height recorded
games_total_df['Perc_BMI'] = round(games_total_df.Num_BMI 
                                / games_total_df.Entries, 2)

# Write to file before further changes for pre 1956 data
games_total_df.to_csv('./data/games_total_before.csv')


######### UPDATE DATA FOR SUMMER ONLY FROM 1956 ##########

# 11. Remove winter
winter_df = all_df[all_df['Season'] == 'Winter']
all_df = all_df.drop(winter_df.index)
#checkpoint('REMOVE WINTER', all_df, True, winter_df)

# 12. Remove years
years_df = all_df[all_df['Year'].isin(range(1896,1955))]
all_df = all_df.drop(years_df.index)
#checkpoint('REMOVE YEARS', all_df, True, years_df)

# 13. Remove irrelevant columns
extra_df = all_df
all_df = all_df.drop(["Season", 'Games'], axis=1) 
#checkpoint('REMOVE EXTRA', all_df, True, extra_df)


######## WRITE TO FILE ##########
all_df.to_csv('./data/all_data.csv')
