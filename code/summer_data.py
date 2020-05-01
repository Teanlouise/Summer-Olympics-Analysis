import pandas as pd

all_df = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/all_data.csv', index_col=0)
print(all_df)


## LOOK AT OVERVIEW OF GAMES DATA
games_total_ath = all_df.groupby(['Games']).ID.count().reset_index()
games_total_ath.columns = ['Games', 'Total_ID'] 
games_athletes = all_df.groupby(['Games']).ID.nunique().reset_index()
games_athletes.columns = ['Games', 'Unique_ID'] 
games_events = all_df.groupby(['Games']).Event.nunique().reset_index()
games_sports = all_df.groupby(['Games']).Sport.nunique().reset_index()
games_medals = all_df.groupby(['Games']).Medal.count().reset_index()
games_countries = all_df.groupby(['Games']).NOC.nunique().reset_index()
games_BMI = all_df[~all_df['BMI'].isna()].groupby('Games', as_index=False).ID.count()
games_BMI.columns = ['Games', 'Num_BMI']

games_total_df = all_df[['Games', 'Season']]
games_total_df = games_total_df.drop_duplicates()
games_total_df = games_total_df.merge(games_total_ath, how='outer') \
                                .merge(games_athletes, how='outer')\
                                .merge(games_events, how='outer')\
                                .merge(games_sports, how='outer')\
                                .merge(games_medals, how='outer')\
                                .merge(games_countries, how='outer')\
                                .merge(games_BMI, how='outer')

games_total_df['Perc_BMI'] = round(games_total_df.Num_BMI / games_total_df.Total_ID, 2)

# UPDATE DATA FOR SUMMER ONLY FROM 1956

# 1. Remove winter
winter_df = all_df[all_df['Season'] == 'Winter']
all_df = all_df.drop(winter_df.index)
#checkpoint('REMOVE WINTER', all_df, True, winter_df)

#print(winter_df)
#print(all_df)

# 2. Remove years
years_df = all_df[all_df['Year'].isin(range(1896,1955))]
all_df = all_df.drop(years_df.index)
#checkpoint('REMOVE YEARS', all_df, True, years_df)

# 3. Remove irrelevant columns
extra_df = all_df
all_df = all_df.drop(["Season", 'Games'], axis=1) #add season, games
#checkpoint('REMOVE EXTRA', all_df, True, extra_df)


all_df.to_csv('F:/TEAN/Portfolio/olympics/data/summer_data.csv')



