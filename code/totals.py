import pandas as pd

# Read in data and split between summer and winter
all_df = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/summer_data.csv', index_col=0).reset_index(drop = True)

######## THE ATHLETE #######

# Athlete Totals (Games, Year, ID, Sex, Age, BMI, Season, NOC, #Events, #Medals)
# Summer only - Replace Games in groupby with Year
athlete_events = all_df.groupby(['Year', 'ID']).Event.count().reset_index()
athlete_medals = all_df.groupby(['Year', 'ID']).Medal.count().reset_index()
athlete_total_df = all_df[['Year', 'ID', 'Sex', 'Age', 'BMI', 'NOC']] #Remove season, games
athlete_total_df = athlete_total_df.drop_duplicates()
athlete_total_df = athlete_total_df.merge(athlete_events, how='outer') \
                                    .merge(athlete_medals, how='outer')
athlete_total_df['Winner'] = athlete_total_df['Medal'] != 0
athlete_total_df['Medal_Perc'] = round((athlete_total_df.Medal / athlete_total_df.Event), 2)

print(athlete_total_df)
athlete_total_df.to_csv('F:/TEAN/Portfolio/olympics/data/athlete_total.csv')

######## THE GAMES #######

# Games totals (Games, Year, #Athletes, #Medals, #Male, #Female, #Events)
# Summer only - Replace Games in groupby with Year
games_total_ath = all_df.groupby(['Year']).ID.count().reset_index()
games_total_ath.columns = ['Year', 'Entries'] 
games_athletes = all_df.groupby(['Year']).ID.nunique().reset_index()
games_athletes.columns = ['Year', 'Athletes'] 
games_events = all_df.groupby(['Year']).Event.nunique().reset_index()
games_sports = all_df.groupby(['Year']).Sport.nunique().reset_index()
games_medals = all_df.groupby(['Year']).Medal.count().reset_index()
games_countries = all_df.groupby(['Year']).NOC.nunique().reset_index()
games_male = all_df[all_df['Sex'] == 'M'].groupby('Year').ID.nunique().reset_index()
games_male.columns = ['Year', 'Male']    
games_female = all_df[all_df['Sex'] == 'F'].groupby('Year').ID.nunique().reset_index()
games_female.columns = ['Year', 'Female']

games_host = all_df[all_df['NOC'] == all_df['Host_NOC']].groupby('Year').Medal.count().reset_index()
games_host.columns = ['Year', 'Host_Medal']
games_visitor = all_df[all_df['NOC'] != all_df['Host_NOC']].groupby('Year').Medal.count().reset_index()
games_visitor.columns = ['Year', 'Visitor_Medal']
games_total_df = all_df[['Year', 'Host_NOC']] #Remove season, games
games_total_df = games_total_df.drop_duplicates()
games_total_df = games_total_df.merge(games_total_ath, how='outer') \
                                .merge(games_athletes, how='outer')\
                                .merge(games_events, how='outer')\
                                .merge(games_sports, how='outer')\
                                .merge(games_medals, how='outer')\
                                .merge(games_countries, how='outer')\
                                .merge(games_host, how='outer')\
                                .merge(games_visitor, how='outer')\
                                .merge(games_male, how='outer')\
                               .merge(games_female, how='outer')

games_total_df['Host_Medal_Perc'] = round((games_total_df.Host_Medal / games_total_df.Medal)*100, 2)
games_total_df['Visitor_Medal_Perc'] = round((games_total_df.Visitor_Medal / games_total_df.Medal)*100, 2)

print(games_total_df)
games_total_df.to_csv('F:/TEAN/Portfolio/olympics/data/games_total.csv')


## COUNTRIES TOTAL
noc_total_ath = all_df.groupby(['Year', 'NOC']).ID.count().reset_index()
noc_total_ath.columns = ['Year', 'NOC', 'Entries'] 
noc_athletes = all_df.groupby(['Year', 'NOC']).ID.nunique().reset_index()
noc_athletes.columns = ['Year', 'NOC', 'Athletes'] 
noc_events = all_df.groupby(['Year', 'NOC']).Event.nunique().reset_index()
noc_medals = all_df.groupby(['Year', 'NOC']).Medal.count().reset_index()
noc_male = all_df[all_df['Sex'] == 'M'].groupby(['Year', 'NOC']).ID.nunique().reset_index()
noc_male.columns = ['Year', 'NOC', 'Male']    
noc_female = all_df[all_df['Sex'] == 'F'].groupby(['Year', 'NOC']).ID.nunique().reset_index()
noc_female.columns = ['Year', 'NOC', 'Female']
games_medals = games_total_df[['Year', 'Medal']]
games_medals.columns = ['Year', 'Games_Medals']
games_athletes = games_total_df[['Year', 'Entries']]
games_athletes.columns = ['Year', 'Games_Entries']

noc_total_df = all_df[['Year', 'NOC', 'Country', 'Host_Country']] #remove season, games
noc_total_df = noc_total_df.drop_duplicates()
noc_total_df['Host']  = noc_total_df['Country'] == noc_total_df['Host_Country']
noc_total_df = noc_total_df.merge(noc_total_ath, how='outer') \
                            .merge(noc_athletes, how='outer') \
                            .merge(noc_events, how='outer') \
                            .merge(noc_medals, how='outer') \
                            .merge(noc_male, how='outer') \
                            .merge(noc_female, how='outer') \
                            .merge(games_medals, how='outer') \
                            .merge(games_athletes, how='outer')
noc_total_df['Unique_Perc'] = round((noc_total_df.Entries / noc_total_df.Athletes), 2)
noc_total_df['Medal_Perc'] = round((noc_total_df.Medal / noc_total_df.Entries), 2)
noc_total_df['Games_Medal_Perc'] = round((noc_total_df.Medal / noc_total_df.Games_Medals)*100, 2)
noc_total_df['Games_Entries_Perc'] = round((noc_total_df.Entries / noc_total_df.Games_Entries)*100, 2)

top_20 = noc_total_df.groupby(['NOC'], as_index=False)['Medal'].sum().sort_values(by='Medal', ascending=False).head(20).NOC.tolist()
noc_total_df['Top_20'] = noc_total_df['NOC'].isin(top_20)
noc_total_df['Top_10'] = noc_total_df['NOC'].isin(top_20[:10])


print(noc_total_df)
noc_total_df.to_csv('F:/TEAN/Portfolio/olympics/data/noc_total.csv')



