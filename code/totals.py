import pandas as pd

# Read in summer data only
all_df = pd.read_csv('./data/summer_data.csv', index_col=0)\
            .reset_index(drop = True)

######## THE ATHLETE #######
# Athlete Totals (Year, ID, Sex, Age, BMI, NOC, #Events, #Medals)
athlete_events = all_df.groupby(['Year', 'ID'])\
                    .Event.count().reset_index()
athlete_medals = all_df.groupby(['Year', 'ID'])\
                    .Medal.count().reset_index()
athlete_total_df = all_df[['Year', 'ID', 'Sex', 
                            'Age','BMI', 'NOC']] 
athlete_total_df = athlete_total_df.drop_duplicates()
athlete_total_df = athlete_total_df \
                    .merge(athlete_events, how='outer') \
                    .merge(athlete_medals, how='outer')
athlete_total_df['Winner'] = athlete_total_df['Medal'] != 0
athlete_total_df['Medal_Perc'] = round((athlete_total_df.Medal 
                                    / athlete_total_df.Event), 2)
athlete_total_df['Sex'] = athlete_total_df.Sex.replace('M', 'Male')
athlete_total_df['Sex'] = athlete_total_df.Sex.replace('F', 'Female')
athlete_total_df.to_csv('./data/athlete_total.csv')

######## THE GAMES #######
# Games totals (Year, #Athletes, #Medals, #Male, #Female, #Events)
games_total_ath = all_df.groupby(['Year']).ID.count().reset_index()
games_total_ath.columns = ['Year', 'Entries'] 
games_athletes = all_df.groupby(['Year']).ID.nunique().reset_index()
games_athletes.columns = ['Year', 'Athletes'] 
games_events = all_df.groupby(['Year']).Event.nunique().reset_index()
games_sports = all_df.groupby(['Year']).Sport.nunique().reset_index()
games_medals = all_df.groupby(['Year']).Medal.count().reset_index()
games_countries = all_df.groupby(['Year']).NOC.nunique().reset_index()
games_male = all_df[all_df['Sex'] == 'M'].groupby('Year')\
                .ID.nunique().reset_index()
games_male.columns = ['Year', 'Male']    
games_female = all_df[all_df['Sex'] == 'F']\
                    .groupby('Year').ID.nunique().reset_index()
games_female.columns = ['Year', 'Female']
# Add column for number of medals awarded to host country
games_host = all_df[all_df['NOC'] == all_df['Host_NOC']]\
                .groupby('Year').Medal.count().reset_index()
games_host.columns = ['Year', 'Host_Medal']
games_visitor = all_df[all_df['NOC'] != all_df['Host_NOC']]\
                .groupby('Year').Medal.count().reset_index()
games_visitor.columns = ['Year', 'Visitor_Medal']
# Merge seperate together
games_total_df = all_df[['Year', 'Host_NOC']]
games_total_df = games_total_df.drop_duplicates()
games_total_df = games_total_df\
                    .merge(games_total_ath, how='outer') \
                    .merge(games_athletes, how='outer')\
                    .merge(games_events, how='outer')\
                    .merge(games_sports, how='outer')\
                    .merge(games_medals, how='outer')\
                    .merge(games_countries, how='outer')\
                    .merge(games_host, how='outer')\
                    .merge(games_visitor, how='outer')\
                    .merge(games_male, how='outer')\
                    .merge(games_female, how='outer')
# Add percentage of medals awarded to host and visitors
games_total_df['Host_Medal_Perc'] = round((games_total_df.Host_Medal 
                                    / games_total_df.Medal)*100, 2)
games_total_df['Visitor_Medal_Perc'] = round(
                                    (games_total_df.Visitor_Medal 
                                    / games_total_df.Medal)*100, 2)
games_total_df.to_csv('./data/games_total.csv')

######## THE COUNTRIES #######
# Group all by Year and NOC to create seperate tallies
noc_total_ath = all_df.groupby(['Year', 'NOC']).ID.count().reset_index()
noc_total_ath.columns = ['Year', 'NOC', 'Entries'] 
noc_athletes = all_df.groupby(['Year', 'NOC']).ID.nunique().reset_index()
noc_athletes.columns = ['Year', 'NOC', 'Athletes'] 
noc_events = all_df.groupby(['Year', 'NOC']).Event.nunique().reset_index()
noc_medals = all_df.groupby(['Year', 'NOC']).Medal.count().reset_index()
noc_male = all_df[all_df['Sex'] == 'M']\
                .groupby(['Year', 'NOC']).ID.nunique().reset_index()
noc_male.columns = ['Year', 'NOC', 'Male']    
noc_female = all_df[all_df['Sex'] == 'F']\
                .groupby(['Year', 'NOC']).ID.nunique().reset_index()
noc_female.columns = ['Year', 'NOC', 'Female']
# Add the number of medals and entries from each games
games_medals = games_total_df[['Year', 'Medal']]
games_medals.columns = ['Year', 'Games_Medals']
games_athletes = games_total_df[['Year', 'Entries']]
games_athletes.columns = ['Year', 'Games_Entries']
# Merge all seperate together
noc_total_df = all_df[['Year', 'NOC', 'Country', 
                        'Host_Country', 'GDP', 'Population']] 
noc_total_df = noc_total_df.drop_duplicates()
noc_total_df['Host']  = noc_total_df['Country'] \
                            == noc_total_df['Host_Country']
noc_total_df = noc_total_df\
                .merge(noc_total_ath, how='outer') \
                .merge(noc_athletes, how='outer') \
                .merge(noc_events, how='outer') \
                .merge(noc_medals, how='outer') \
                .merge(noc_male, how='outer') \
                .merge(noc_female, how='outer') \
                .merge(games_medals, how='outer') \
                .merge(games_athletes, how='outer')
# Add percentage calculations of how noc did at games
noc_total_df['Unique_Perc'] = round((noc_total_df.Entries 
                                / noc_total_df.Athletes), 2)
noc_total_df['Medal_Perc'] = round((noc_total_df.Medal 
                                / noc_total_df.Entries), 2)
noc_total_df['Games_Medal_Perc'] = round((noc_total_df.Medal 
                            / noc_total_df.Games_Medals)*100, 2)
noc_total_df['Games_Entries_Perc'] = round((noc_total_df.Entries 
                            / noc_total_df.Games_Entries)*100, 2)
# Add column denoting whether they were in top 20 or top 10
top_20 = noc_total_df.groupby(['NOC'], as_index=False)['Medal']\
                        .sum().sort_values(by='Medal', ascending=False)\
                        .head(20).NOC.tolist()
noc_total_df['Top_20'] = noc_total_df['NOC'].isin(top_20)
noc_total_df['Top_10'] = noc_total_df['NOC'].isin(top_20[:10])
noc_total_df.to_csv('F:/TEAN/Portfolio/olympics/data/noc_total.csv')


#### Host Medal Percentage vs. Average Percentage ####
# Create host percentage and regular percentage df
medals_all = all_df.groupby(['Year', 'NOC', 'Country'])\
                    .Medal.count().reset_index()
medals_all.columns=['Year','NOC', 'Country', 'Total_Medal']
medals_all = medals_all.merge(games_total_df[['Year', 'Medal']], 
                                how='outer')
medals_all['Medal_Perc'] = round((medals_all.Total_Medal 
                            / medals_all.Medal)*100, 2)
medals_all = round(medals_all.groupby(['NOC', 'Country'])
                    .Medal_Perc.mean(),2).reset_index()
medals_all.columns=['NOC', 'Country', 'Medal_Perc'] 
host_medals = games_total_df[['Year', 'Host_NOC', 
                            'Host_Medal_Perc']]
host_medals.columns=['Year', 'NOC', 'Host_Medal_Perc'] 
host_difference = pd.merge(host_medals, medals_all, how='left')
host_difference.to_csv('F:/TEAN/Portfolio/olympics/data/host_difference.csv')