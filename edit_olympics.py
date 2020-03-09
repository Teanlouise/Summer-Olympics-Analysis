import pandas
import csv

############################ READ FILES ############################

# Read athlete event info file as dataframe (Guru99)
athlete_df = pandas.read_csv('athlete_events.csv')

# Create dictionary for host cities
host_list = athlete_df.City.unique()
host_dict = {i : None for i in host_list}
# Edit dictionary manually with NOC values
host_dict = {
    'Barcelona': 'ESP',
    'London': "GBR",
    'Antwerpen': "BEL",
    'Paris': "FRA",
    'Calgary': "CAN",
    'Albertville': "FRA",
    'Lillehammer': "NOR",
    'Los Angeles': "USA",
    'Salt Lake City': "USA",
    'Helsinki': "FIN",
    'Lake Placid': "USA",
    'Sydney': "AUS",
    'Atlanta': "USA",
    'Stockholm': "SWE",
    'Sochi': "RUS",
    'Nagano': "JPN",
    'Torino': "ITA",
    'Beijing': "CHN",
    'Rio de Janeiro': "BRA",
    'Athina': "GRE",
    'Squaw Valley': "USA",
    'Innsbruck': "AUT",
    'Sarajevo': "YUG",
    'Mexico City': "MEX",
    'Munich': "GER",
    'Seoul': "KOR",
    'Berlin': "GER",
    'Oslo': "NOR",
    "Cortina d'Ampezzo": "ITA",
    'Melbourne': "AUS",
    'Roma': "ITA",
    'Amsterdam': "NED",
    'Montreal': "CAN",
    'Moskva': "URS",
    'Tokyo': "JPN",
    'Vancouver': "CAN",
    'Grenoble': "FRA",
    'Sapporo': "JPN",
    'Chamonix': "FRA",
    'St. Louis': "USA",
    'Sankt Moritz': "SUI",
    'Garmisch-Partenkirchen': "GER"}

# Add host country NOC (GeeksForGeeks, Kanoki)
athlete_df["Host"] = athlete_df["City"].map(host_dict)

# Remove columns (Shanelynn)
athlete_df = athlete_df.drop(["Name", "Team", "Event", "Games", "City"], axis=1)




# Add Population
pop_df = pandas.read_csv('worldbank_pop.csv')
#pop_df = pandas.merge(athlete_df, pop_df[['Country Code', 'Year']], on='Country Code')
print(pop_df)


############################ WRITE FILES ############################


############ ALL ############
# CSV
all_csv = athlete_df.to_csv('data_all.csv', index=False)

# COUNTRY - COUNT
country_df = athlete_df.groupby('NOC')['Year']\
    .value_counts().unstack().fillna(0)
country_df.head()

# FEMALE - COUNT
female_df = athlete_df[athlete_df['Sex'] == 'F']\
    .groupby('NOC')['Year']\
    .value_counts().unstack().fillna(0)
female_df.head()

# MALE - COUNT
male_df = athlete_df[athlete_df['Sex'] == 'M']\
    .groupby('NOC')['Year']\
    .value_counts().unstack().fillna(0)
male_df.head()

############ SUMMER ############

# CSV
summer_df = athlete_df[athlete_df['Season'] == 'Summer']
summer_csv = summer_df.to_csv('data_summer.csv', index=False)

# COUNTRY - COUNT
country_df = summer_df.groupby('NOC')['Year'].value_counts().unstack().fillna(0)
country_df.head()

# NO MEDALS - COUNT
no_medals = summer_df[(summer_df['Medal'] != 'Gold')
                      & (summer_df['Medal'] != 'Silver')
                      & (summer_df['Medal'] != 'Bronze')]\
                     .groupby('NOC')['Year']\
                     .value_counts().unstack().fillna(0)
no_medals.head()


# MEDALS - COUNT
medals = summer_df[(summer_df['Medal'] == 'Gold')
                   | (summer_df['Medal'] == 'Silver')
                   | (summer_df['Medal'] == 'Bronze')]\
                    .groupby('NOC')['Year']\
                    .value_counts().unstack().fillna(0)
medals.head()