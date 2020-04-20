import pandas as pd

def get_season_df(file_name, season):
    host_data = []
    with open(file_name) as file_var:
        for line in file_var.readlines():
                year = line[:5]
                location = line[6:-1].split(', ')
                city = location[0]
                country = location[1]
                host_data.append([year, city, country, season])
    
    season_df = pd.DataFrame(host_data, columns=['Year', 'City', 'Host_Country', 'Season'])          
    return season_df


# Read in the data for each seasons
summer_df = get_season_df('F:/TEAN/Portfolio/olympics/code/host_summer.txt', 'Summer')
winter_df = get_season_df('F:/TEAN/Portfolio/olympics/code/host_winter.txt', 'Winter')
# Combine to create 1 DF
host_df = pd.merge(summer_df, winter_df, how='outer')

#Check if all cities accounted for:
athlete_df = pd.read_csv('F:/TEAN/Portfolio/olympics/data/athlete_events.csv')
def check_city(season_df):
    for host_city in host_df.City.unique():
        for athlete_city in season_athlete_df.City.unique():
            if (host_city not in season_df.City.unique()) and (athlete_city not in season_df.City.unique()):
                print("Host City: ", host_city)
                print("Athlete City: ", athlete_city)

# Write to CSV
host_df.to_csv('F:/TEAN/Portfolio/olympics/data/host_countries.csv')