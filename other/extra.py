import pandas as pd


# Read file
all_df = pd.read_csv('F:/TEAN/Portfolio/olympics/athlete_events.csv', index_col=0, engine='python')
all_df.head()





############################ EXTRA ############################

# # Get NOC regions
# noc_df = pandas.read_csv('noc_regions.csv')\
#     .rename(columns={"NOC": "NOC", "region": "Country"})


# # Read host city file as dataframe, remove whitespace
#
# host_df = pandas.read_csv('olym.csv', encoding = 'unicode_escape')\
#     .drop(["Summer (Olympiad)", "Winter", "latitude", "longitude", "Unnamed: 7", "Unnamed: 8"], axis=1)
# host_df.Country = host_df.Country.str.lstrip()
# host_df = pandas.merge(host_df, noc_df[['NOC', 'Country']], on='Country')
# print(host_df)


# with open('athlete_events.csv', mode='r') as athlete_csv_file:
#     athlete_csv_reader = csv.DictReader(athlete_csv_file)
#     #athlete_csv_writer = csv.writer(athlete_csv_file)
#
#     fieldnames = ['ID', 'Sex', 'Age', 'Height', 'Weight', 'NOC', 'Year', 'Season', 'Sport', 'Medal', 'Host City']
#     writer = csv.DictWriter(athlete_csv_file, fieldnames=fieldnames)
#     writer.writeheader()
#
#     line_count = 0
#     for athlete in athlete_csv_reader:
#         # Add host country and remove city
#
#         athlete["Host"] = host_cities.get(athlete["City"])
#
#         # Remove Name, Team, Event, Games,
#         athlete.pop("Name", None)
#         athlete.pop("Team", None)
#         athlete.pop("Event", None)
#         athlete.pop("City", None)
#
#         writer.writerow(athlete)#
#         #print(athlete)
#
#         # if line_count == 0:
#         #     print(f'ID |\t Sex |\t Age |\t Height |\t Weight |\t NOC |\t Year |\t Season |\t Sport |\t Medal |\t Host City')
#         #     print('------------------------------------------------------------------------------------------------------------------------------------------------')
#         #     line_count += 1
#         # print(f'{athlete["ID"]} |\t {athlete["Sex"]} |\t {athlete["Age"]} |\t {athlete["Height"]} |\t {athlete["Weight"]} |\t {athlete["NOC"]} |\t {athlete["Year"]} |\t {athlete["Season"]} |\t {athlete["Sport"]} |\t {athlete["Medal"]} |\t {athlete["Host"]}')
#         # line_count += 1


# # Create NOC regions dictionary
# with open('noc_regions.csv', mode='r') as noc_csv_file:
#     noc_csv_reader = csv.DictReader(noc_csv_file)
#     # Print NOc regions
#     line_count = 0
#     for row in noc_csv_reader:
#         if line_count == 0:
#             print(f'NOC \t|\t Region')
#             print('-------------------------')
#             line_count += 1
#         print(f'{row["NOC"]} \t|\t {row["region"]}, {row["notes"]}')
#         line_count += 1
#     print(f'There are {line_count} NOC regions.')