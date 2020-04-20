import pandas as pd

# ATHLETES
athlete_df = pd.read_csv('F:/TEAN/Portfolio/olympics/data/athlete_events.csv')
# NOC REGIONS
noc_df = pd.read_csv('F:/TEAN/Portfolio/olympics/data/noc_regions.csv')
# HOST CITIES
host_df = pd.read_csv('F:/TEAN/Portfolio/olympics/data/host_countries.csv')

# CHECK VALUES THROUGHOUT
def check_values(df1, df2):
    item_list = []
    for item in df1.unique():
        if item not in df2.unique():
            item_list.append(item)
    return item_list


print("START: There are {} unique countries and {} unique athletes."
.format(athlete_df.NOC.nunique(), athlete_df.ID.nunique()))

# 1. Remove art competitions
art = athlete_df[athlete_df['Sport'] == 'Art Competitions']
athlete_df = athlete_df.drop(art.index)
print("""
REMOVE ART
{unique_country} unique countries (- {lost_country})
{unique_athletes} unique athletes. (- {lost_athletes})
"""
.format(unique_country = athlete_df.NOC.nunique(), unique_athletes=athlete_df.ID.nunique(), lost_country=check_values(art.NOC, athlete_df.NOC), lost_athletes=art.ID.nunique()))


# 2. Remove irrelevant columns
athlete_df = athlete_df.drop(["Name", "Team", "Event", "Games", "Sport"], axis=1)
print("REMOVE EXTRA COLUMNS: There are {} unique countries and {} unique athletes."
.format(athlete_df.NOC.nunique(), athlete_df.ID.nunique()))

# 3. Remove years
athlete_df = athlete_df.drop(athlete_df[athlete_df['Year'].isin(range(1896,1921))].index)
print("REMOVE YEARS (-ANZ, BOH): There are {} unique countries and {} unique athletes."
.format(athlete_df.NOC.nunique(), athlete_df.ID.nunique()))

# 4. Replace redundant NOC with correct in athlete_events
# URS & EUN -> RUS / EUA & FRG & GDR -> GER / TCH & BOH -> CZE / SCG & YUG-> SRB / SGP -> SIN
athlete_df.loc[(athlete_df.NOC == 'EUN') | (athlete_df.NOC == 'URS'),'NOC'] = 'RUS'
athlete_df.loc[(athlete_df.NOC == 'EUA') | (athlete_df.NOC == 'FRG') | (athlete_df.NOC == 'GDR'),'NOC'] = 'GER'
athlete_df.loc[(athlete_df.NOC == 'TCH'),'NOC'] = 'CZE'
athlete_df.loc[(athlete_df.NOC == 'SCG') | (athlete_df.NOC == 'YUG'),'NOC'] = 'SRB'
athlete_df.loc[(athlete_df.NOC == 'SGP'),'NOC'] = 'SIN'
# Looks like changed already: ROC (1952-1976) -> TPE & ROC (1924-1948) -> CHN
print("Replace NOC: There are {} unique countries and {} unique athletes."
.format(athlete_df.NOC.nunique(), athlete_df.ID.nunique()))

# 5. Add Country Column to match NOC
athlete_df = athlete_df.merge(noc_df[['region', 'NOC']].rename(columns={'region':'Country'})).reset_index(drop = True)
print("Add Country: There are {} unique countries and {} unique athletes."
.format(athlete_df.NOC.nunique(), athlete_df.ID.nunique()))

# 6. Update Host City names that don't match 
# Athens -> Athina / Rome -> Roma / Antwerp -> Antwerpen / Moscow -> Moskva / Turin -> Torino / St Moritz -> Sankt Moritz
athlete_df.loc[(athlete_df.City == 'Athina'),'City'] = 'Athens'
athlete_df.loc[(athlete_df.City == 'Roma'),'City'] = 'Rome'
athlete_df.loc[(athlete_df.City == 'Antwerpen'),'City'] = 'Antwerp'
athlete_df.loc[(athlete_df.City == 'Moskva'),'City'] = 'Moscow'
athlete_df.loc[(athlete_df.City == 'Torino'),'City'] = 'Turin'
athlete_df.loc[(athlete_df.City == 'Sankt Moritz'),'City'] = 'St. Moritz'

# 7. Add Host Country
athlete_df = athlete_df.merge(host_df[['Host_Country', 'City']]).sort_values("Year").reset_index(drop = True)
print("Add Host: There are {} unique countries and {} unique athletes."
.format(athlete_df.NOC.nunique(), athlete_df.ID.nunique()))










# # Athletes with same ID in both winter and summer
# count = 0
# for athlete in summer_df.ID.unique():
#     if athlete in winter_df.ID.unique():
#         count += 1
# print(count)
