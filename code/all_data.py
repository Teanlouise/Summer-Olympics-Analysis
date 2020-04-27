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

# START
noc_df = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/noc_regions.csv')
host_df = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/host_countries.csv')
athlete_df = pd.read_csv(
    'F:/TEAN/Portfolio/olympics/data/athlete_events.csv')

# START
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

# 3. Remove years
years_df = all_df[all_df['Year'].isin(range(1896,1921))]
all_df = all_df.drop(years_df.index)
#checkpoint('REMOVE YEARS', all_df, True, years_df)

# 4. Make NOC codes consistent for countries that have changed.
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

# 5. Add Country Column to match NOC
all_df = all_df.merge(noc_df[['region', 'NOC']]
                        .rename(columns={'region':'Country'})) \
                        .reset_index(drop = True)
#checkpoint('ADD COUNTRY', all_df)

# 6. Update Host City names that don't match 
all_df.loc[(all_df.City == 'Athina'),'City'] = 'Athens'
all_df.loc[(all_df.City == 'Roma'),'City'] = 'Rome'
all_df.loc[(all_df.City == 'Antwerpen'),'City'] = 'Antwerp'
all_df.loc[(all_df.City == 'Moskva'),'City'] = 'Moscow'
all_df.loc[(all_df.City == 'Torino'),'City'] = 'Turin'
all_df.loc[(all_df.City == 'Sankt Moritz'),'City'] = 'St. Moritz'
#checkpoint('UPDATE HOST', all_df)

# 7. Add Host Country NOC
all_df = all_df.merge(host_df[['Host_Country', 'Host_NOC', 'City']]) \
                                .sort_values("Year") \
                                .reset_index(drop = True)
#checkpoint('ADD HOST COUNTRY', all_df)




# 8. Add GDP
# 9. Add Population

# WRITE TO FILE
all_df.to_csv('F:/TEAN/Portfolio/olympics/data/all_data.csv')
