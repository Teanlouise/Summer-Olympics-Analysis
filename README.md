![readme_title](./readme_title.PNG)

This report is about the Modern Summer Olympics from 1956 to 2016. The aim is to investigate whether there is equal distribution and provide insight into the following questions:
- Does hosting at home increase chances of a medal?
- Does countries population/GDP impact chances?
- Is there a difference in physical characteristics for medallists?
- What is the difference between men/women?
- Is Olympics fair representation of talent?
- How has the athlete changed over time?

In addition to the report a presentation was given providing an overview. 
[![video](readme_video.png)](https://www.youtube.com/watch?v=delurBPtI74)

# DATA

## DATA SOURCES
- athlete_events.csv - 120 years of Olympic history - https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results 
![athlete_events](./images/data/athlete_events.PNG)
- noc_regions.csv - 120 years of Olympic history - https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results
![noc_regions](./images/data/noc_regions.PNG)
- gdp.csv - World Bank - https://data.worldbank.org/indicator/NY.GDP.MKTP.CD 
![worldbank_gdp](./images/data/worldbank_gdp.PNG)
- population.csv - World Bank - https://data.worldbank.org/indicator/NY.GDP.MKTP.CD
![worldbank_pop](./images/data/worldbank_pop.PNG)
- host_cities.csv - Manually made from https://architectureofthegames.net/olympic-host-cities/\
![host_countries](./images/data/host_countries.PNG)

## DATA PROCESSING 

### COMBINED DATASET
Start with athlete_events.csv:
1. Remove art competitions from sports
2. Remove NAME and TEAM
3. Make NOC consistent (SIN, RUS, TPE, CHN, GER, CZE, SRB)
4. Add column COUNTRY by matching with NOC in noc_regions.csv
5. Update host CITY to match common in host_cities.csv (Athens, Rome, Antwerp, Moscow, Turin, St Moritz)
6. Add column HOST_NOC by matching NOC from host_cities.csv
7. Add column BMI with weight(kg)/height^2(m)
8. Add column with Boolean whether medallist
9. Add column with GDP from gdp.csv
10. Add column POPULATION from pop.csv
11. Remove years 1896-1952
12. Remove winter season
13. Remove SEASON and GAMES columns
![all_data](./images/data/all_data.png)

### TOTALS
- games_total.csv - Totals for all each attributes each year
![games_total](./images/data/games_total.png)
- athlete_total.csv - Totals for each athlete (rather than each entry)
![athlete_total](./images/data/athlete_total.png)
- noc_total.csv - Totals for each country each year
![noc_total](./images/data/noc_total.png)
- host_difference.csv - The number and percentage of medals for host countries only

# DISCUSSION

## THE GAMES
![games_histogram](./images/graph/games_histogram.png)

## THE ATHLETES
![athlete_age_boxplot](./images/graph/athlete_age_boxplot.png)
![athlete_bmi_violinplot](./images/graph/athlete_bmi_violinplot.png)
![athlete_event_barplot](./images/graph/athlete_event_barplot.png)
![athlete_medal_pointplot](./images/graph/athlete_medal_pointplot.png)
![athlete_difference_qqplot](./images/graph/athlete_difference_qqplot.png)

## THE COUNTRIES
![countries_host_lmplot](./images/graph/countries_host_lmplot.png)
![countries_stats_heatmap](./images/graph/countries_stats_heatmap.png)
![countries_pop_gdp_3d](./images/graph/countries_pop_gdp_3d.png)

## REPRESENTATION
![countries_entryperc_swarm](./images/graph/countries_entryperc_swarm.png)
![countries_medals_stacked](./images/graph/countries_medals_stacked.png)
![countries_medals_resid](./images/graph/countries_medals_resid.png)