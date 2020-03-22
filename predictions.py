import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import plotly.express as px


def clean_data(a):
    a = a.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], value_name='Cases', var_name='Date')
    a = a.set_index(['Country/Region', 'Province/State', 'Date'])
    return a


def max_cases(df, day):
    return df[['Province/State', 'Country/Region', day]].sort_values(by=day, ascending=True).tail(10)


def graph(x, z, day, place):
    infected_number = []
    for a, b in enumerate(x['Cases'].tail(z)):
        infected_number.append(b)
    plt.figure(figsize=(18, 8))
    plt.title('Confirmed COVID-19 cases in ' + place)
    plt.xlabel('Date')
    plt.xlim(0, 20)
    plt.ylabel('Cases ')
    for i in range(z):
        plt.text(x=i-0.05, y=infected_number[i]+5, s=infected_number[i], size=12, color='blue')
    plt.plot(day, infected_number, color='red', marker='o')
    plt.show()


def world_map(number, day):
    number = number.groupby(['Country/Region']).max()
    number = number.reset_index()
    number['size'] = number[day]
    fig = px.choropleth(number, locations='Country/Region', locationmode='country names', color=day, hover_name='Country/Region',
                     range_color=[0, 1000], projection='natural earth', title='COVID deaths all over the world')
    fig.show()


if __name__ == "__main__":
    pd.set_option("display.max_columns", 15)
    number_of_days = 20
    where = 'Poland'
    how_many_days_ago = 2

    confirmed_cases_dirty = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')
    deaths_dirty = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')
    recoveries_dirty = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')
    confirmed_cases = clean_data(confirmed_cases_dirty)
    deaths = clean_data(deaths_dirty)
    recoveries = clean_data(recoveries_dirty)
    date = confirmed_cases_dirty.columns[-number_of_days:]

    poland = confirmed_cases.loc['Poland', :]
    tunisia = confirmed_cases.loc['Tunisia', :]
    spain = confirmed_cases.loc['Spain', :]
    china = confirmed_cases.loc['China', 'Hubei']
    italy = confirmed_cases.loc['Italy', :]
    #graph(poland, number_of_days, date, where)


    date = datetime.datetime.now().date()
    yesterday = date - datetime.timedelta(how_many_days_ago)
    yesterday = yesterday.strftime('%m/%d/%y').lstrip("0")
    #print(max_cases(deaths_dirty, yesterday))

    china_deaths = deaths_dirty.loc[deaths_dirty['Country/Region'] == 'China']
    #print(max_cases(china_deaths, yesterday))

    world_map(deaths_dirty, yesterday)
