import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def clean_data(a):
    a = a.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], value_name='Cases', var_name='Date')
    a = a.set_index(['Country/Region', 'Province/State', 'Date'])
    return a


def graph(x, z, day, place):
    infected_number = []
    for a, b in enumerate(x['Cases'].tail(z)):
        infected_number.append(b)
    plt.figure(figsize=(18, 8))
    plt.title('Confirmed COVID-19 cases in ' + place)
    plt.xlabel('Date')
    plt.ylabel('Cases ')
    for i in range(z):
        plt.text(x=i-0.05, y=infected_number[i]+0.3, s=infected_number[i], size=12, color='blue')
    plt.plot(day, infected_number, color='red', marker='o')
    plt.show()


if __name__ == "__main__":
    pd.set_option("display.max_columns", 15)
    number_of_days = 20
    where = 'Poland'

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
    graph(poland, number_of_days, date, where)
