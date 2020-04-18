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
    plt.xlim(0, z)
    plt.ylabel('Cases')
    plt.xticks(np.arange(1, z, 2.0))
    plt.grid(ls='dotted')
    for i in range(z):
        plt.text(x=i-0.1, y=infected_number[i]+5, s=int(infected_number[i]), size=12, color='blue')
    plt.plot(day, infected_number, color='red', marker='o')
    plt.show()
    plt.close()
    plt.clf()


def merged_graph(data, z,  place, place_2):
    maximum_cases = max(max(merged['Cases_x']), max(merged['Cases_y']))
    data = data.tail(z)
    data = data.rename({'Cases_x': place, 'Cases_y': place_2}, axis=1)
    plt.close()
    data.plot(y=[place, place_2], figsize=(18, 8), marker='o', title=('Confirmed COVID-19 cases in ' + place + ' and ' + place_2))
    plt.xlabel('Date')
    plt.xlim(0, z)
    plt.ylabel('Cases')
    plt.yticks(np.arange(0, maximum_cases, 10000))
    plt.grid(ls='dotted')
    plt.show()


def world_map(number, day):
    number = number.groupby(['Country/Region']).max()
    number = number.reset_index()
    number['size'] = number[day]
    fig = px.choropleth(number, locations='Country/Region', locationmode='country names', color=day, hover_name='Country/Region',
                     range_color=[0, 1500], projection='natural earth', title='COVID deaths all over the world',
                     color_continuous_scale=px.colors.sequential.Viridis)
    fig.show()


if __name__ == "__main__":
    pd.set_option("display.max_columns", 15)
    number_of_days = 35
    where = 'Poland'
    where_2 = 'Germany'
    how_many_days_ago = 1

    confirmed_cases_dirty = pd.read_csv('https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv&filename=time_series_covid19_confirmed_global.csv')
    deaths_dirty = pd.read_csv('https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_deaths_global.csv&filename=time_series_covid19_deaths_global.csv')
    confirmed_cases = clean_data(confirmed_cases_dirty)
    deaths = clean_data(deaths_dirty)
    date = confirmed_cases_dirty.columns[-number_of_days:]

    poland_cases = confirmed_cases.loc['Poland', :]
    spain_cases = confirmed_cases.loc['Spain', :]
    china_cases = confirmed_cases.loc['China', 'Hubei']
    italy_cases = confirmed_cases.loc['Italy', :]
    belarus_cases = confirmed_cases.loc['Belarus', :]
    germany_cases = confirmed_cases.loc['Germany', :]
    china_deaths = deaths_dirty.loc[deaths_dirty['Country/Region'] == 'China']
    merged = poland_cases.merge(germany_cases, left_on="Date", right_on="Date")

    now = datetime.datetime.now().date()
    exact_day = now - datetime.timedelta(how_many_days_ago)
    exact_day = exact_day.strftime('%m/%#d/%y').lstrip("0")

    graph(poland_cases, number_of_days, date, where)
    merged_graph(merged, number_of_days, where, where_2)
    world_map(deaths_dirty, exact_day)
