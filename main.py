import pandas as pd
import streamlit as st
import plotly.express as px
from enum import Enum
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from math import sqrt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from scipy import stats
from scipy.stats import chi2_contingency

# df1 = pd.read_csv("clean/normalized_data/231_december_upcoming_lanes.csv")
# df = pd.read_csv("clean/normalized_data/237_march_upcoming_lanes.csv")

class Grouped(Enum):
    hour = "Час"
    week = "Неделя"

class Groupedx(Enum):
    car = "Автомобилей"
    speed = "Ср.скорость, км/ч"
    

def get_clean_data(df1_q = None, df_q = None) -> pd.DataFrame:
    if df1_q is not None  or df_q is not None:
        df1 = pd.read_csv("data_new/230 old december.csv")
        df = pd.read_csv("data_new/230 old march.csv")
    else:
        df1 = pd.read_csv("clean/normalized_data/231_december_upcoming_lanes.csv")
        df = pd.read_csv("clean/normalized_data/237_march_upcoming_lanes.csv")
    df1_just_numb = df1.loc[df1.index % 5 == 0]
    df1_just_numb = df1_just_numb.iloc[:5311]
    df_just_numb = df.loc[df.index % 5 == 0]
    return df_just_numb, df1_just_numb


def get_grouped_by(dataframe:pd.DataFrame, sort_by:Grouped, sort_by_x:Groupedx):
    dataframe['Время'] = pd.to_datetime(dataframe['Время'])
    if sort_by == Grouped.hour:
        # создание нового столбца с часами
        dataframe['Час'] = dataframe['Время'].dt.hour
        # группировка данных по часам и подсчет количества автомобилей
        if sort_by_x == Groupedx.car:
            grouped = dataframe.groupby('Час')['Автомобилей'].median()
        elif sort_by_x == Groupedx.speed:
            grouped = dataframe.groupby('Час')['Ср.скорость, км/ч'].median()
        return grouped
    elif sort_by == Grouped.week:
        dataframe['День недели'] = dataframe['Время'].dt.weekday
        if sort_by_x == Groupedx.car:
            grouped = dataframe.groupby('День недели')['Автомобилей'].median()
        elif sort_by_x == Groupedx.speed:
            grouped = dataframe.groupby('День недели')['Ср.скорость, км/ч'].median()
        return grouped
    else:
        print("Error: Wrong sort_by value")


def plot_grouped(sort_by: Grouped, sort_by_x: Groupedx):
    d1, d2 = get_clean_data()
    grouped = get_grouped_by(d1, sort_by, sort_by_x)
    grouped1 = get_grouped_by(d2, sort_by, sort_by_x)
    if sort_by == Grouped.hour:
        if sort_by_x == Groupedx.car:
            fig = px.line(grouped, x=grouped.index, y=grouped.values, title='Кол-во машин в час',\
                        color_discrete_sequence=['blue'])
            fig.data[0].name="Декабрь 2022"
            fig.update_traces(showlegend=True)
            fig.add_bar(x=grouped1.index, y=grouped1.values, name='Март 2023', marker_color='red')
            fig.update_traces(name='Декабрь 2022', selector=dict(line_color='blue'))
            fig.update_layout(
                legend_title_text='Сравнение времени движения по часам',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    title_font_family="Times New Roman",
                    traceorder='grouped'
                ),
                # itemsizing='constant',
                xaxis_title="Время",
                yaxis_title="Кол-во машин"
            )
            n = len(grouped)
            val = []
            for i in range(len(grouped)):
                val.append(grouped1.values[i] - grouped.values[i])
            mean = sum(val) / n
        elif sort_by_x == Groupedx.speed:
            fig = px.line(grouped, x=grouped1.index, y=grouped1.values, title='Средняя скорость в час',\
                        color_discrete_sequence=['blue'])
            fig.data[0].name="Декабрь 2022"
            fig.update_traces(showlegend=True)
            fig.add_bar(x=grouped.index, y=grouped.values, name='Март 2023', marker_color='red')
            fig.update_traces(name='Декабрь 2022', selector=dict(line_color='blue'))
            fig.update_layout(
                legend_title_text='Сравнение времени движения по часам',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    title_font_family="Times New Roman",
                    traceorder='grouped'
                ),
                # itemsizing='constant',
                xaxis_title="Время",
                yaxis_title="Средняя скорость"
            )
            n = len(grouped1)
            val = []
            for i in range(len(grouped1)):
                val.append(grouped.values[i] - grouped1.values[i])
            mean = sum(val) / n
        return fig, val, mean, grouped1
    elif sort_by == Grouped.week:
        if sort_by_x == Groupedx.car:
            fig = px.line(grouped, x=grouped.index, y=grouped.values, title='Кол-во машин в час',\
                        color_discrete_sequence=['blue'])
            # Add name to the line : Декабрь 2022
            fig.data[0].name="Декабрь 2022"
            fig.update_traces(showlegend=True)
            fig.add_bar(x=grouped1.index, y=grouped1.values, name='Март 2023', marker_color='red')
            fig.update_layout(
                legend_title_text='Сравнение времени движения по часам',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    title_font_family="Times New Roman",
                    traceorder='grouped'
                ),
                # itemsizing='constant',
                xaxis_title="День недели",
                yaxis_title="Кол-во машин",
                xaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1, 2, 3, 4, 5, 6],
                    ticktext=['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
                ),
            )
            n = len(grouped)
            val = []
            for i in range(len(grouped)):
                val.append(grouped1.values[i] - grouped.values[i])
            mean = sum(val) / n
        elif sort_by_x == Groupedx.speed:
            fig = px.line(grouped, x=grouped1.index, y=grouped1.values, title='Средняя скорость в час',\
                        color_discrete_sequence=['blue'])
            fig.data[0].name="Декабрь 2022"
            fig.update_traces(showlegend=True)
            fig.add_bar(x=grouped.index, y=grouped.values, name='Март 2023', marker_color='red')
            fig.update_layout(
                legend_title_text='Сравнение времени движения по часам',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    title_font_family="Times New Roman",
                    traceorder='grouped'
                ),
                # itemsizing='constant',
                xaxis_title="День недели",
                yaxis_title="Средняя скорость",
                xaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1, 2, 3, 4, 5, 6],
                    ticktext=['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
                ),
            )
            n = len(grouped1)
            val = []
            for i in range(len(grouped1)):
                val.append(grouped.values[i] - grouped1.values[i])
            mean = sum(val) / n
        return fig, val, mean, grouped1
    else:
        print("Error: Wrong sort_by value")
        

st.title(""" Сравнение данных декабрь 2022 и март 2023 года""")
add_selectbox = st.sidebar.selectbox(
    "Что вы хотите посмотреть?",
    ("Визуализация данных", "Статисика", "Математическое моделирование")
)
if add_selectbox == "Визуализация данных":
    option = st.sidebar.radio(options=["Часы", "День недели"], label='Выберите частоту графика')
    option1 = st.sidebar.radio(options=["Кол-во автомобилей","Ср.скорость, км/ч"], label='Выберите метрику')
    if option == "Часы":
        if option1 == "Кол-во автомобилей":
            fig,valu,mean, grouped = plot_grouped(Grouped.hour, Groupedx.car)
        elif option1 == "Ср.скорость, км/ч":
            fig,valu,mean, grouped = plot_grouped(Grouped.hour, Groupedx.speed)
        st.plotly_chart(fig)
        if option1 == "Кол-во автомобилей":
            st.write(f"Среднее количество машин на светофоре в час уменьшилось на : {-1*round(mean,2)}")
            for i,j in enumerate(valu):
                if len(valu) - 5 == i:
                    break
                col1, col2,col3, col4, col5 = st.columns(5)
                col1.metric(f"C {i}:00 по {i+1}:00", value=grouped[i], delta=valu[i], delta_color="inverse")
                i+=1
                col2.metric(f"C {i}:00 по {i+1}:00", value=grouped[i], delta=valu[i], delta_color="inverse")
                i+=1
                col3.metric(f"C {i}:00 по {i+1}:00", value=grouped[i], delta=valu[i], delta_color="inverse")
                i+=1
                col4.metric(f"C {i}:00 по {i+1}:00", value=grouped[i], delta=valu[i], delta_color="inverse")
                i+=1
                col5.metric(f"C {i}:00 по {i+1}:00", value=grouped[i], delta=valu[i], delta_color="inverse")
        elif option1 == "Ср.скорость, км/ч":
            st.write(f"Средняя скорость движения на светофоре в час увеличилась на : {round(mean,2)}")
            for i,j in enumerate(valu):
                if len(valu) - 5 == i:
                    break
                col1, col2,col3, col4, col5 = st.columns(5)
                col1.metric(f"C {i}:00 по {i+1}:00", value=grouped[i], delta=valu[i])
                i+=1
                col2.metric(f"C {i}:00 по {i+1}:00", value=grouped[i], delta=valu[i])
                i+=1
                col3.metric(f"C {i}:00 по {i+1}:00", value=grouped[i], delta=valu[i])
                i+=1
                col4.metric(f"C {i}:00 по {i+1}:00", value=grouped[i], delta=valu[i])
                i+=1
                col5.metric(f"C {i}:00 по {i+1}:00", value=grouped[i], delta=valu[i])

    elif option == "День недели":
        if option1 == "Кол-во автомобилей":
            fig,valu,mean, grouped = plot_grouped(Grouped.week, Groupedx.car)
        elif option1 == "Ср.скорость, км/ч":
            fig,valu,mean, grouped = plot_grouped(Grouped.week, Groupedx.speed)
        st.plotly_chart(fig)
        st.write(f"Среднее количество машин на светофоре в зависимости от дня недели уменьшилось на : {round(mean,2)}")
        if option1 == "Кол-во автомобилей":
            col1, col2,col3, col4, col5, col6, col7 = st.columns(7)
            col1.metric(f"Понедельник", value=grouped[0], delta=valu[0], delta_color="inverse")
            col2.metric(f"Вторник", value=grouped[1], delta=valu[1], delta_color="inverse")
            col3.metric(f"Среда", value=grouped[2], delta=valu[2], delta_color="inverse")
            col4.metric(f"Четверг", value=grouped[3], delta=valu[3], delta_color="inverse")
            col5.metric(f"Пятница", value=grouped[4], delta=valu[4], delta_color="inverse")
            col6.metric(f"Суббота", value=grouped[5], delta=valu[5], delta_color="inverse")
            col7.metric(f"Воскресенье", value=grouped[6], delta=valu[6], delta_color="inverse")
        elif option1 == "Ср.скорость, км/ч":
            col1, col2,col3, col4, col5, col6, col7 = st.columns(7)
            col1.metric(f"Понедельник", value=grouped[0], delta=valu[0])
            col2.metric(f"Вторник", value=grouped[1], delta=valu[1])
            col3.metric(f"Среда", value=grouped[2], delta=valu[2])
            col4.metric(f"Четверг", value=grouped[3], delta=valu[3])
            col5.metric(f"Пятница", value=grouped[4], delta=valu[4])
            col6.metric(f"Суббота", value=grouped[5], delta=valu[5])
            col7.metric(f"Воскресенье", value=grouped[6], delta=valu[6])


elif add_selectbox == "Статисика":
    st.subheader("Статистика")

    option = st.sidebar.radio("Выбирите раздел стаистики", ("Визуализация статистических показателей по месяцам", "Сравнение"))

    if option == "Визуализация статистических показателей по месяцам":
        option = st.radio("Выберите месяц", ("Декабрь", "Март"))
        df1 = 1
        df2 = 2

        if option == "Декабрь":
            df_a = get_clean_data(df1,df2)[0]
        elif option == "Март":
            df_a = get_clean_data(df1,df2)[1]

        # Преобразование колонки 'Время' в формат времени
        df_a['Время'] = pd.to_datetime(df_a['Время'])

        # Установка 'Время' как индекс
        df_a.set_index('Время', inplace=True)
        df_a = df_a.resample('5T').mean()
        df_a['Автомобилей'] = df_a['Автомобилей'].dropna()

        st.write("Визуализация временного ряда")
        st.line_chart(df_a['Автомобилей'])

        st.write("Визуализация распределения")
        st.area_chart(df_a['Автомобилей'])

        st.write("Визуализация среднего значения")
        st.bar_chart(df_a['Автомобилей'])

        st.write("Визуализация корреляции")
        st.write(df_a.corr())

        st.write("Визуализация статистики")
        st.write(df_a.describe())

    elif option == "Сравнение":
        # Сравнение месяцев по количеству автомобилей Декабрь и Март
        df1 = 1
        df2 = 2
        df_a = get_clean_data(df1,df2)[0]
        df_b = get_clean_data(df1,df2)[1]

        # Преобразование колонки 'Время' в формат времени
        df_a['Время'] = pd.to_datetime(df_a['Время'])
        df_b['Время'] = pd.to_datetime(df_b['Время'])

        # Установка 'Время' как индекс
        df_a.set_index('Время', inplace=True)
        df_b.set_index('Время', inplace=True)

        df_a = df_a.resample('5T').mean()
        df_b = df_b.resample('5T').mean()

        # AB тестирование
        st.subheader("AB тестирование")
        st.write("Сравнение средней скрости в марте и декабре")
        t_statistic, p_value = stats.ttest_ind(df_a['Ср.скорость, км/ч'], df_b['Ср.скорость, км/ч'], nan_policy='omit')

        # Вывод результатов веб-приложении с помощью Streamlit
        print("t-статистика:", t_statistic)
        print("p-значение:", p_value)
        if p_value < 0.05:
            print("Отвергаем нулевую гипотезу: между выборками есть статистически значимая разница")
        st.write("Результаты t-теста:")
        if t_statistic > 0:
            st.write("t-статистика:", t_statistic)
            st.write("t-статистика > 0 => Средняя скорость в марте больше, чем в декабре")
        else:
            st.write("t-статистика:", t_statistic)
            st.write("t-статистика < 0 => Средняя скорость в декабре больше, чем в марте")
        st.write("t-статистика:", t_statistic)
        if p_value < 0.05:
            st.write("p-значение < 0.05")
            st.write("Отвергаем нулевую гипотезу: между выборками есть статистически значимая разница")
        else:
            st.write("p-значение > 0.05")
            st.write("Не получилось отвергнуть нулевую гипотезу: между выборками нет статистически значимой разницы")

        # Сравнение месяцев по количеству автомобилей Декабрь и Март
        df1 = 1
        df2 = 2
        df_a = get_clean_data(df1,df2)[0]
        df_b = get_clean_data(df1,df2)[1]

        # Преобразование колонки 'Время' в формат времени
        df_a['Время'] = pd.to_datetime(df_a['Время'])
        df_b['Время'] = pd.to_datetime(df_b['Время'])

        # Установка 'Время' как индекс
        df_a.set_index('Время', inplace=True)
        df_b.set_index('Время', inplace=True)

        df_a = df_a.resample('5T').mean()
        df_b = df_b.resample('5T').mean()

        # AB тестирование
        st.write("Сравнение средней кол-ва автомоблей в марте и декабре")
        t_statistic, p_value = stats.ttest_ind(df_a['Автомобилей'], df_b['Автомобилей'], nan_policy='omit')
        
        # Вывод результатов веб-приложении с помощью Streamlit
        print("t-статистика:", t_statistic)
        print("p-значение:", p_value)
        if p_value < 0.05:
            print("Отвергаем нулевую гипотезу: между выборками есть статистически значимая разница")
        st.write("Результаты t-теста:")
        if t_statistic > 0:
            st.write("t-статистика:", t_statistic)
            st.write("t-статистика > 0 => Средняя скорость в марте больше, чем в декабре")
        else:
            st.write("t-статистика:", t_statistic)
            st.write("t-статистика < 0 => Средняя скорость в декабре больше, чем в марте")
        st.write("t-статистика:", t_statistic)
        if p_value < 0.05:
            st.write("p-значение < 0.05")
            st.write("Отвергаем нулевую гипотезу: между выборками есть статистически значимая разница")
        else:
            st.write("p-значение > 0.05")
            st.write("Не получилось отвергнуть нулевую гипотезу: между выборками нет статистически значимой разницы")

        

elif add_selectbox == "Математическое моделирование":

    option = st.radio("Выберите месяц", ("Декабрь", "Март"))
    df1 = 1
    df2 = 2

    if option == "Декабрь":
        df_a = get_clean_data(df1,df2)[0]
    elif option == "Март":
        df_a = get_clean_data(df1,df2)[1]

    # Преобразование колонки 'Время' в формат времени
    df_a['Время'] = pd.to_datetime(df_a['Время'])

    # Установка 'Время' как индекс
    df_a.set_index('Время', inplace=True)
    df_a = df_a.resample('5T').mean()
    df_a['Автомобилей'] = df_a['Автомобилей'].dropna()

    st.write("Визуализация временного ряда")
    st.pyplot(df_a.plot(figsize=(10, 5)).figure)

    # Разложение временного ряда на тренд, сезонность и остаток
    df_a.fillna(method='ffill', inplace=True)
    decomp = seasonal_decompose(df_a['Автомобилей'], model='additive', period=288)

    st.header("Визуализация разложенных компонентов временного ряда")
    st.write('''
        График 1: График наблюдений (Observations) показывает исходные данные временного ряда, которые мы разложили на компоненты.

        График 2: График тренда (Trend) показывает среднее значение временного ряда на каждый момент времени. Эта компонента показывает, как меняется общий тренд ряда в течение времени.

        График 3: График сезонности (Seasonality) показывает, как наш временной ряд повторяется в определенный период времени. На этом графике мы можем увидеть повторяющиеся паттерны, которые повторяются в течение определенного периода времени.

        График 4: График остатков (Residuals) показывает разницу между исходными данными временного ряда и их предсказанными значениями. Эта компонента показывает шум в данных, который не может быть объяснен трендом и сезонностью.

        График 5: На этом графике мы можем увидеть исходный временной ряд (Original) и скользящее среднее (Rolling Mean). Скользящее среднее применяется для сглаживания временного ряда, удаляя шум и другие аномалии, которые могут мешать анализу данных.
    ''')
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    decomp.observed.plot(ax=axs[0], title='Observations')
    decomp.trend.plot(ax=axs[1], title='Trend')
    decomp.seasonal.plot(ax=axs[2], title='Seasonality')
    decomp.resid.plot(ax=axs[3], title='Residuals')
    plt.tight_layout()
    st.pyplot(fig)

    st.write("Проверка стационарности временного ряда")
    result = adfuller(df_a['Автомобилей'])
    st.write('ADF Statistic:', result[0])
    st.write('p-value:', result[1])
    st.write('Critical Values:')
    for key, value in result[4].items():
        st.write(f"\t{key}: {value:.3f}")

    rolling_mean = df_a['Автомобилей'].rolling(window=10).mean()
    fig, ax = plt.subplots()
    ax.plot(df_a['Автомобилей'], label='Original')
    ax.plot(rolling_mean, label='Rolling Mean')
    ax.legend(loc='upper left')
    st.pyplot(fig)


    # Вычитание скользящего среднего из исходного временного ряда
    df_minus_rolling_mean = df_a['Автомобилей'] - rolling_mean
    df_minus_rolling_mean.dropna(inplace=True)

    st.write("Проверка стационарности вычитанного скользящего среднего из исходного временного ряда")
    result = adfuller(df_minus_rolling_mean)
    st.write('ADF Statistic:', result[0])
    st.write('p-value:', result[1])
    st.write('Critical Values:')
    for key, value in result[4].items():
        st.write(f"\t{key}: {value:.3f}")

    st.write("Построение модели")
    st.write("Разделение временного ряда на обучающую и тестовую выборки")
    train = df_a['Автомобилей'][:int(0.8*(len(df_a)))]
    test = df_a['Автомобилей'][int(0.8*(len(df_a))):]

    st.write("Построение модели ARIMA")
    model = ARIMA(train, order=(1, 1, 1))
    model_fit = model.fit()
    st.write(model_fit.summary())

    st.write("Построение прогноза")
    forecast_values = model_fit.forecast(len(test), alpha=0.0, return_conf_int=True)[0]
    fc_series = pd.Series(forecast_values, index=test.index)

    st.write("Визуализация прогноза")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train, label='training')
    ax.plot(test, label='actual')
    ax.plot(fc_series, label='forecast')
    ax.set_title('Forecast vs Actuals')
    ax.legend(loc='upper left', fontsize=8)
    st.pyplot(fig)

    st.write("Проверка точности прогноза")
    print(test.shape, fc_series)
    mse = mean_squared_error(test, fc_series)
    rmse = sqrt(mse)
    st.write(f"RMSE: {rmse:.3f}")

    st.write("Построение модели SARIMA")
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit()
    st.write(model_fit.summary())

    st.write("Построение прогноза")
    forecast_values = model_fit.forecast(len(test), alpha=0.05, return_conf_int=True)
    fc_series = pd.Series(forecast_values, index=test.index)

    st.write("Визуализация прогноза")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train, label='training')
    ax.plot(test, label='actual')
    ax.plot(fc_series, label='forecast')
    ax.set_title('Forecast vs Actuals')
    ax.legend(loc='upper left', fontsize=8)
    st.pyplot(fig)


