# am o lista de proprietati de inchiriat pe Airbnb din BARCELONA
# vreau sa fac grafic pentru preturi si persoane acomodate
import random
from tkinter import ttk

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tkinter import *
from wordcloud import WordCloud
import json
import folium
import geopandas as gpd
import plotly.express as px
from math import radians, cos, sin, asin, sqrt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


sns.set_theme(style="whitegrid")# am un grid pe care vad valorile

sns.set()

listings = pd.read_csv('listings_BCN.csv') #import tabelul index_col=0

df_nou= listings.loc[:, ['accommodates', 'price','neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'room_type', 'availability_90','number_of_reviews','bathrooms_text', 'amenities',
                         'number_of_reviews_l30d', 'bedrooms', 'beds', 'minimum_nights', 'review_scores_cleanliness', 'review_scores_location',
                         'review_scores_value', 'calculated_host_listings_count_entire_homes', 'latitude', 'longitude' ]]




# curatarea datasetului

def clean_dataset():

    print(' Aflam statistici din tabelul nostru', '\n')
    print(df_nou.describe())
    print(' Aflam cate valori sunt non nule in tabelul nostru', '\n')
    print(df_nou.info())

    df_nou.dropna(inplace = True) # sterg toare randurile cu valori nule

    index_sharedroom = (df_nou[(df_nou['room_type'] == 'Private room') | ( df_nou['room_type'] == 'Shared room' ) | ( df_nou['room_type'] == 'Hotel room' )].index)
    df_nou.drop( index_sharedroom, inplace = True) # renunt la cazarile tip private room si shared room ca ma incurca la predictia de preturi

    index_sharedbathroom = (df_nou[df_nou['bathrooms_text'].str.contains('Half-bath', na = False)].index) # cautam string care contine Half/ bath si il stergem din lista daca contine valori nule le sarim
    df_nou.drop(index_sharedbathroom, inplace=True)  # renunt la cazarile tip private room si shared room ca ma incurca la predictia de preturi

    df_nou['bathroom_number'] = df_nou["bathrooms_text"].str.split(" ", expand=True)[0].astype('float') # facem o noua coloana cu nr de bai si luam doar primul cuvant din lista de stringuri

    index_price = (df_nou[(df_nou['price'] > 600) | (df_nou['price']< 20)].index)
    df_nou.drop(index_price, inplace=True)  # renunt la cazarile mai scumpe de 600 de euro si mai ieftine de 10 euro

    index_availability = (df_nou [(df_nou['availability_90'] < 1) | (df_nou['availability_90'] > 89)].index)
    df_nou.drop(index_availability, inplace=True ) # renunt la cazarile care nu a fost inchiriate deloc sau care au blocata urmatoarele 3 luni

    index_accommodates = (df_nou [df_nou['accommodates'] < 1].index)
    df_nou.drop(index_accommodates, inplace=True )# renunt la cazarile care nu pot acomoda minim o persoana

    index_bathroom_accommodates = (df_nou [ df_nou['bathroom_number'] == 0].index)
    df_nou.drop(index_bathroom_accommodates, inplace = True) # renunt la cazarile fara baie

    df_nou.drop(['bathrooms_text'], axis =1, inplace = True) # renunt la coloana cu numarul de bai

    df_nou.drop_duplicates()



    def amenities_create_column(word, column_name):

        index_element = (df_nou[df_nou['amenities'].str.contains(word, na=False)].index)
        # cautam string care contine pets_allowed si il stergem din lista, daca contine valori nule le sarim - asta inseamna na= False

        df_nou[column_name] = [1 if i in index_element else 0 for i in df_nou.index]
        # facem o noua coloana care contine 1 daca allows pets si zero daca nu allows pets

    list_words_columns= [('Air conditioning', 'air_conditioning'), ('Kitchen', 'kitchen'), ('Coffee maker', 'coffee_maker'), ('Elevator', 'elevator')]

    for words_pair in list_words_columns:
        amenities_create_column(words_pair[0],words_pair[1])

    df_nou.reset_index(drop = True, inplace=True ) # resetam indecsii din data frame ul nostru si nu pastram indecsii initiali in alta coloana

    df_nou.sort_values(by= ['price', 'neighbourhood_group_cleansed'], ascending= False, inplace=True)

    df_nou.reset_index(drop = True, inplace=True ) # resetam indecsii din data frame ul nostru si nu pastram indecsii initiali in alta coloana

    print(df_nou['price'].head(18))

    creare_tabel_geo() # creez tabelul date geo

    '''
        def functie_aplicare_pret_median(cartier):

            try:
                return(df_geo[df_geo['neighbourhood'] == cartier].loc[0,'median_price'])
            except:
                return 0

        df_nou['pret_median_cartier']= df_nou['neighbourhood_cleansed'].apply(lambda cartier: functie_aplicare_pret_median(cartier))

        df_nou.drop(df_nou[df_nou['pret_median_cartier'] == 0].index, inplace=True)
        df_nou.reset_index(drop=True, inplace=True) # resetez indecsii

        print("Acesta este tabelul meu" , df_nou['pret_median_cartier'])

    '''

    # Function to calculate distance using Haversine formula
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371 # radius of Earth in kilometers
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return R * c


        # Calculate distance from mean coordinates
    mean_lat = df_nou['latitude'].mean()
    mean_lon = df_nou['longitude'].mean()
    df_nou['distance'] = df_nou.apply(lambda row: haversine_distance(row['latitude'], row['longitude'], mean_lat, mean_lon), axis=1)

    df_nou.head(200).to_csv('BCN_new.csv', index=False) # creez un csv unde vad valorile

onehot_df= pd.DataFrame()

# encoding neighbourhood values

def functie_encoding():
    # Create a OneHotEncoder object
    onehot_encoder = OneHotEncoder()

    # Select the columns to be encoded
    categorical_columns = ['neighbourhood_cleansed']

    # Encode the columns
    onehot_encoded = onehot_encoder.fit_transform(df_nou[categorical_columns])

    # Create a DataFrame from the one-hot encoded data
    global onehot_df
    onehot_df = pd.DataFrame(onehot_encoded.toarray(), columns=onehot_encoder.get_feature_names_out())



def functie_linear_regression():

    # Select the features and target variable

    functie_encoding()

    list_keys = onehot_df.columns.tolist()

    df_linear_rg = pd.concat([df_nou, onehot_df], axis=1)

    list_columns = ['accommodates', 'availability_90', 'number_of_reviews', 'number_of_reviews_l30d',
     'bedrooms', 'beds', 'minimum_nights', 'review_scores_cleanliness', 'review_scores_location', 'review_scores_value',
     'bathroom_number', 'air_conditioning', 'coffee_maker', 'elevator', 'distance'] + list_keys


    X = df_linear_rg[list_columns]

    y = df_linear_rg['price']

    # Split the data into training and test sets
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=10)

    print('data being testes is of type ', type(X_train))

    # Create a LinearRegression model and fit it to the training data
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    print(y_pred)

    functie_rmse(y_test, y_pred)

rmse=0.0

def functie_rmse(y_test, y_pred):
    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    global rmse
    rmse= mse ** 0.5

    # Calculate the baseline RMSE
    y_mean = np.mean(y_test)
    baseline_mse = mean_squared_error(y_test,
                                      [y_mean] * len(y_test))  # compar cu o lista care contine valori identice medii.
    baseline_rmse = baseline_mse ** 0.5

    print('Root mean squared error (RMSE): ', rmse)
    print('Baseline RMSE: ', baseline_rmse)

    # Compare the RMSE of the model to the baseline RMSE
    if rmse < baseline_rmse:
        print("The model is making better predictions than the baseline.")

        np.random.seed(10)
        ytest_sample = np.random.choice(y_test, size=20, replace=False)
        np.random.seed(10)
        ypred_sample = np.random.choice(y_pred, size=20, replace=False)

        grafic_comparare_functii(ytest_sample, ypred_sample)

    else:
        print("The model is making worse predictions than the baseline.")


# comparam predictiile si vedem diferenta dintre pretul prezis si cel real

def grafic_comparare_functii(x, y):
    df = pd.DataFrame({'True Price': x, 'Predicted Price': y})

    print(df.index)

    x = df.index
    width = 0.35

    plt.figure(figsize=(13, 8))
    plt.bar(x - width / 2, df['True Price'], width, label='Actual values')
    plt.bar(x + width / 2, df['Predicted Price'], width, label='Predicted values')

    plt.xlabel('Comparative prices')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted prices for Linear Regression'+'\n'+'Root mean square error is ' + str(round(rmse,2)))
    plt.xticks(x)
    plt.legend()

    plt.show()



def functie_wordcloud():
    # Create a list of word
    text =(df_nou['amenities'].str.cat(sep=', '))

    # Create the wordcloud object
    wordcloud = WordCloud(width=480, height=480, margin=0).generate(text)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.show()



# heatmap and kendall rank correlation

def functie_heatmap():
    df_corr = df_nou.loc[:, ['accommodates', 'price', 'availability_90', 'number_of_reviews', 'number_of_reviews_l30d',
                 'bedrooms', 'beds', 'minimum_nights','review_scores_cleanliness', 'review_scores_location', 'review_scores_value',
                'bathroom_number', 'air_conditioning', 'coffee_maker', 'elevator', 'distance']]

    corr = df_corr.corr(method='kendall') # metoda de corelare a datelor

    plt.figure(figsize=(13, 13))

    plt.title("Correlation Between Different Variables")

    sns.heatmap(corr, annot=True)

    plt.show()


# tabel boxplot cu explorarea preturilor pe cartiere0

def distributie_price_boxplot():

    plt.style.use('classic')
    plt.figure(figsize=(19, 7))
    plt.title("Listings Price Distribution per Neighbourhood")
    sns.boxplot(data=df_nou.loc[:, 'price'], x=df_nou.loc[:, 'neighbourhood_group_cleansed'], y=df_nou.loc[:, 'price'], palette="dark")
    plt.xlabel('Neighbourhood')
    plt.show()


# histograma preturilor cu matplotlib


def functie_histograme():

    etichete_x = []  # creez o marja de preturi
    x_ticks_labels = []  # creez etichete pentru marja de preturi

    for i in range(20):
        etichete_x.append((i + 1) * 25)

    plt.figure(figsize=(10,6))

    plt.hist(df_nou.loc[:, 'price'], bins=20, edgecolor='white', orientation='vertical',
             range=(0, 500))  # histograma pentru preturi
    plt.title('Listings price graphic for Barcelona')
    plt.xlabel('Price range in dollars')  # axa x
    plt.ylabel('Average number of listings')  # axa y
    plt.xticks(etichete_x)
    plt.show()


# pie chart cartiere

def functie_pie_chart_cartiere():

    # pie chart in seaborn and matplotlib
    colors = sns.color_palette('pastel')

    # create a pie chart

    data_series_neighbourhoods = df_nou['neighbourhood_group_cleansed'].value_counts()

    plt.figure(figsize=(15, 10))

    plt.title('Distribution of listings in each neighbourhood')
    plt.pie(data_series_neighbourhoods, labels=data_series_neighbourhoods.index, colors=colors, autopct='%.0f%%')
    plt.show()


#  regplot function

def regplot_function():

    plt.figure(figsize=(19, 8))

    plt.subplot(1, 2, 1)
    # plot
    sns.regplot(x= df_nou.accommodates, y=df_nou.beds, line_kws={"color": "r", "alpha": 0.7, "lw": 5})
    plt.title('Linear correlation between indices')
    plt.xlabel('Number of guests')  # axa x
    plt.ylabel('Number of beds')  # axa y


    plt.subplot(1, 2, 2)
    # plot
    sns.regplot(x= df_nou.accommodates, y=df_nou.price, line_kws={"color": "r", "alpha": 0.7, "lw": 5})
    plt.title('Linear correlation between indices')
    plt.xlabel('Number of guests')  # axa x
    plt.ylabel('Price in dollars')  # axa y

    plt.show()

# creare tabel unique amenities si sortare in functie de popularitate

def functie_sortare_amenities():

    # create a concatenated string of all the amenities
    text = ' '.join(df_nou['amenities'])
    words = text.split('"')

    list_words= []

    for word in words:
            if  ('] [' not in word) and ('[' not in word) and ('\\' not in word) and (']' not in word) and (', ' not in word):
                list_words.append(word)

    unique_list = []
    appearances_list = []

    for i in list_words:
        if i not in unique_list:
            unique_list.append(i)
            appearances_list.append(list_words.count(i))

    sorted_unique_list= sorted(unique_list, key=lambda x: appearances_list[unique_list.index(x)], reverse=True)
    sorted_appearances_list= sorted(appearances_list, reverse=True)

    unique_median_price = []

    for amenity in sorted_unique_list:
        try:
            if (df_nou[df_nou['amenities'].str.contains(amenity, na=False)]['price'].shape[0] > 20) and (df_nou[df_nou['amenities'].str.contains(amenity, na=False)]['price'].median() != float('nan')) :
                unique_median_price.append(df_nou[df_nou['amenities'].str.contains(amenity, na=False)]['price'].median())
            else:
                unique_median_price.append(0)
        except:
            unique_median_price.append(0)


    sorted_unique_amenities_price= sorted(sorted_unique_list, key=lambda x: unique_median_price[sorted_unique_list.index(x)], reverse=True)
    sorted_unique_median_price= sorted(unique_median_price, reverse=True)

    return(sorted_unique_list, sorted_appearances_list, sorted_unique_amenities_price, sorted_unique_median_price )



def functie_barplot():


    sorted_unique_list, sorted_appearances_list, sorted_unique_amenities_price, sorted_unique_median_price = functie_sortare_amenities()

    lista_aparitii_array= np.array(sorted_appearances_list)

    plt.figure(figsize=(19, 5))

    plt.subplot(2, 1, 1)

    # Create bars
    plt.barh(sorted_unique_list[:30], sorted_appearances_list[:30])

    plt.xticks([x for x in range(0, 6000, 500)], fontsize=12)
    plt.yticks(sorted_unique_list[:30], fontsize=8)
    plt.title('Most popular amenities')
    # Show graphic


    plt.subplot(2, 1, 2)
    plt.barh(sorted_unique_amenities_price[:30], sorted_unique_median_price[:30])

    plt.xticks([x for x in range(0, 360, 50)], fontsize=12)
    plt.yticks(sorted_unique_amenities_price[:30], fontsize=8)
    plt.title('Amenities correlated with the highest prices')


    plt.show()


# functia tkinter care afiseaza meniul

root = Tk()

def functie_afisare_tkinter():

    global root


    # Create the root window
    bg = PhotoImage(file='barca_map.PNG')
    my_label = Label(root, image=bg)
    my_label.pack()
    root['bg'] = 'white'
    root.title("   B a r c e l o n a")
    root.iconbitmap('label_airbnb.ico')

    # Create the menu bar
    menu_bar = Menu(root)

    # Create the About menu
    about_menu = Menu(menu_bar, tearoff=0, background = 'white',  activebackground="white", activeforeground="grey")
    about_menu.add_command(label="a b o u t")
    about_menu.add_separator()
    about_menu.add_command(label="e x i t",  command=root.quit)
    menu_bar.add_cascade(label="a b o u t", menu=about_menu)

    # Create the Maps menu

    maps_menu = Menu(menu_bar, tearoff=0, background = 'white',  activebackground="white", activeforeground="grey")
    maps_menu.add_command(label="d i s t r i c t s", command=afisare_plotly_harta)
    maps_menu.add_command(label="l i s t i n g s", command=afisare_plotly_scatter)
    maps_menu.add_command(label="c o r r e l a t e", command= functie_afisare_harta)
    menu_bar.add_cascade(label="m a p s", menu=maps_menu)

    # Create the Charts menu
    charts_menu = Menu(menu_bar, tearoff=0, background = 'white',  activebackground="white", activeforeground="grey")
    charts_menu.add_command(label="h i s t", command= functie_histograme)
    charts_menu.add_command(label="p i e", command= functie_pie_chart_cartiere)
    charts_menu.add_command(label="b o x p l t", command= distributie_price_boxplot)
    charts_menu.add_command(label="w o r d c l o u d", command=functie_wordcloud)
    charts_menu.add_command(label="b a r p l t", command=functie_barplot)
    charts_menu.add_command(label="h e a t m a p", command= functie_heatmap)
    charts_menu.add_command(label="s c a t t e r", command=regplot_function)

    menu_bar.add_cascade(label="c h a r t s", menu=charts_menu)


    # Create the Explore menu
    explore_menu = Menu(menu_bar, tearoff=0, background = 'white',  activebackground="white", activeforeground="grey")
    explore_menu.add_command(label="l i s t i n g", command = openNewWindow )
    explore_menu.add_command(label="l i n e a r  r e g", command=functie_linear_regression)
    menu_bar.add_cascade(label="e x p l o r e", menu=explore_menu)

    # Display the menu bar
    root.config(menu=menu_bar)

    # Run the main loop
    root.mainloop()

df_cautat= pd.DataFrame()


# afisarea graficului rezultat in urma selectiei optiunilor posibile

def afisare_grafic(c_area, c_people, c_bedrooms, c_beds, c_amenity1, c_amenity2, c_amenity3):


    global df_cautat
    df_cautat = df_nou.copy()

    df_cautat['neighbourhood_group_cleansed'] = df_cautat['neighbourhood_group_cleansed'].astype(str)
    df_cautat.reset_index(drop=True, inplace=True)

    int_people = int(c_people)
    int_bedrooms = int(c_bedrooms)
    int_beds = int(c_beds)

    def filter_number(column, int_value):
        global df_cautat
        if df_cautat[df_cautat[column] == int_value].shape[0] > 50:
            index_value = (df_cautat[df_cautat[column] != int_value].index)
            df_cautat.drop(index_value, inplace=True)  # renunt la randurile care nu se regasesc in conditie

    def filter_string(column, string_value):
        global df_cautat
        if df_cautat[df_cautat[column].str.contains(c_area)].shape[0]>40:
            index_value = (df_cautat[df_cautat[column].str.contains(string_value, regex=False, case=False, na=False) == False] .index)
            df_cautat.drop(index_value, inplace=True)  # renunt la randurile care nu se regasesc in conditie

    filter_number('accommodates', int_people)
    print("dupa accommodates ", len(df_cautat.index))

    filter_string('neighbourhood_group_cleansed', c_area)
    print("dupa cartier ", len(df_cautat.index))

    filter_number('bedrooms', int_bedrooms)
    print("dupa bedrooms ", len(df_cautat.index))

    filter_number('beds', int_beds)
    print("dupa beds ", len(df_cautat.index))

    filter_string('amenities', c_amenity1)
    print("dupa amenities1 ", len(df_cautat.index))

    filter_string('amenities', c_amenity2)
    print("dupa amenities2 ", len(df_cautat.index))

    filter_string('amenities', c_amenity3)
    print("dupa amenities3 ", len(df_cautat.index))


    plt.figure(figsize=(19, 5))

    sns.kdeplot(df_cautat['price'], fill=True, bw_method=0.5)
    plt.axvline(df_cautat['price'].median(), 0, 1, label='median value', c = 'c') # afisez o linie verticala cu mediana valorilor
    plt.axvline(df_cautat['price'].quantile(0.25), 0, 1, label='25% of total values', c = 'r') # afisez o linie verticala cu mediana valorilor
    plt.axvline(df_cautat['price'].quantile(0.75), 0, 1, label='75% of total values', c = 'r') # afisez o linie verticala cu mediana valorilor
    plt.legend()
    list= [df_cautat['price'].median(), df_cautat['price'].quantile(0.25), df_cautat['price'].quantile(0.75)]
    plt.xticks(list,  fontsize=8)
    string_title= np.array2string(df_cautat['price'].median())
    plt.xlabel('Prices in dollars')
    plt.ylabel('Probability density function')
    plt.title('Price estimate: '+ string_title +'$')

    plt.show()

# intitalizare tabel geo
df_geo = pd.DataFrame()

# import json
with open('neighbourhoods.geojson') as f:
    state_geo = json.load(f)

def creare_tabel_geo():


    # creare dataframe cu subcartiere, cartiere, pret_mediu_culori, nr de listinguri

    df_neighbourhood_counts = df_nou['neighbourhood_cleansed'].value_counts()

    print(df_neighbourhood_counts)

    # Create a DataFrame from the DataSeries

    global df_geo
    df_geo['Values_count'] = df_neighbourhood_counts

    print(df_geo.head())

    list = df_neighbourhood_counts.index.tolist()
    # Extract the index to a column
    df_geo['neighbourhood'] = list

    df_geo.reset_index(drop=True, inplace=True)

    print(type(df_geo['Values_count']))

    # adaug coloana cu cartierul major

    cartier_mare_list = []
    for cartier in df_geo['neighbourhood']:
        cartier_mare = df_nou.loc[(df_nou['neighbourhood_cleansed']) == cartier, 'neighbourhood_group_cleansed'].iloc[0]
        cartier_mare_list.append(cartier_mare)
    # create new column
    df_geo['big_neighbourhood'] = cartier_mare_list

    # adaug coloana cu pretul median
    median_list = []
    for cartier in df_geo['neighbourhood']:
        median_val = df_nou.loc[(df_nou['neighbourhood_cleansed']) == cartier]['price'].median()
        median_list.append(int(median_val))
    # Create a new column in the dataframe based on the list of items
    df_geo['median_price'] = median_list


def functie_afisare_harta():

    global state_geo

     #Read the geoJSON file using geopandas
    geo_df = gpd.read_file('neighbourhoods.geojson')
    geo_df = geo_df[['neighbourhood', 'geometry']]  # only select 'coty_code' (country fips) and 'geometry' columns

    #print(geo_df.head())

    #merge two tables
    df_final = geo_df.merge(df_geo, left_on='neighbourhood', right_on="neighbourhood", how="outer")

    #print('acesta e tabelul final')
    #print(df_final.info())


    bcn_map = folium.Map(location=[df_nou.latitude.mean(), df_nou.longitude.mean()], zoom_start=13, control_scale= True, tiles='openstreetmap')


    folium.Choropleth( geo_data=state_geo, name="choropleth", data=df_final, columns=['neighbourhood', 'median_price'],
                       key_on='properties.neighbourhood', fill_color="YlGnBu", fill_opacity=0.7, line_opacity=.1, legend_name="Price median in Barcelona (%)", nan_fill_color="White").add_to(bcn_map)


    # Add Customized Tooltips to the map
    folium.features.GeoJson(
        data=df_final,
        name='Median price for listings',
        smooth_factor=2,
        style_function=lambda x: {'color': 'black', 'fillColor': 'transparent', 'weight': 0.5},
        tooltip=folium.features.GeoJsonTooltip(
            fields=['neighbourhood',
                    'big_neighbourhood',
                    'median_price',
                    'Values_count'],
            aliases=["Neighbourhood:",
                     "District:",
                     "Median price:",
                     "Number of listings:"],
            localize=True,
            sticky=False,
            labels=True,
            style="""
                                background-color: #F0EFEF;
                                border: 2px solid black;
                                border-radius: 3px;
                                box-shadow: 3px;
                            """,
            max_width=800 ),
        highlight_function=lambda x: {'weight': 3, 'fillColor': 'grey'},
    ).add_to(bcn_map)

    #for index, location_info in df_nou.iterrows():
        #folium.Marker([location_info["latitude"], location_info["longitude"]], popup=location_info["price"]).add_to(bcn_map)

    for index, location_info in df_nou.iterrows():
        folium.Circle( radius=5, location=[location_info["latitude"], location_info["longitude"]], color='crimson', fill=True, ).add_to(bcn_map)


    bcn_map.save('maps.html')


def afisare_plotly_harta():

    global state_geo

    # afisam datele

    fig = px.choropleth_mapbox(df_geo, geojson=state_geo, locations='neighbourhood', color='median_price', featureidkey="properties.neighbourhood",
                               color_continuous_scale="Viridis",
                               range_color=(10, 200),
                               mapbox_style="carto-positron",
                               zoom=11.5, center={"lat": df_nou.latitude.mean(), "lon": df_nou.longitude.mean()},
                               opacity=0.5, hover_data={ 'big_neighbourhood', 'Values_count'},
                               labels={'neighbourhood': 'neighbourhood', 'median_price': 'median price of listings',
                                       'big_neighbourhood': 'district', 'Values_count': 'listings count'  } )

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    fig.show()

def afisare_plotly_scatter():

    #import datele json
    with open('neighbourhoods.geojson') as f:
        state_geo = json.load(f)

    # afisam datele

    fig = px.scatter_mapbox(df_nou, lat="latitude", lon="longitude", color="price", size="price", hover_name="price",
                            hover_data=["neighbourhood_cleansed", "neighbourhood_group_cleansed"],
                            color_continuous_scale=px.colors.cyclical.IceFire, size_max=10, zoom=11.5)

    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()

# function to open a new window on a button click

def openNewWindow():

    # be treated as a new window
    newWindow = Toplevel(root)
    newWindow['bg'] = 'white'
    newWindow.iconbitmap('label_airbnb.ico')
    newWindow.geometry('300x250')
    newWindow.title("L I S T I N G")


    # Create a button to store the selected value
    def combo_click():
        c_area = combo_area.get()
        c_people= combo_people.get()
        c_bedrooms= combo_bedrooms.get()
        c_beds= combo_beds.get()
        c_amenity1= combo_amenity1.get()
        c_amenity2= combo_amenity2.get()
        c_amenity3= combo_amenity3.get()

        print(c_amenity2)

        afisare_grafic(c_area, c_people, c_bedrooms, c_beds, c_amenity1, c_amenity2, c_amenity3)


    # Create a combo area
    combo_area = ttk.Combobox(newWindow, value=df_nou.loc[:, 'neighbourhood_cleansed'].unique().tolist())
    combo_area.current(0)
    combo_area.grid(row=1, column=2)
    # combo_area.place(x=20, y=0)

    # Create a combo people
    combo_people = ttk.Combobox(newWindow, value=[x for x in range(1,17)])
    combo_people.current(0)
    combo_people.grid(row=2, column=2)

    # Create a combo bedrooms
    combo_bedrooms = ttk.Combobox(newWindow, value=[x for x in range(1, int(df_nou.bedrooms.max())+1)])
    combo_bedrooms.current(0)
    combo_bedrooms.grid(row=3, column=2)
    print(df_nou.bedrooms.min())

    # Create a combo beds
    combo_beds = ttk.Combobox(newWindow, value=[x for x in range(1, int(df_nou.beds.max())+1)])
    combo_beds.current(0)
    combo_beds.grid(row=4, column=2)

    sorted_unique_list= functie_sortare_amenities()

    # Create a combo amenitiy 1
    combo_amenity1 = ttk.Combobox(newWindow, value=sorted_unique_list[0][:20])
    combo_amenity1.current(0)
    combo_amenity1.grid(row=5, column=2)

    # Create a combo amenitiy 2
    combo_amenity2 = ttk.Combobox(newWindow, value=sorted_unique_list[0][20:40])
    combo_amenity2.current(0)
    combo_amenity2.grid(row=6, column=2)

    # Create a combo amenitiy 3
    combo_amenity3 = ttk.Combobox(newWindow, value=sorted_unique_list[0][40:60])
    combo_amenity3.current(0)
    combo_amenity3.grid(row=7, column=2)

    # Create label blank space
    label_blank = Label(newWindow, text="     ", bg="white")
    label_blank.grid(row=0, column=0)
    # label_areas.place(x=0, y=0)

    # Create label area for drop down menu
    label_areas = Label(newWindow, text="a r e a", bg="white")
    label_areas.grid(row=1, column=3, sticky="W")
    # label_areas.place(x=0, y=0)

    # Create label people for drop down menu
    label_people = Label(newWindow, text="g u e s t s", bg="white")
    label_people.grid(row=2, column=3, sticky="W")
    # label_people.grid(row = 2, column=0)

    # Create label bedrooms for drop down menu
    label_bedrooms = Label(newWindow, text="b e d r o o m s", bg="white")
    label_bedrooms.grid(row=3, column=3, sticky="W")

    # Create label beds for drop down menu
    label_beds = Label(newWindow, text="b e d s", bg="white")
    label_beds.grid(row=4, column=3, sticky="W")

    # Create label amenity1 for drop down menu
    label_amenity1 = Label(newWindow, text="a m e n i t y", bg="white")
    label_amenity1.grid(row=5, column=3, sticky="W")

    # Create label amenity2 for drop down menu
    label_amenity2 = Label(newWindow, text="a m e n i t y", bg="white")
    label_amenity2.grid(row=6, column=3, sticky="W")

    # Create label amenity3 for drop down menu
    label_amenity3 = Label(newWindow, text="a m e n i t y", bg="white")
    label_amenity3.grid(row=7, column=3, sticky="W")

    # Create label blank space
    label_blank2 = Label(newWindow, text="             ", bg="white")
    label_blank2.grid(row=8, column=0, sticky="W")

    buton_calculate = Button(newWindow, text='p r i c e   e s t i m a t i o n', height=1, bg='white', width=19, command=combo_click)

    buton_calculate.grid(row=9, column=2)


clean_dataset()

functie_afisare_tkinter()




















