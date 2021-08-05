import glob
import os
import numpy as np
np.set_printoptions(suppress=True)
import bs4 as bs
import pandas as pd
pd.set_option('max_colwidth', 8000)
import xgboost as xgb
from sklearn import preprocessing
import streamlit as st
import base64

st.title('PoRoRIT: Voorspel de Potentiele RIT Club- en Wedstrijdroeiers App!')

st.markdown("""
Op basis van van de inschrijfformulieren van voorgaande jaren voorspelt deze app of de nieuwe leden van Skoll potentie hebben om te gaan club- of wedstrijdroeien. 
Upload direct de CSV, gedownload van de website, waar alle nieuwe ritters instaan met hun gegevens, bijvoorbeeld genaamd 'nieuwe_leden_20210803.xls'. 
* **Python libraries:** bs4, pandas, streamlit, numpy, sklearn, xgboost, base64
* **Github:** https://github.com/rensvendrig/PoRoRIT.
\nGemaakt door Rens Vendrig
""")


def aggr_files():
    path = r'raw_data'  # use your path

    all_files = glob.glob(os.path.join(path, "*.xls"))
    all_files1 = [f.replace(r".xls", ".txt") for f in all_files]  ## replacing the file extension with .txt
    df_from_each_file = (pd.read_csv(f, delimiter="\t") for f in all_files)  ## reading the .txt files using csv reader
    df = pd.concat(df_from_each_file, ignore_index=True)  ## concatenating all the individual files
    return df


def drop_private_cols(df):
    df = df.drop(['Heb je je zwemdiploma A?', 'Anders, namelijk.1', 'Wat is het fanatiekst dat je hebt gesport?',
                  'Indien je niet in Amsterdam woont, ben je op zoek naar een kamer in Amsterdam?',
                  'Op Skøll eten veel mensen vegetarisch, pas je voorkeur hieronder aan als je toch wel vlees wilt eten',
                  'Namelijk', 'Wanneer verwacht je af te studeren?', 'Instituut',
                  'Wanneer ben je begonnen met je studie?',
                  'Heb je medische bijzonderheden waar wij rekening mee moeten houden?',
                  'Heb je allergieën waar wij rekening mee moeten houden?', 'Waar ken je Skøll van?',
                  'Heb je je naast sport op een ander gebied buitengewoon ingezet?',
                  'Zo ja, bij welke vereniging?', 'Hoeveel jaar heb je je langst beoefende sport beoefend?'
                  ], axis=1)
    return df


def transform_df(df):
    df['Geslacht'].replace('Vrouw', 0, inplace=True)
    df['Geslacht'].replace('Man', 1, inplace=True)

    df['Heb je eerder geroeid?'].replace('Nee', 0, inplace=True)
    df['Heb je eerder geroeid?'].replace('Ja', 1, inplace=True)

    df['Gewicht (in kg)'] = df['Gewicht (in kg)'].str.extract('(\d+)')
    df['Lengte (in cm)'] = df['Lengte (in cm)'].str.extract('(\d+)')

    df['Ik wil naar Skøll voor'].fillna('no', inplace=True)
    df['skollvoorroeien'] = np.where(df['Ik wil naar Skøll voor'].str.contains('top_rowing'), 1, 0)
    df.drop(['Ik wil naar Skøll voor'], axis=1, inplace=True)

    df['Zo ja, hoe lang?'].fillna('no', inplace=True)
    df['1+ jaar al geroeid'] = np.where(df['Zo ja, hoe lang?'].str.contains('jaar'), 1, 0)
    df.drop(['Zo ja, hoe lang?'], axis=1, inplace=True)

    df['Hoe vaak kan roeien'] = df['Wanneer kun je roeien?'].str.count('ja').fillna(0)
    df.drop(['Wanneer kun je roeien?'], axis=1, inplace=True)

    df['roeimotivatie'] = np.where(
        df['Waarom zou jij goed bij Skøll passen?'].str.contains('roei|sport|hard|wedstrijd'), 1, 0)
    df.drop(['Waarom zou jij goed bij Skøll passen?'], axis=1, inplace=True)

    df['Welke sporten heb je beoefend?'] = df['Welke sporten heb je beoefend?'].str.split(',').str.len()
    df['Anders, namelijk'] = df['Anders, namelijk'].fillna('nothing')
    df['Anders, namelijk'] = df['Anders, namelijk'].str.split('(,|and)').str.len()
    df['hoeveel sporten vroeger beoefend'] = pd.to_numeric(df['Welke sporten heb je beoefend?']) + pd.to_numeric(
        df['Anders, namelijk']) - 1
    df.drop(['Anders, namelijk', 'Welke sporten heb je beoefend?'], axis=1, inplace=True)

    mapdict1 = {'2 à 3 keer per week': 2.5
        , 'Minder dan 2 keer per week': 1
        , '3 à 4 keer per week': 3.5
        , 'Meer dan 4 keer per week': 5
        , '? keer per week': np.nan}
    df['# van plan te gaan roeien'] = df['Hoe vaak ben je van plan te gaan roeien per week?'].map(mapdict1)
    mapdict2 = {'2 à 3 keer per week': 2.5,
                '3 à 4 keer per week': 3.5,
                '4 à 5 keer per week': 4.5,
                '? keer per week': 0,
                'Meer dan 5 keer per week': 6,
                'Minder dan 2 keer per week': 1}
    df['# al geroeid'] = df['Zo ja, hoe vaak per week?'].map(mapdict2)
    df.drop(['Hoe vaak ben je van plan te gaan roeien per week?', 'Zo ja, hoe vaak per week?'], axis=1, inplace=True)

    df['Wat is je beste sportprestatie?'] = df['Wat is je beste sportprestatie?'].fillna('nothing')
    df['sportroeiprestatie'] = np.where(df['Wat is je beste sportprestatie?'].str.contains('roei'), 1, 0)
    df.drop(['Wat is je beste sportprestatie?'], axis=1, inplace=True)

    return df

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="PoRoRIT_Results.csv">Download als csv bestand</a>'
    return href

def fileupload():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)

        return df


df = pd.read_csv('raw_data/training_data.csv')
df = drop_private_cols(df)
df = transform_df(df)


bootjes = pd.read_csv('raw_data/people_with_boats.csv')
bootjes['Boten'] = bootjes['Boten'].str.split(',')
bootjes = bootjes[bootjes['Boten'].str.len() != 1]
bootjes.dropna(axis=0, inplace = True)

# bootjes['TopC4'] = np.where(bootjes['Boten'].astype(str).str.contains('TopC'), 1, 0)
# bootjes['Talenten'] = np.where(bootjes['Boten'].astype(str).str.contains('Talenten '), 1, 0)

bootjes['Club'] = np.where(bootjes['Boten'].astype(str).str.contains('Club'), 1, 0)
bootjes['EerstejaarsWed'] = np.where(bootjes['Boten'].astype(str).str.contains('Eerstejaars'), 1, 0)
bootjes['MiddengroepWed'] = np.where(bootjes['Boten'].astype(str).str.contains('Middengroep'), 1, 0)
bootjes['Ouderejaars'] = np.where(bootjes['Boten'].astype(str).str.contains('Ouderejaars'), 1, 0)
bootjes['OudeVier'] = np.where(bootjes['Boten'].astype(str).str.contains('Oude Vier'), 1, 0)

bootjes['goldLabel'] = np.where(bootjes.iloc[:, 4:].any(axis='columns'), 1, 0)
# indb = bootjes[bootjes.iloc[:, 4:].any(axis='columns')].index.tolist() # select only the rowers who have done at least one of the above
label = bootjes.loc[:, ['Name', 'goldLabel']]

enddf = pd.merge(label, df, how='inner', left_on='Name', right_on='Naam')
enddf.drop(['Naam', 'Name'], axis=1, inplace=True)
enddf['Gewicht (in kg)'] = enddf['Gewicht (in kg)'].astype(int)
enddf['Lengte (in cm)'] = enddf['Lengte (in cm)'].astype(int)
enddf.fillna(0, inplace=True)



scale_data = False
# als er andere vragen bijkomen bij het inschrijfformulier dan nu (03-08-2021), dan alle kolommen scalen
# aangezien er dus kolommen bijzitten die ik niet ken, en dan kan je ze maar beter scalen.
if len(enddf.columns) != 13:
    scale_data = True
if scale_data:
    allcols = enddf.columns
    x = enddf.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    enddf = pd.DataFrame(x_scaled)
    enddf.columns = allcols
    enddf.drop(['Lengte (in cm)', 'Gewicht (in kg)'], axis=1,
               inplace=True)  # gewicht and lengte appears to work negatively on the prediction score
else:
    enddf.drop(['Lengte (in cm)', 'Gewicht (in kg)'], axis=1,
               inplace=True)  # gewicht and lengte appears to work negatively on the prediction score

dffemale = enddf[enddf['Geslacht'] == 0].drop(['Geslacht'], axis=1)
dfmale = enddf[enddf['Geslacht'] == 1].drop(['Geslacht'], axis=1)

Xfem, yfem = dffemale.iloc[:,1:], dffemale.iloc[:,0]
Xmale, ymale = dfmale.iloc[:,1:], dfmale.iloc[:,0]

xg_regmale = xgb.XGBRegressor(objective ='binary:logistic')
xg_regfemale = xgb.XGBRegressor(objective ='binary:logistic')
xg_regmale.fit(Xmale,ymale)
xg_regfemale.fit(Xfem,yfem)



uploaded_file = st.file_uploader("Kies het RIT bestand")

if uploaded_file is not None:
    start_execution = st.button('Genereer voorspelling')
    if start_execution:
        inputdf = pd.read_csv(uploaded_file, delimiter = "\t")

        inputdf = inputdf.drop(['Geboortedatum', 'E-mailadres', 'Wat studeer je?', 'Telefoonnummer', 'Studentnummer'],
                               axis=1)  # privacy sensitive
        inputdf = drop_private_cols(inputdf)
        inputdf = transform_df(inputdf)
        inputdf.fillna(0, inplace = True)
        inputdf = inputdf.drop(['Lengte (in cm)','Gewicht (in kg)'],axis = 1) # gewicht and lengte appears to work negatively on the prediction score

        X_male_test_w_name = inputdf[inputdf['Geslacht'] == 1].drop(['Geslacht'], axis=1)
        X_female_test_w_name = inputdf[inputdf['Geslacht'] == 0].drop(['Geslacht'], axis=1)
        X_male_test = X_male_test_w_name.drop(['Naam'], axis = 1)
        X_female_test = X_female_test_w_name.drop(['Naam'], axis = 1)


        predmale = xg_regmale.predict(X_male_test)
        predfemale = xg_regfemale.predict(X_female_test)

        X_male_test_w_name['prediction'] = np.around(predmale, 5)
        X_female_test_w_name['prediction'] = np.around(predfemale, 5)
        test_set = pd.concat([X_male_test_w_name, X_female_test_w_name])
        test_set = test_set.loc[:, ['Naam','prediction']]
        test_set.sort_values(by='prediction', ascending = False, inplace = True)

        st.write(test_set)
        st.markdown(filedownload(test_set), unsafe_allow_html=True)

# hide_streamlit_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)
