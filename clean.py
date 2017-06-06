import pandas as pd
from ast import literal_eval
import numpy as np
import csv

df = pd.read_csv('yelp_academic_dataset_business.csv')

#filter for city/state/region
#df = df[(df['city']=='Urbana') | (df['city'] == 'Champaign')]
#df = df[(df['city'] == 'Montreal') | (df['city'] == 'Laval') | (df['city'] == 'Longueuil') | (df['city'] == 'Terrebonne') | (df['city'] == 'Repentigny') | (df['city'] == 'Brossard') | (df['city'] == 'Saint-Jerome') | (df['city'] == 'Blainville') | (df['city'] == 'Dollard-des-Ormeaux') | (df['city'] == 'Chateauguay')]
#df = df[df['city'] == 'Toronto']
df = df[df['state'] == 'BW'] #for Stuttgart

#y-attribute to predict
y_col = 'ethnic'

extra_cols = ['WiFi', 'DriveThru', 'DogsAllowed', 'Caters', 'NoiseLevel', 'RestaurantsReservations', 'Alcohol', 'HasTV', 'RestaurantsAttire', 'RestaurantsDelivery', 'OutdoorSeating', 'RestaurantsTakeOut', 'BusinessAcceptsCreditCards', 'RestaurantsPriceRange2', 'BikeParking', 'GoodForKids', 'RestaurantsGoodForGroups', 'GoodForDancing', 'RestaurantsTableService', 'WheelchairAccessible']

ambience_cols = ['romantic', 'intimate', 'classy', 'hipster', 'divey', 'touristy', 'trendy', 'upscale', 'casual']

meal_cols = ['dessert', 'latenight', 'lunch', 'dinner', 'breakfast', 'brunch']

parking_cols = ['garage', 'street', 'validated', 'lot', 'valet']

existing_cols = ['business_id', 'latitude', 'longitude', 'stars', 'review_count']

def is_nan(x):
    return (x is np.nan or x != x)

def eval_string(cell):
    try:
        result = literal_eval(cell)
#        print result
        return result
    except:
#        print cell
        return np.nan

#convert string to list
df['attributes'] = df['attributes'].apply(eval_string)

#convert categories to string
df['categories'] = df['categories'].apply(eval_string)

def cat_exist(cell, cat):
    if is_nan(cell):
        return 0
    if cat in cell:
        return 1
    else:
        return 0

#filter dataframe for those rows in which 'Restaurants' appears in 'categories'
df['restaurant'] = df['categories'].apply(lambda x: cat_exist(x, 'Restaurants'))
print 'all biz: ', len(df)
df = df[df['restaurant']==1]
print 'rests only: ', len(df)

#the property we want to predict
df[y_col] = df['categories'].apply(lambda x: cat_exist(x, 'Ethnic Food')) 

#extra category for bars, for investigating whether bars tend to be 'new american'
z_col = 'bars'
df[z_col] = df['categories'].apply(lambda x: cat_exist(x, 'Bars')) 


#this works for non-nested attributes. Goes through list of attributes and returns value corresponding to att_name
def break_list(cell, att_name):
    if is_nan(cell):
        return np.nan
    else:
        for item in cell:
            name, _, val = item.partition(': ')
            if name == att_name:
                return val            

#this function finds the string after the colon and attribute name in Ambience, GoodForMeal, and BusinessParking (these are the gen_att_name), and assigns this as the value of that attribute in a new column labelled by spec_att_name.
def extract_value(cell, gen_att_name, spec_att_name):
    if is_nan(cell):
        return np.nan
    else:
        for item in cell:
            name, _, val = item.partition(': ')
            if name == gen_att_name:
                val_dict = literal_eval(val)
                try:
                    return val_dict[spec_att_name]
                except:
                    return np.nan

for col_name in extra_cols:
    df[col_name] = df['attributes'].apply(lambda x: break_list(x, col_name))

for col_name in ambience_cols:
    df[col_name] = df['attributes'].apply(lambda x: extract_value(x, 'Ambience', col_name))

for col_name in meal_cols:
    df[col_name] = df['attributes'].apply(lambda x: extract_value(x, 'GoodForMeal', col_name))

for col_name in parking_cols:
    df[col_name] = df['attributes'].apply(lambda x: extract_value(x, 'BusinessParking', col_name))

for col in parking_cols + meal_cols + ambience_cols + extra_cols:
    if col !=  'RestaurantsPriceRange2': #price range is the only already numeric column in all this.
#    if col=='Alcohol': #for checking cat to number mapping
        df[col] = df[col].astype('category')
#        df['delcode'] = df[col].cat.codes
        df[col] = df[col].cat.codes

#for checking cat to number mapping
#print df[['delcode', 'Alcohol']]
        
df['RestaurantsPriceRange2'] = pd.to_numeric(df['RestaurantsPriceRange2'], errors='coerce')
        
df[existing_cols + parking_cols + meal_cols + ambience_cols + extra_cols + [y_col] + [z_col]].to_csv('ethnic_bars_stuttgart_restaurants_data.csv', index=False, quoting=csv.QUOTE_NONE)

## for col in parking_cols + meal_cols + ambience_cols + extra_cols:
##     print col, df[col].mean()

#print df['specialty'].mean()
