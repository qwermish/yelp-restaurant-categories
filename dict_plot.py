import seaborn as sns
import matplotlib.pyplot as plt

#THIS PLOTS BAR CHARTS FROM DICTS (FOR IMPORTANCE GRAPHS)

fig, ax = plt.subplots()

D = {'Alcohol': 84, 'HasTV': 110, 'NoiseLevel': 36, 'RestaurantsAttire': 24, 'WheelchairAccessible': 119, 'BusinessAcceptsCreditCards': 18, 'street': 29, 'breakfast': 6, 'RestaurantsGoodForGroups': 19, 'casual': 89, 'review_count': 461, 'Caters': 62, 'WiFi': 98, 'RestaurantsReservations': 106, 'valet': 8, 'RestaurantsTableService': 151, 'RestaurantsTakeOut': 75, 'GoodForKids': 191, 'DriveThru': 26, 'DogsAllowed': 24, 'stars': 154, 'intimate': 1, 'dessert': 8, 'OutdoorSeating': 61, 'lunch': 60, 'RestaurantsPriceRange2': 171, 'lot': 47, 'RestaurantsDelivery': 96, 'brunch': 52, 'latenight': 6, 'romantic': 5, 'BikeParking': 41, 'garage': 61, 'dinner': 20, 'GoodForDancing': 127}


plt.bar(range(len(D)), D.values(), align='center')
plt.xticks(range(len(D)), D.keys(), rotation=90)
plt.tight_layout()
plt.title('Important features for predicting New American restaurants in Pittsburgh')
plt.ylabel('Importances')

plt.show()
