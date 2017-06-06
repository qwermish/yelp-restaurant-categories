import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('newam_pgh.csv')

eth_df = df[df['new_american']==1]
df = df[df['new_american']==0]

sns.set()

#THIS PART IS FOR PLOTTING HISTOGRAMS, for continuously distributed data

plt.hist(eth_df['review_count'], normed=True, label='=mediterranean', alpha=0.5)
plt.hist(df['review_count'], normed=True, label='not mediterranean', alpha=0.5)
#plt.yscale('log')
plt.legend()
plt.show()

#THIS PART IS FOR PLOTTING BAR CHARTS FROM DF COLUMNS, for discrete data

## fig, ax = plt.subplots(1,2)

## feature = 'RestaurantsTableService'
## sns.countplot(eth_df[feature], ax = ax[0])
## sns.countplot(df[feature], ax = ax[1])

## ax[0].set_title('New American Restaurants in Pittsburgh')
## ax[1].set_title('Non-New American Restaurants in Pittsburgh')

## #plt.setp(ax, xticklabels = ['unknown', 'beer & wine', 'full bar', 'no alcohol'])
## plt.setp(ax, xticklabels = ['unknown', 'no table service', 'table service'])

## #auto formats xticks labels to not overlap
## fig.autofmt_xdate()

## fig.savefig('pgh_newam_stars.png')


