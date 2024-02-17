from bs4 import BeautifulSoup
import os
import pandas as pd
import re
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np

pd.set_option('display.min_rows', 800)
pd.set_option('display.max_rows', 800)


                
def findAddress(house):
    return house.find('h2',class_='sold-property-listing__heading qa-selling-price-title hcl-card__title')

def findLocation(house):
    return house.find('div',class_='sold-property-listing__location').find('div').contents[2]

def findDateOfSale(house):
    return house.find('span', class_='hcl-label hcl-label--state hcl-label--sold-at').text.strip()[5:]
'''
def findAreaAndRooms(house):
    area = house.find('div', class_='sold-property-listing__subheading sold-property-listing__area').text.strip().replace('\n','').replace(' ','') 
    boarea = re.match(r'\d+', area)
    totalArea = re.match(r'\d+[+]\d+', area)
    rooms = re.search(r'(\d)\s+rum', area)

    if boarea:
        boarea = int(boarea.group())
    else:
        boarea = 'NaN'

    if totalArea:
        totalArea = totalArea.group()
    else:
        totalArea = 'NaN'

    if rooms:
        rooms = int(rooms.group(1))
    else:
        rooms = 'NaN'

    return boarea, totalArea, rooms
'''
def findPlotArea(house):
    plot = house.find('div', class_='sold-property-listing__land-area')
    if plot is None:
        return np.NaN
    plotArea = plot.text.strip().replace('2&nbsp;','').replace('m² tomt','')
    return plotArea

def findClosingPrice(house):
    return int(house.find('span', class_='hcl-text hcl-text--medium').text.strip().replace('Slutpris ','').replace('kr','').replace('\xa0', ''))


def findAreaAndRooms(house):
    area = house.find('div', class_='sold-property-listing__subheading sold-property-listing__area').text.strip().replace('\n','').replace(' ','')
    if totalArea := re.match(r'\d+[+]\d+', area):
        totalArea = totalArea.group().replace('\xa0','')
        tmp = totalArea.split('+')
        boarea = int(tmp[0])
        biarea = int(tmp[1])
        totalArea = boarea + biarea
    elif boarea := re.match(r'(\d+)\s*m²', area):
        totalArea = np.NaN
        boarea = int(boarea.group(1))
    else:
        boarea = np.NaN
        totalArea = np.NaN

    if rooms := re.search(r'(\d+)\s+rum', area):
        rooms = int(rooms.group(1))
    else:
        rooms = np.NaN
    return boarea, totalArea, rooms
    

    
df = pd.DataFrame()



i = 0
for filename in os.listdir('kungalv_slutpriser/'):
    file = open('kungalv_slutpriser/'+filename, encoding='UTF-8')
    soup = BeautifulSoup(file, 'html.parser')
    table = soup.find('ul',id='search-results')
    for house in table.find_all('li', class_='sold-results__normal-hit'):
        df.loc[i, 'Address'] = findAddress(house).text.strip()
        df.loc[i,'Location'] = findLocation(house).text.strip().replace('\n','') 
        df.loc[i, 'Date of sale'] = findDateOfSale(house)
        df.loc[i, 'Boarea (m²)'], df.loc[i, 'Total Area (m²)'], df.loc[i, 'Rooms']  = findAreaAndRooms(house)
        #df.loc[i, 'Plot Area (m²)'] = house.find('div', class_='sold-property-listing__land-area').text.strip().replace('2&nbsp;','').replace('m² tomt','')
        df.loc[i, 'Plot Area (m²)'] = findPlotArea(house)
        df.loc[i, 'Closing price (kr)'] = findClosingPrice(house)

        i += 1


df.to_csv('result.csv')

print(df)
df_2022 = df[df['Date of sale'].str.contains('2022')]
df_2022.loc[:,'Closing price (kr)'] = df_2022['Closing price (kr)'].divide(1000000)

#cp_2022_np = df_2022['Closing price (kr)'].to_numpy()
#Histogram
fig1, ax1 = plt.subplots()
ax1.hist(df_2022['Closing price (kr)'], 'auto', rwidth=0.95)
ax1.set_xlabel('Sale price (million kr)')
ax1.set_ylabel('Number of sales')

ax1.set_xticks([2,3,4,5,6,7,8,9,10])
plt.savefig('Figure 1.pdf')

#Scatterplot
fig2, ax2 = plt.subplots()
ax2.scatter(df_2022['Boarea (m²)'], df_2022['Closing price (kr)'], s=8)
ax2.set_xlabel('Boarea (m²)')
ax2.set_ylabel('Price (kr)')
plt.savefig('Figure 2.pdf')

fig3, ax3 = plt.subplots()
df_2022_NaN = df_2022[df_2022['Rooms'].isna()]

#colors = ['FFFF33','FF9933','80FF00', '00FFFF','FF0000','FF00FF','0000FF', '4C0099','000000','000000']
custom_cycler = (cycler(color=['darkgrey', 'darkorange','red','lime','seagreen','lightskyblue' ,'royalblue','violet','darkviolet','black']))
ax3.set_prop_cycle(custom_cycler)
plt.scatter(df_2022_NaN['Boarea (m²)'], df_2022_NaN['Closing price (kr)'], label='N/A', s=8)
for i in range(int(df_2022['Rooms'].min()), int(df_2022['Rooms'].max())+1):
    ys = df_2022.query(f'Rooms == {i}')
    plt.scatter(ys['Boarea (m²)'], ys['Closing price (kr)'],label=f'{i} Rooms', s=8)
plt.xlabel('Boarea (m²)')
plt.ylabel('Price (kr)')
plt.legend()
plt.savefig('Figure 3.pdf')
plt.show()
