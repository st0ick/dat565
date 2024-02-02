from bs4 import BeautifulSoup
import os
import pandas as pd
import re

pd.set_option('display.max_rows', None)


def removeWhiteSpace(location):
    newLocation = ""
    for (i, char) in enumerate(location):
        if char != ' ':
            newLocation += str(char)
        else:
            newLocation += ' '
            for j, char in enumerate(location[i:]):
                if char != ' ':
                    newLocation += location[j+i:]
                    return newLocation
                
def findAddress(house):
    return house.find('h2',class_='sold-property-listing__heading qa-selling-price-title hcl-card__title')

def findLocation(house):
    return house.find('div',class_='sold-property-listing__location').find('div').contents[2]

def findDateOfSale(house):
    return house.find('span', class_='hcl-label hcl-label--state hcl-label--sold-at').text.strip()[5:]

def findAreaAndRooms(house):
    area = house.find('div', class_='sold-property-listing__subheading sold-property-listing__area').text.strip().replace('\n','').replace(' ','') 

    if boarea := re.match(r'\d*', area):
        boarea = boarea.group()
    else:
        boarea = 'NaN'

    if totalArea := re.match(r'\d*[+]\d*', area):
        totalArea = totalArea.group()
    else:
        totalArea = 'NaN'

    if rooms := re.search(r'(\d)\s+rum', area):
        rooms = rooms.group(1)
    else:
        rooms = 'Nan'

    return boarea, totalArea, rooms

def findPlotArea(house):
    plot = house.find('div', class_='sold-property-listing__land-area')
    if plot is None:
        return 'Nan'
    plotArea = plot.text.strip().replace('2&nbsp;','').replace('m² tomt','')
    return plotArea


#xf = 2
#file = open(f"kungalv_slutpriser/kungalv_slutpris_page_0{xf}.html", encoding='UTF-8')




df = pd.DataFrame()

#s = soup.find('ul',id='search-results')
i = 0
for filename in os.listdir('kungalv_slutpriser/'):
    file = open('kungalv_slutpriser/'+filename, encoding='UTF-8')
    soup = BeautifulSoup(file, 'html.parser')
    table = soup.find('ul',id='search-results')
    for house in table.find_all('li', class_='sold-results__normal-hit'):
        df.loc[i, 'Address'] = findAddress(house).text.strip()
        df.loc[i,'Location'] = findLocation(house).text.strip().replace('\n','') 
        df.loc[i, 'Date of sale'] = findDateOfSale(house)
        df.loc[i, 'Bo area (m²)'], df.loc[i, 'Total Area (m²)'], df.loc[i, 'Rooms']  = findAreaAndRooms(house)
        #df.loc[i, 'Plot Area (m²)'] = house.find('div', class_='sold-property-listing__land-area').text.strip().replace('2&nbsp;','').replace('m² tomt','')
        df.loc[i, 'Plot Area (m²)'] = findPlotArea(house)
        df.loc[i, 'Closing price (kr)'] = house.find('span', class_='hcl-text hcl-text--medium').text.strip().replace('Slutpris ','').replace('2&nbsp;','').replace('kr','')

        i += 1


print(df)
df.to_csv('result.csv')







