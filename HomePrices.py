from bs4 import BeautifulSoup
import pandas as pd
import re

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
    boarea = re.match(r'\d*', area).group()
    rooms = re.search(r'(\d)\s+rum', area).group(1)
    if totalArea := re.match(r'\d*[+]\d*', area):
        return boarea, totalArea.group(), rooms
    
    return boarea, 'NaN', rooms


xf = 1
f = open(f"kungalv_slutpriser/kungalv_slutpris_page_0{xf}.html", encoding='UTF-8')




soup = BeautifulSoup(f, 'html.parser')
x = 1
a = chr(x)

addresses = []

df = pd.DataFrame()

s = soup.find('ul',id='search-results')
#for i, house in enumerate(s.find_all('li', class_='sold-results__normal-hit')):
#    addresses.append( house.find('h2',class_='sold-property-listing__heading qa-selling-price-title hcl-card__title').text.strip()  )
#df.loc[:,'Address'] = addresses

for i, house in enumerate(s.find_all('li', class_='sold-results__normal-hit')):
    df.loc[i, 'Address'] = findAddress(house).text.strip()
    df.loc[i,'Location'] = findLocation(house).text.strip().replace('\n','') 
    df.loc[i, 'Date of sale'] = findDateOfSale(house)
    df.loc[i, 'Bo area (m²)'], df.loc[i, 'Total Area (m²)'], df.loc[i, 'Rooms']  = findAreaAndRooms(house)
    df.loc[i, 'Plot Area (m²)'] = house.find('div', class_='sold-property-listing__land-area').text.strip().replace('2&nbsp;','').replace('m² tomt','')


test = s.find('div', class_='sold-property-listing__subheading sold-property-listing__area').text.replace('\n','').replace(' ','').strip()

#m = re.match(r'\brum\b', test)
print(df)
#print(re.search(r'(\d)\s+rum', test).group(1))






