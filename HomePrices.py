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

def findArea(house):
    area = house.find('div', class_='sold-property-listing__subheading sold-property-listing__area').text.strip()
    return area


xf = 1
f = open(f"kungalv_slutpriser/kungalv_slutpris_page_0{xf}.html", encoding='UTF-8')




soup = BeautifulSoup(f, 'html.parser')
x = 1
a = chr(x)

addresses = []

df = pd.DataFrame(columns=['Address', 'Location', 'Date of sale', 'Area'])

s = soup.find('ul',id='search-results')
#for i, house in enumerate(s.find_all('li', class_='sold-results__normal-hit')):
#    addresses.append( house.find('h2',class_='sold-property-listing__heading qa-selling-price-title hcl-card__title').text.strip()  )
#df.loc[:,'Address'] = addresses

for i, house in enumerate(s.find_all('li', class_='sold-results__normal-hit')):
    df.loc[i, 'Address'] = findAddress(house).text.strip()
    df.loc[i,'Location'] = findLocation(house).text.strip().replace('\n','') 
    df.loc[i, 'Date of sale'] = findDateOfSale(house)
    df.loc[i, 'Area'] = findArea(house)


test = s.find('div',class_='sold-property-listing__location').find('div').contents[2].text.strip().replace('\n','')
print(df)



