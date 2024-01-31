from bs4 import BeautifulSoup
import pandas as pd

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
            

f = open('kungalv_slutpriser/kungalv_slutpris_page_01.html', encoding='UTF-8')




soup = BeautifulSoup(f, 'html.parser')
x = 1
a = chr(x)

addresses = []

df = pd.DataFrame(columns=['Address', 'Location', 'Date of sale'])

s = soup.find('ul',id='search-results')
#for i, house in enumerate(s.find_all('li', class_='sold-results__normal-hit')):
#    addresses.append( house.find('h2',class_='sold-property-listing__heading qa-selling-price-title hcl-card__title').text.strip()  )
#df.loc[:,'Address'] = addresses

for i, house in enumerate(s.find_all('li', class_='sold-results__normal-hit')):
    df.loc[i, 'Address'] = house.find('h2',class_='sold-property-listing__heading qa-selling-price-title hcl-card__title').text.strip()
    df.loc[i,'Location'] = removeWhiteSpace(house.find('div',class_='sold-property-listing__location').find('div').contents[2].text.strip().replace('\n',''))


test = s.find('div',class_='sold-property-listing__location').find('div').contents[2].text.strip().replace('\n','')
print(df)



