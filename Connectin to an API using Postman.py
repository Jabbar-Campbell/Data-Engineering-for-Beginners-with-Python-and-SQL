#                                          Step 1: Choose an API
# Visit RapidAPI or GitHub's Public APIs.
# Select an API of your choice that provides interesting data for your ETL project. Ensure that the API is free to use.
# Document the chosen API's base URL, endpoints, and any required authentication (?if applicable). You will need this information to connect to the API.

#                                          Step 2: Connect to the API
#Use Python to connect to the chosen API. You can use libraries like requests to make API requests.

#Retrieve data from the API by sending appropriate HTTP requests to the specified endpoints.
#Store the API response in a Python variable.

#                                         Step 3: Load Data into Pandas
#Import the pandas library in your Python script.
#Create a pandas DataFrame using the data obtained from the API response.
#Explore the data by displaying the first few rows and checking the data types.

#                                         Step 4: Perform Data Transformations
#Apply at least three data transformations to the DataFrame. Transformations can include filtering, sorting, grouping, or creating new columns.
#Document each transformation step in your code and provide comments explaining the purpose of each transformation.

#                                          Step 5: Save as CSV
#Save the transformed data as a CSV (Comma-Separated Values) file.
#Choose a meaningful name for the CSV file that reflects the data and the transformations performed.
#Specify the directory on your local machine where you want to save the CSV file.



# Instagram is part of facebook so you need to log into your facebook and then go the the facebook
# for developers page and start an app in order to generate  a token
# a facebook business account will have its own token
# in our case we use our own. The Facebook Graph API will generate this for you
# unfortunately it wont suggest code for python thats been discontinued by facebook

    # Access token
    #EAAEkW7n34icBOwjtJ4fWtZA9De5joDrppfFtIb4LUuR1BFRfVwy4EU9uGNnf821WiWFMK4xVeADkM0MZB5ToHHa2oGx69ZBGKZABiS718wQZBf1nGzxtSRGPdT9dZB7JKy9dUDQ4OvzDxBLNmbwEolVXKLZC5rfqS7rxeXicmS2ZCZAmeBQYvFKisnZCD0HerBlLI8r0rrfI0KTjvLJjFxHkxZBePcDfVnZAwYupCrV5c7QWZC4r97SYJTOpoUdfma9UZD

    # Facebook id
    # "id: 10161346497628307"
 
    # Facebook name
    # Jabbar Campbell


# something easier would be  connecting to less secure public API called deck of cards
# https://www.deckofcardsapi.com/api/deck/new/shuffle/?deck_count=1

import requests

url = "https://openlibrary.org/search/authors.json?q=twain"

#payload = {}
#headers = {}

#response = requests.request("GET", url, headers=headers, data=payload)

#print(response.text)




r = requests.get(url)
x = r.json()
df = pd.DataFrame(x)
df=df.iloc[1: , :]

 # iterating the columns
#for col in df.columns:
 #   print(col)


# list(data) or
#list(df.columns)



df=df['docs'].apply(pd.Series)

df = df.drop('type', axis=1)
df

df.to_csv(my_path_books.csv, encoding='utf-8', index=False)