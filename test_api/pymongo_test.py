from pymongo import MongoClient
from pprint import pprint

client = MongoClient()
db = client.mDB
print (db)

# get collection
name = db["customer"]
print (name)
name_customer = {"cus_name": "AAA",
                 'age': 18}

name.insert_one(name_customer)

# Query for the inserted document.
Queryresult = name.find_one({'cus_name': "AAA"})
pprint(Queryresult)


