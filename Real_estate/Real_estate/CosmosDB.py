from azure.cosmos import CosmosClient, exceptions
import os
import streamlit as st

# Configure your Cosmos DB connection
if os.getenv("Azure_Cosmos_EndPoint"):
    endpoint = os.getenv("Azure_Cosmos_EndPoint")
else:
    endpoint = st.secrets["Azure_Cosmos_EndPoint"]
if os.getenv("Azure_Cosmos_Key"):
    key = os.getenv("Azure_Cosmos_Key")
else:
    key = st.secrets["Azure_Cosmos_Key"]

database_name = "RealEstate"
container_name = "Properties"

# Initialize the Cosmos client
client = CosmosClient(endpoint, key)

# Get a reference to the database
database = client.get_database_client(database_name)

# Get a reference to the container
container = database.get_container_client(container_name)

# Define a query
query = "SELECT * FROM c Where [Property Type]= 'Villa/House'"

try:
    # Execute the query
    items = container.query_items(query, enable_cross_partition_query=True)

    # Process the query results
    for item in items:
        print(item)

except exceptions.CosmosHttpResponseError as e:
    print("Error:", e)
