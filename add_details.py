import pymongo
from pymongo import MongoClient

# MongoDB Atlas connection URI
client = MongoClient("mongodb+srv://piyushhole:Piyushhole2001@ecom.neu3z5n.mongodb.net/users?retryWrites=true&w=majority")

# Select database and collection
db = client["users"]
collection = db["plantslist"]

# Region data with coordinates
regions = {
    "Western Himalaya": {"coordinates": [78.6569, 32.9655]},
    "Eastern Himalaya": {"coordinates": [88.6145, 27.3207]},
    "Western Ghats": {"coordinates": [73.8372, 15.8150]},
    "Eastern Ghats": {"coordinates": [80.2415, 17.6910]},
    "Uttarakhand": {"coordinates": [78.9629, 30.0668]}
}

# Plant data with region information and Wikipedia link
plantslist = [
    {
        "name": "Arive-Dantu",
        "description": "Amaranthus viridis is an annual herb with an upright, light green stem that grows to about 60–80 cm in height. Numerous branches emerge from the base, and the leaves are ovate, 3–6 cm long, 2–4 cm wide, with long petioles of about 5 cm. The plant has terminal panicles with few branches, and small green flowers with 3 stamens.",
        "leafImages": [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Amaranthus_viridis_25042014_1.jpg/330px-Amaranthus_viridis_25042014_1.jpg"
        ],
        "regions": ["Eastern Himalaya","Western Ghats"],  # Example of one region
        "wikipediaLink": "https://en.wikipedia.org/wiki/Amaranthus_viridis"
    },
    {
        "name": "Basale",
        "description": "Basale has an anti-inflammatory activity and wound healing ability. It can be helpful as a first aid, and the leaves of this plant can be crushed and applied to burns, scalds, and wounds to help in healing of the wounds.",
        "leafImages": [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Basella_alba_leaves_27052014.jpg/1280px-Basella_alba_leaves_27052014.jpg"
        ],
        "regions": ["Western Ghats","Eastern Ghats","Eastern Himalaya"],  # Single region for this plant
        "wikipediaLink": "https://en.wikipedia.org/wiki/Basella_alba"
    }
]

# Insert data into MongoDB collection
for plant in plantslist:
    region_names = plant.get("regions", [])
    plant["locations"] = []  # Prepare an empty list for multiple locations
    for region_name in region_names:
        if region_name in regions:
            # Append each region's coordinates to the "locations" array
            plant["locations"].append({
                "type": "Point",
                "coordinates": regions[region_name]["coordinates"]
            })
    if not plant["locations"]:
        plant["locations"] = None  # Handle cases where no regions are found

# Insert the plants into the database
result = collection.insert_many(plantslist)

# Print the inserted IDs to confirm success
print("Data inserted with record ids", result.inserted_ids)
