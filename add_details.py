import pymongo
from pymongo import MongoClient

# MongoDB Atlas connection URI
client = MongoClient("mongodb+srv://piyushhole:Piyushhole2001@ecom.neu3z5n.mongodb.net/users?retryWrites=true&w=majority")

# Select database and collection
db = client["users"]
collection = db["plantslist"]

# Region data with coordinates
regions = {
    # Maharashtra
    "Nasik": {"coordinates": [73.7900, 19.9975]},
    "Pune": {"coordinates": [73.8567, 18.5214]},

    # Uttar Pradesh
    "Allahabad": {"coordinates": [81.8463, 25.4484]},
    "Agra": {"coordinates": [77.9680, 27.1767]},

    # Punjab
    "Ludhiana": {"coordinates": [75.8573, 30.9009]},

    # Haryana
    "Hisar": {"coordinates": [75.5774, 29.1492]},

    # Andhra Pradesh
    "Krishna": {"coordinates": [80.2182, 16.2007]},
    "Guntur": {"coordinates": [80.4395, 16.3060]},

    # Tamil Nadu
    "Kanyakumari": {"coordinates": [77.0580, 8.0889]},
    "Madurai": {"coordinates": [78.1198, 9.9252]},

    # Karnataka
    "Bangalore": {"coordinates": [77.5946, 12.9716]},
    "Mysuru": {"coordinates": [76.6394, 12.2958]}
}

# Plant data with region information and Wikipedia link
plantslist = [
    {
        "name": "Guava",
        "description" : "Guava leaves (Psidium guajava) are highly regarded for their numerous health benefits and medicinal properties. Rich in antioxidants, these leaves possess anti-inflammatory and antimicrobial properties that can help combat infections and promote overall health. Guava leaves are known for their ability to improve digestive health by alleviating symptoms such as diarrhea and indigestion, thanks to their tannin content, which has astringent properties. They are also recognized for their potential in managing blood sugar levels, making them beneficial for individuals with diabetes. Additionally, guava leaves contain essential nutrients, including vitamins A, C, and E, which contribute to skin health and may help in reducing acne and other skin issues. The leaves can be brewed into a tea or used in extracts, offering a natural remedy for various ailments. Their anti-diabetic properties, combined with their ability to enhance gut health, make guava leaves a valuable addition to traditional medicine.",
        "leafImages": [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Goya_blancs.JPG/1920px-Goya_blancs.JPG"
        ],
        "regions": [
            "Jaipur", "Jodhpur",  # Rajasthan
            "Ahmedabad", "Surat",  # Gujarat
            "Amritsar", "Ludhiana",  # Punjab
            "Hisar", "Rohtak",  # Haryana
            "Indore", "Bhopal",  # Madhya Pradesh
            "Bangalore", "Mysuru", "Chitradurga",  # Karnataka
            "Pune", "Nagpur",  # Maharashtra
            "Allahabad", "Agra",  # Uttar Pradesh
            "Nasik"  # Maharashtra
        ],        # Example of one region
        "wikipediaLink": "https://en.wikipedia.org/wiki/Guava"
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
