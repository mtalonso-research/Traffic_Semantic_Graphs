import os
import json

path="./data/NuPlan/"
path2train=os.path.join(path, "training_data/clean/graphs/")
path2test=os.path.join(path, "evaluation_data/clean/graphs/")

unique_cities = set()
for filename in os.listdir(path2train):
    with open(os.path.join(path2train, filename), 'r') as f:
        scene = json.load(f)
        scene_meta = scene["metadata"]
        city = scene_meta["city"].split("_")[-1]
        unique_cities.add(city)

for filename in os.listdir(path2test):
    with open(os.path.join(path2test, filename), 'r') as f:
        scene = json.load(f)
        scene_meta = scene["metadata"]
        city = scene_meta["city"].split("_")[-1]
        unique_cities.add(city)

print("Unique cities in training data:", unique_cities)
# answer is {'singapore', 'boston', 'pittsburgh'}