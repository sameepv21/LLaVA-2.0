import json
import random

# Load the data from llava_instruct_80k.json
with open('llava_instruct_80k.json', 'r') as file:
    data = json.load(file)

# Randomly sample 5000 data points
sampled_data = random.sample(data, 5000)

# Save the sampled data to llava_instruct_5k.json
with open('llava_instruct_5k.json', 'w') as file:
    json.dump(sampled_data, file, indent=4)
