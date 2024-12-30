import json
import random

# 创建包含64个独特prompt和随机seed的字典
objects_and_scenes = [
    "A peaceful mountain lake during sunrise",
    "A bustling medieval marketplace",
    "An astronaut planting a flag on Mars",
    "A cozy library filled with ancient books",
    "A mysterious castle on a foggy hill",
    "A vibrant coral reef under the ocean",
    "A futuristic city with flying cars",
    "A serene Japanese garden in autumn",
    "A spaceship approaching a distant planet",
    "A magical forest with glowing mushrooms",
    "A panda eating bamboo in a lush forest",
    "A stormy sea with a lone ship braving the waves",
    "A quiet snow-covered village at dusk",
    "A desert oasis with palm trees and clear water",
    "A robot building a sandcastle on the beach",
    "A tiger prowling through a dense jungle",
    "A charming Italian village by the seaside",
    "A dragon perched atop a mountain peak",
    "A bustling New York street in the 1920s",
    "A serene waterfall surrounded by mossy rocks",
    "A child discovering a hidden treasure chest",
    "A vibrant flower field under a rainbow",
    "A medieval knight standing in a lush meadow",
    "An ancient temple hidden in the jungle",
    "A dolphin leaping out of the ocean at sunset",
    "A serene winter landscape with frosted trees",
    "A farmer working in a golden wheat field",
    "A colorful hot air balloon floating above mountains",
    "A forest clearing lit by magical fireflies",
    "A polar bear walking on the Arctic ice",
    "A modern art museum with abstract sculptures",
    "A fisherman casting a net in a tranquil river",
    "A group of penguins sliding on icy slopes",
    "A cheerful sunflower field under a bright sun",
    "A caravan traveling across the Sahara desert",
    "A steampunk city with intricate machinery",
    "A mystical cave filled with glowing crystals",
    "A peaceful island with a single palm tree",
    "A cheetah running through an open savannah",
    "A bustling coffee shop in a rainy city",
    "A mysterious figure in a hooded cloak",
    "A garden party with lanterns and music",
    "A snowy cabin with smoke rising from the chimney",
    "A beautiful sunset over the Grand Canyon",
    "A scientist working in a futuristic lab",
    "A lush green valley with a meandering river",
    "A grand ballroom with dancers in elegant attire",
    "A quiet beach with gentle waves and seashells",
    "A neon-lit street in a cyberpunk city",
    "A flock of birds migrating at sunrise",
    "A herd of elephants walking across the savannah",
    "A magical bridge leading to an enchanted land",
    "A lighthouse guiding ships during a storm",
    "A serene monastery in the mountains",
    "A bakery filled with delicious pastries",
    "A wolf howling under a full moon",
    "A tropical rainforest with colorful parrots",
    "A scientist exploring a deep underwater trench",
    "A medieval village with cobblestone streets",
    "A vibrant carnival with colorful floats",
    "A rustic barn surrounded by farmland",
    "A train traveling through a snowy landscape",
    "A giant whale swimming through the open ocean",
    "A peaceful meadow with grazing deer"
]

data = {
    str(i + 1): {
        "prompt": objects_and_scenes[i],
        "seed": random.randint(100000, 999999)
    }
    for i in range(len(objects_and_scenes))
}

# 保存到文件
output_path = "./prompt.json"
with open(output_path, "w") as file:
    json.dump(data, file, indent=4)

output_path
