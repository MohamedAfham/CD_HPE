import json

with open('train_cover1.json', 'r') as f:
    train_cover1 = json.load(f)
    
with open('train_cover2.json', 'r') as f:
    train_cover2 = json.load(f)
    
train_cover = train_cover1 + train_cover2

with open('train_cover.json', 'w') as f:
    json.dump(train_cover, f)
