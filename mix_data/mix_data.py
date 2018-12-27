# Veri setinin satırlarını random karıştırma
import random
with open('spambase.data','r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
print(data)
with open('../spambase_mix2.data','w') as target:
    for _, line in data:
        target.write( line )