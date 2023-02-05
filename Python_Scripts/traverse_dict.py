import numpy as np
import math


obj = {"name": "Apple", "childrenB" : {"name" : "B"}, 
       "childrenB" : {"name" : "B"}, 
       "childrenC": {"name" :"C", "childrenD": {}}}


def traverse(obj):
    for k, v in obj.items():
        if type(v) == dict:
            traverse(v)
        else:
            print(v)

# obj['name']
# obj['childrenB']['name']
# obj['childrenC']['childrenD']
# another test case
# D1={1: {2: {3: 4, 5: 6}, 3: {4: 5, 6: 7}}, 2: {3: {4: 5}, 4: {6: 7}}}
traverse(obj)

#for k,v in obj.items():
 #   print(v)
    
