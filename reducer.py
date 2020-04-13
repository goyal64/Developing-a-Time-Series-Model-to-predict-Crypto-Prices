#!/usr/bin/python

import sys
countTotal = 0
old1 = 'None'
y1 = None
y2 = None
y3 = None

# Loop around the data
# It will be in the format key \t value
for line in sys.stdin:
    data_mapped = line.strip().split("\t")
    
    if len(data_mapped) != 2:
        # Something is wrong. Skip this line.'
        continue
        
    thisKey, Price = data_mapped
    
    if old1 != thisKey:
        
        if countTotal == 0:
            
            y2 = float(Price)
            old1 = thisKey
            
        elif countTotal == 1:
            
            y1 = float(Price)
            old1 = thisKey
            
        elif countTotal == 2:
            
            y3 = float(Price)
            old1 = thisKey
            
              
        elif countTotal >= 3:
            
            y = 0.00007 -(0.0395*(y1-y2)) - (0.0342*(y2-y3)) + y1
            
            print thisKey, "\t", Price, "\t", y

            y3 = float(y2)
            y2 = float(y1)
            y1 = float(Price)
            
            
    countTotal += 1

if old1 != 'None':
    print thisKey, "\t", y
