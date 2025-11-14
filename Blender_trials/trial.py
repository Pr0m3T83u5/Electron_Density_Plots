import csv
import numpy as np

fp = r"C:\Users\ACER\Documents\Python_Scripts\Atomic_Structure\QNfirst.csv"


Koordinates = []

with open( fp ) as csvfile:
    rdr = csv.reader(csvfile, delimiter=',')
    data = [[float(line[0]),float(line[1]),float(line[2])] for line in rdr]
   
    #data_kart = [[float(line[0])*np.sin(float(line[1]))*np.cos(float(line[2])),float(line[0])*np.sin(float(line[1]))*np.sin(float(line[2])), float(line[0])*np.cos(float(line[1]))] for line in rdr]

# print(data_kart[0])

# for x,y,z in data_kart:
#     print(x,y,z)
print(np.mean(data, axis=0))
    
        
    
