# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 20:44:29 2022

@author: Roman
"""

import random
import math
import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap

class Knn():
    __savedData = []
    def __init__(self, neighbors = 5):
        self.neighbors = neighbors
    def fit(self,data):
        for m in data:
            self.__savedData.append(m)
        
    def predict(self,data):
        __predictionList = []
        for row in data:
            __diss = []
            __nearestNeighbors = []
            __classesDict = {}
            __mx = 0
            __predictedClass = -1
            for datum in self.__savedData:
                sm = [(row[i]-datum[i])**2 for i in range(len(row))] 
                __diss.append([math.sqrt(sum(sm)), datum[len(datum)-1]])
                
            __nearestNeighbors = sorted(__diss)[:self.neighbors]
            for i in range(len(__nearestNeighbors)):
                try:
                    __classesDict[__nearestNeighbors[i][1]] += 1
                except KeyError:
                        __classesDict[__nearestNeighbors[i][1]] = 1
            for k,v in __classesDict.items():
                if v > __mx:
                    __mx = v
                    __predictedClass = k
            __predictionList.append(__predictedClass)
        return __predictionList
    
    

def generateData(classNum, objectsNum):
    data = []
    for iClass in range(classNum):
        x, y = random.random()*5.0,random.random()*5.0
        
        for iRow in range(objectsNum):
            data.append([ random.gauss(x,0.5), random.gauss(y,0.5), iClass])
    return data

            

def showData (nClasses, nItemsInClass):
    trainData      = generateData (nClasses, nItemsInClass)
    classColormap  = ListedColormap(['#FF0000', '#00FF00', '#000'])
    pl.scatter([trainData[i][0] for i in range(len(trainData))],
               [trainData[i][1] for i in range(len(trainData))],
               c=[trainData[i][2] for i in range(len(trainData))],
               cmap=classColormap)
    pl.show() 
    
    
def split_train_test(data, testPercent):
    train_data = []
    test_data = []
    length = len(data)
    for _ in range(length):
        if len(train_data)/length*100  < 70:
            row = random.choice(data)
            train_data.append(row)
            data.remove(row)
        else:
            row = random.choice(data)
            test_data.append(row)
            data.remove(row)
            
    return train_data, test_data


showData (3, 90)
data = generateData(3, 90)
train_data,test_data = split_train_test(data, 70)
labels = []
for i in range(len(test_data)):
    labels.append(test_data[i][2])
    test_data[i].pop(2)






            
            
                
            
        
   
clf = Knn()
clf.fit(train_data)

res = clf.predict(test_data)
s = [res[i]-labels[i] for i in range(len(labels))]
print("accuracy: {} \n List of predictions: {} \n List of real classes: {}".format(s.count(0)/len(s)*100, res, labels))

                
    


