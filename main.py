import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import math
import operator

class KNN():
    def __init__(self):
        

        dataset = {"x1":[],"x2":[],"x3":[],"x4":[],"x5":[]}
        with open('iris.data.txt', 'r') as csvfile:

            lines = csv.reader(csvfile)
            
            for i in lines:
                for z in range(1,6):
                    dataset[f"x{z}"].append(i[z-1])

        self.df = pd.DataFrame(dataset)
        lb = LabelEncoder()
        self.df["x5"] = lb.fit_transform(self.df['x5'])
        self.df = self.df.astype(float)
    

    def euclideanDistance(data1,data2,length):
        d = 0
        for i in range(0,length):
            d += data1[i] + data2[i]
        d = math.sqrt(d)
    
        return d
    def getNeighbors(trainingSet, testInstance, k):

        distances = []

        length = len(testInstance)-1

        for x in range(len(trainingSet)):

            dist = KNN.euclideanDistance(testInstance, trainingSet[x], length)

            distances.append((trainingSet[x], dist))

            distances.sort(key=operator.itemgetter(1))

            neighbors = []

        for x in range(k):

            neighbors.append(distances[x][0])

        return neighbors
    def getResponse(neighbors):

        classVotes = {}

        for x in range(len(neighbors)):

            response = neighbors[x][ -1] #complete with appropriate number

        if response in classVotes:
            classVotes[response] = classVotes[response]+1
        else:
            classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

        return sortedVotes[0][0]
    def getAccuracy(testSet,predictions):
        c = 0
        for i in range(len(testSet)):
            if testSet[i][-1] == predictions[i]:
                c += 1
        return c / len(testSet) * 100

    def manhattan(a, b):
        return sum(abs(val1-val2) for val1, val2 in zip(a,b))

