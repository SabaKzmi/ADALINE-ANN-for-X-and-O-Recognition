import pandas as pd
pd.options.mode.chained_assignment = None
import random as rand

class Adaline:
    def __init__(self):
        self.trainingDF = pd.read_excel('XOdata.xlsx')
        self.weightsDF = pd.read_excel('weights.xlsx')
        self.accuracyDF = pd.read_excel('accuracyCheck.xlsx')

        # set weights
        for r in range(5):
            for c in range(5):
                self.weightsDF.loc[0, 'w{}{}'.format(r, c)] = rand.uniform(0, 1)
        self.weightsDF.loc[0, 'b'] = rand.uniform(0, 1)

    def train(self):
        epoch = 0
        alpha = 0.001
        MSE = 10

        # set weights
        for r in range(5):
            for c in range(5):
                self.weightsDF.loc[0, 'w{}{}'.format(r, c)] = rand.uniform(0, 1)
        self.weightsDF.loc[0, 'b'] = rand.uniform(0, 1)

        while MSE > 0.8 :
            SumError = 0
            randomList = rand.sample(range(len(self.trainingDF)), len(self.trainingDF))
            for i in randomList:     #self.trainingDF.index
                # calculate the y_net input
                yNI = self.weightsDF.loc[0, 'b']
                for r in range(5):
                    for c in range(5):
                        yNI += self.weightsDF.loc[0, 'w{}{}'.format(r, c)] * self.trainingDF.loc[i, 'x{}{}'.format(r, c)]
                # update weights and bias
                delta = (self.trainingDF['t'][i] - yNI)
                SumError += delta ** 2
                for r in range(5):
                    for c in range(5):
                        self.weightsDF.loc[0, 'w{}{}'.format(r, c)] += (alpha * delta * self.trainingDF.loc[i, 'x{}{}'.format(r, c)])
                self.weightsDF.loc[0, 'b'] += delta

            MSE = SumError / len(self.trainingDF)
            epoch += 1
            print("epoch: {}     MSE: {}".format(epoch, MSE))

        #save the new weights
        self.weightsDF.to_excel('weights.xlsx', index=False)

        #accuract check
        print("alpha: {}      MSE > 0.8".format(alpha))
        print('Accuracy: {}'.format(self.checkAccuracy()))

    def activation(self, yNI):
        if yNI >= 0:
            return 1
        elif yNI < 0:
            return -1

    def test(self, test):
        result = ":("
        self.weightsDF = pd.read_excel('weights.xlsx')

        #calculate the Net Input
        NI = self.weightsDF.loc[0, 'b']

        t = 0
        for r in range(5):
            for c in range(5):
                NI += self.weightsDF.loc[0, 'w{}{}'.format(r, c)] * test[t]
                t += 1

        #activation function
        f = self.activation(NI)

        if f == 1:
            return 'X'
        elif f == -1:
            return 'O'
        return result

    def checkAccuracy(self):
        correctAnswers = 0
        accRow = 0
        while accRow < len(self.accuracyDF):
            #test
            res = self.test(self.accuracyDF.loc[accRow, : ].tolist())
            #see if the answer was correct
            if res == 'X' and self.accuracyDF['t'][accRow] == 1:
                correctAnswers += 1
            elif res == 'O' and self.accuracyDF['t'][accRow] == -1:
                correctAnswers += 1

            accRow += 1

        return '{} %'.format((correctAnswers / len(self.accuracyDF))  * 100)