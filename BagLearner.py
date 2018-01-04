import numpy as np
import RTLearner as rt

class BagLearner(object):

    def __init__(self,learner = rt.RTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False):
        self.learner = learner;
        self.bags = bags;
        self.boost = boost;
        self.verbose = verbose;
        self.learners = [];
        for i in range(0, self.bags):
            self.learners.append(learner(**kwargs));

    def author(self):
        return 'pdesai75'

    def addEvidence(self, Xtrain, Ytrain):
        for i in range(0,self.bags):
            indices = np.random.randint(Xtrain.shape[0], size=Xtrain.shape[0]);
            X = Xtrain[indices[:], :];
            Y = Ytrain[indices[:]];
            self.learners[i].addEvidence(X,Y);

    def query(self, Xtest):

        outputs = []
        for i in range(0,self.bags):
            output = self.learners[i].query(Xtest);
            outputs.append(output);

        return np.mean(outputs,axis=0)

if __name__ == "__main__":
    pass