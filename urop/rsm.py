"""
Replicated Softmax (RSM) Layer
"""

import scipy as sp
import numpy as np

class RSM(object):
    def __init__(self, data, learning_rate=1e-3, weightinit=0.001, momentum=0.9, batchsize=100):
        # initialize weights
        self.w_vh = weightinit * np.random.randn(dictsize, units)
        self.w_v = weightinit * np.random.randn(dictsize)
        self.w_h = weightinit * np.random.randn(dictsize)

        # initialize weight updates
        self.wu_vh = np.zeroes(dictsize, units)
        self.wu_v = np.zeroes((dictsize))
        self.wu_h = np.zeroes((units))

        self.delta = learning_rate / batchsize

        self.batches = data.shape[0] / batchsize

    def train(self, epochs):
        for epoch in xrange(epochs):
            err = []
            for batch in xrange(self.batches):
                start = batch * batchsize
                v1 = data[start : start + batchsize]
                # hidden bias scaling factor
                D = v1.sum(axis=1)
                h1 = sigmoid((np.dot(v1, w_vh) + np.outer(D, w_h)))

                # sample hiddens
                h_rand = np.random.rand(batchsize, units)
                h_sampled = np.array(h_rand < h1, dtype=int)

                # calculate visible activations
                v2 = np.dot(h_sampled, w_vh.T) + w_v
                tmp = np.exp(v2)
                sum = tmp.sum(axis=1)
                sum = sum.reshape((batchsize, 1))
                v2_pdf = tmp / sum

                # sample D times from multinomial
                v2 *= 0
                for i in xrange(batchsize):
                    v2[i] = np.random.multinomial(D[i], v2_pdf[i], size=1)
                # use activations
                h2 = sigmoid(np.dot(v2, w_vh), np.outer(D, w_h))

                # compute updates
                wu_vh = wu_vh * momentum + np.dot(v1.T, h1) - np.dot(v2.T, h2)
                wu_v = wu_v * momentum + v1.sum(axis=0) + v2.sum(axis=0)
                wu_h = wu_h * momentum + h1.sum(axis=0) + h2.sum(axis=0)

                # update
                w_vh += wu_vh * delta
                w_v += wu_v * delta
                w_h += wu_h * delta

                # calculate reconstruction error
                err.append(np.linalg.norm(v2-v1)**2/(dictsize*btsz))
            mean = np.mean(err)
            print("MSE: " + str(mean))
            err_list.append(float(mean))
        return {"w_vh": w_vh,
                "w_v" : w_v,
                "w_h" : w_h,
                "err": err_list}

def sigmoid(x):
    return (1 - sp.tanh(x/2))/2
