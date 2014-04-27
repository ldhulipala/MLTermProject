from numpy import *
from scipy.sparse import coo_matrix
class SKMeans(object):
    def __init__(self, n_clusters = 8,
                 batch_size=1000,
                 max_iters=30,
                 verbose=False, init='random', n_init=3, max_no_improvement=30):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.verbose = verbose
        self.cluster_centers_ = None

    def fit(self, X, y=None):
        x2 = sum(X**2, axis=1)
        centroids = 0.1*random.randn(self.n_clusters, X.shape[1])
        for itr in xrange(self.max_iters):
            if self.verbose:
                print "K-means iteration %d / %d"%(itr, self.max_iters)

            c2 = 0.5*sum(centroids**2, axis=1)
            summation = zeros((self.n_clusters, X.shape[1]))
            counts = zeros((self.n_clusters, 1))
            loss = 0
            for i in xrange(0, X.shape[0], self.batch_size):
                lastIndex = min(i+self.batch_size, X.shape[0])
                m = lastIndex - i# + 1
                dcx = dot(centroids, X[i:lastIndex, :].T)
                dcx -= c2[:, newaxis]
                labels = argmax(dcx, axis=0)
                val = dcx[labels]
                loss = loss + sum(0.5*x2[i:lastIndex] - val.T)
                #print labels.shape, m
                S = coo_matrix((ones(m), (labels, range(m))), shape=(self.n_clusters, m)).T
                summation += S.T.dot(X[i:lastIndex, :])
                counts += S.sum(axis=0).T
            #print counts
            badIndex = where(counts == 0)[0]
            #print badIndex.shape
            print sum(counts==0)
            #centroids = centroids[where(counts != 0)[0], :]
            #print centroids.shape
            #self.n_clusters = centroids.shape[0]#len(counts != 0)
            centroids = summation/counts
            idxs = random.choice(len(X), len(badIndex), replace=False)
            print idxs
            centroids[badIndex, :] = X[idxs,:]#0.1*random.randn(len(badIndex),X.shape[1])

        self.cluster_centers_ = centroids
        return self

    def transform(self, X):
        return

def test_skmeans():
    A = random.randn(2000, 2) + 15.0
    A = vstack((A, random.randn(1000,2)))
    km = SKMeans(2, verbose=True)
    km.fit(A)
    print km.cluster_centers_
    import matplotlib.pyplot as plt
    plt.scatter(A[:, 0], A[:, 1])
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='r')
    plt.show()

if __name__ == "__main__":
    test_skmeans()
