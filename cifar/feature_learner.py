from numpy import *
import util, os.path, pickle

from sklearn.feature_extraction import image
from sklearn.cluster import MiniBatchKMeans as KMeans

class FeatureLearner(object):

    def __hash__(self):
        return abs(hash((self.n_clusters, self.patch_size, self.pool_size)))%2**32

    def __init__(self, preprocessor, patch_extractor, n_clusters=100, pool_size=2, patch_size=8, stride = 1):
        self.pre = preprocessor
        self.patch_extractor = patch_extractor
        self.patch_extractor_all = image.PatchExtractor(patch_size=(patch_size,
                                                                    patch_size))
        self.n_clusters = n_clusters
        self.pool_size = pool_size
        self.patch_size = patch_size
        self.stride = stride
        self.km = KMeans(n_clusters = n_clusters,
                         n_init=3, batch_size=120000, max_no_improvement=20, verbose=True)
        return

    def view_features(self):
        util.view_patches(self.pre.inverse_transform(self.km.cluster_centers_))

    def fit(self, images):
        filename = "./memo/%d.npy"%hash(self)
        if not os.path.isfile(filename):
            patches = self.patch_extractor.transform(util.reshape_images(images))
            patches = patches.reshape(len(patches), -1)

            patches = self.pre.fit_transform(patches)

            self.km.fit(patches)
            pickle.dump(self, open(filename, "wb"))
        else:
            old_self = pickle.load(open(filename, "rb"))
            self.pre = old_self.pre
            self.km = old_self.km
        return self

    def transform_im(self, image):
        self.counter += 1
        if self.counter % 10 == 0: print self.counter
        patches = self.patch_extractor_all.transform(array([image]))
        patches = patches[:, ::self.stride, ::self.stride]
        patch_shape = patches[0].shape
        patches = patches.reshape(len(patches), -1)
        patches = self.pre.transform(patches)

        from scipy.spatial.distance import cdist
        all_dists = cdist(patches, self.km.cluster_centers_)
        ave_dists = mean(all_dists, axis=0)
        all_dists = ave_dists - all_dists
        all_dists[all_dists < 0] = 0

        n = int(round(sqrt(all_dists.shape[0])))
        all_dists = all_dists.reshape(n, n, self.n_clusters)
        ds = self.n_clusters
        A = zeros((self.pool_size, self.pool_size, self.n_clusters))
        for i in xrange(self.pool_size):
            for j in xrange(self.pool_size):
                nr = n/self.pool_size
                A[i, j, :] = all_dists[i*nr:(i+1)*nr, j*nr:(j+1)*nr].sum(axis=0).sum(axis=0)
        return A.reshape(-1)

    def transform(self, images, memo=True):
        #self.view_features()
        self.counter = 0
        filename = "./memo/features%d.npy"%((len(images)<<20)+self.n_clusters)
        if memo and os.path.isfile(filename):
            A = load(filename)
        else:
            A = map(self.transform_im, util.reshape_images(images))
            save(filename, A)
        return A
