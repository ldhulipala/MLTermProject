import load_cifar, util, feature_learner
from numpy import *

from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction import image

from sklearn.svm import SVC as SVC

N_PATCHES = 100000
N_TRAIN = 2000
N_TEST = 1000
N_CENTROIDS = 140
PATCH_SIZE = 8
STRIDE = 2
GRAYSCALE = False
WHITEN = True

def main():
    dataset = load_cifar.load_cifar(n_train=N_TRAIN, n_test=N_TEST,
                                    grayscale=GRAYSCALE, shuffle=True)

    train_data = dataset['train_data']
    train_labels = dataset['train_labels']
    test_data = dataset['test_data']
    test_labels = dataset['test_labels']

    fl = feature_learner.FeatureLearner(dictionary_size=N_CENTROIDS,
                                        whiten=WHITEN, n_components=0.99, stride=STRIDE, patch_size=PATCH_SIZE, grayscale=GRAYSCALE, normalize=False)

    patch_extractor = image.PatchExtractor(patch_size=(PATCH_SIZE, PATCH_SIZE),
                                           max_patches = N_PATCHES/
                                           len(train_data))
    patches = patch_extractor.transform(util.reshape_images(train_data))
    patches = patches.reshape(len(patches), -1)
    #nm = Normalizer()
    #patches = nm.fit_transform(patches)
    fl.fit(patches)
    ims = fl.transform(train_data)

    im_mean = ims.mean()
    im_std = sqrt(ims.var() + 0.01)

    ims -= im_mean
    ims /= im_std

    nm = Normalizer()
    #ims = nm.fit_transform(ims)

    svm = SVC(C=1000, gamma= 0.01, verbose=True)
    svm.fit(ims, train_labels)

    ls = fl.transform(test_data)
    ls -= im_mean
    ls /= im_std
    #ls = nm.transform(ls)
    print svm.predict(ls)

    print svm.score(ls, test_labels)
    fl.view_dictionary()

    #util.view_patches(train_data[:25])


if __name__ == "__main__":
    main()
