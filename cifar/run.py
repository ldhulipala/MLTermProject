import load_cifar, util, preprocessing, feature_learner
from numpy import *

from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction import image

from sklearn.svm import LinearSVC as SVC

N_PATCHES = 900000
N_TRAIN = 50000
N_TEST = 1000
N_CENTROIDS = 800
PATCH_SIZE = 8
STRIDE = 2
GRAYSCALE = False
WHITEN = True

def main():
    dataset = load_cifar.load_cifar(n_train=N_TRAIN, n_test=N_TEST,
                                    grayscale=GRAYSCALE, shuffle=False)

    train_data = dataset['train_data']
    train_labels = dataset['train_labels']
    test_data = dataset['test_data']
    test_labels = dataset['test_labels']

    print train_data.shape, test_data.shape

    patch_extractor = image.PatchExtractor(patch_size=(PATCH_SIZE, PATCH_SIZE),
                                           max_patches = N_PATCHES/
                                           len(train_data))

    pp = preprocessing.Preprocessor(n_components=0.99)

    fl = feature_learner.FeatureLearner(pp, patch_extractor, n_clusters=N_CENTROIDS)
    fl.fit(train_data)
    train = fl.transform(train_data)
    m_train = mean(train, axis=0)
    train -= m_train
    v_train = sqrt(var(train, axis=0) + 0.01)
    train /= v_train

    test = fl.transform(test_data)
    test -= m_train
    test /= v_train

    classifier = SVC(C=10.0)#, gamma=1e-3, verbose=False)
    classifier.fit(train, train_labels)
    print classifier.score(test, test_labels)

    return

if __name__ == "__main__":
    main()
