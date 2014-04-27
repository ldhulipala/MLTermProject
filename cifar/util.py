import matplotlib.pyplot as plt

def reshape_images(X):
    import load_cifar, main
    if main.GRAYSCALE:
        return X.reshape(-1, load_cifar.width, load_cifar.height)
    else:
        return X.reshape(-1, load_cifar.width, load_cifar.height,
                         load_cifar.channels)

def view_patches(patches, scale=True):
    if len(patches.shape) > 2:
        patches = patches.reshape(patches.shape[0], -1)
    N = patches.shape[0]
    if N > 500: print "Warning: Showing %d patches. This may take a while."%N

    dim = patches.shape[1]
    isGray = (abs((dim**0.5 - round(dim**0.5))) % 1) <= 1e-5
    if isGray:
        size = int(round(dim**0.5))
        patches = patches.reshape(N, size, size)
        patches -= patches.min()
        patches /= patches.max()
    else:
        size = int(round((dim/3)**0.5))
        patches = patches.reshape(N, size, size, 3)
        patches = patches.clip(patches.min(), 1.0)
        #for i in xrange(3):
        #    patches[:, :, :, i] -= (patches[:, :, :, i].min())##.reshape(N, -1).min(axis=0)).mean()
        #    patches[:, :, :, i] /= (patches[:, :, :, i].max())#.reshape(N, -1).max(axis=0)).mean()
        #for i in xrange(N):
        #    for j in xrange(1):
        #        patches[i, :, :, :] -= patches[i, :, :, :].min()
        #        patches[i, :, :, :] /= patches[i, :, :, :].max()
        mn = -1.3
        mx = 1.3
        if scale:
            patches = (patches - mn)/(mx-mn)
    x = y = int(round(N**0.5))
    for i in xrange(y):
        for j in xrange(x):
            if i*x+j >= N:
                break
            ax = plt.subplot2grid((y,x),(i,j))
            ax.imshow(patches[i*x+j], cmap='gray')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

    plt.subplots_adjust(wspace=-.5 ,hspace=0.2)
    plt.show()
