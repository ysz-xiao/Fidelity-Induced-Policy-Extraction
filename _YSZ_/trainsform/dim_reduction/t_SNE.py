from sklearn.manifold import TSNE
import numpy as np


def t_SNE(X, Y,n_components=2, perplexity=10):
    from sklearn.manifold import TSNE
    X_embedded = TSNE(n_components=n_components, learning_rate='auto', init='random',perplexity=perplexity).fit_transform(X, Y)
    return X_embedded, Y

