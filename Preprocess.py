from sklearn.decomposition import PCA
import GlobalVars as gv
import numpy as np

scalers = gv.scalers

PCA_total_steps = gv.PCA_total_steps
PCA_min = gv.PCA_min


def scale(data):
    for sclr in scalers[:-1]:
        scaler = sclr
        scaler.fit(data)
        yield scaler.transform(data), sclr
    if scalers[-1]:
        yield data, "Raw Data"


def pri_comp_an(data):
    for n_comp in sorted(list(set([int(x) for x in np.linspace(PCA_min, data.shape[1], num=PCA_total_steps)])), reverse=True):
        if n_comp == data.shape[1]:
            yield data, data.shape[1]
        else:
            pca = PCA(n_components=n_comp)
            pca.fit(data)
            yield pca.transform(data), n_comp

