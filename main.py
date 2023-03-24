import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time

from sklearn import preprocessing
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score, v_measure_score, completeness_score, homogeneity_score, mean_squared_error
from sklearn.decomposition import PCA, FastICA
from sklearn import random_projection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier

from scipy.stats import kurtosis

import warnings

# I HAVE HAD IT WITH THESE WARNING GOOD BYE
warnings.simplefilter("ignore")

# I AGGRESSIVELY COPY AND PASTE CODE FROM THE SCIKITLEARN DOCS

# The wine quality dataset
wine = pd.read_csv('data/wine.csv')
wine = wine.drop('Id', axis=1)

wine_y = wine[["quality"]].copy()
wine_x = wine.drop("quality", axis=1)
wine_scaler = preprocessing.StandardScaler().fit(wine_x)
wine_x = wine_scaler.transform(wine_x)

# We're only going to be using the test set for the later sections, but pull it out now.
wine_x_train, wine_x_test, wine_y_train, wine_y_test = train_test_split(wine_x,wine_y,test_size=.10,random_state=0)

# The pumpkin seeds dataset
pumpkin = pd.read_csv('data/pumpkin.csv')

pumpkin_y = pumpkin[["Class"]].copy()
pumpkin_x = pumpkin.drop("Class", axis=1)
pumpkin_scaler = preprocessing.StandardScaler().fit(pumpkin_x)
pumpkin_x = pumpkin_scaler.transform(pumpkin_x)

# We're only going to be using the test set for the later sections, but pull it out now.
pumpkin_x_train, pumpkin_x_test, pumpkin_y_train, pumpkin_y_test = train_test_split(pumpkin_x,pumpkin_y,test_size=.10,random_state=0)


def section_1():
    pumpkin_knn_silhouettes = []
    pumpkin_em_silhouettes = []

    pumpkin_knn_times = []
    pumpkin_em_times = []

    indices = range(2, 20)
    for i in indices:
        start = time.time()
        cluster = KMeans(n_clusters=i, n_init="auto")
        pred = cluster.fit_predict(pumpkin_x_train)
        pumpkin_knn_times.append(time.time()-start)
        silhouette = silhouette_score(pumpkin_x_train, pred, metric="cosine")
        pumpkin_knn_silhouettes.append(silhouette)
        print(f"KNN for Pumpkin with {i} clusters, silhouette score is {silhouette}")

        start = time.time()
        cluster = GaussianMixture(n_components=i)
        pred = cluster.fit_predict(pumpkin_x_train)
        pumpkin_em_times.append(time.time()-start)
        silhouette = silhouette_score(pumpkin_x_train, pred, metric="cosine")
        pumpkin_em_silhouettes.append(silhouette)
        print(f"EM for Pumpkin with {i} clusters, silhouette score is {silhouette}")

    plt.plot(indices, pumpkin_knn_silhouettes, label="KNN")
    plt.plot(indices, pumpkin_em_silhouettes, label="EM")
    plt.xticks(indices)
    plt.legend(loc="upper right")
    plt.title("Pumpkin Seeds Silhouette scores versus cluster size")
    plt.xlabel("Cluster size")
    plt.ylabel("Scores")
    plt.savefig("plots/pumpkin_clusters.png")
    plt.close()

    plt.plot(indices, pumpkin_knn_times, label="KNN")
    plt.plot(indices, pumpkin_em_times, label="EM")
    plt.xticks(indices)
    plt.xlabel("Cluster size")
    plt.ylabel("Time")
    plt.title("Pumpkin Seeds Time to cluster versus cluster size")
    plt.legend(loc="upper right")
    plt.savefig("plots/pumpkin_cluster_time.png")
    plt.close()

    print("")

    wine_knn_silhouettes = []
    wine_em_silhouettes = []

    wine_knn_times = []
    wine_em_times = []
    for i in indices:
        start = time.time()
        cluster = KMeans(n_clusters=i, n_init="auto")
        pred = cluster.fit_predict(wine_x_train)
        wine_knn_times.append(time.time()-start)
        silhouette = silhouette_score(wine_x_train, pred, metric="cosine")
        wine_knn_silhouettes.append(silhouette)
        print(f"KNN for Wine with {i} clusters, silhouette score is {silhouette}")

        start = time.time()
        cluster = GaussianMixture(n_components=i)
        pred = cluster.fit_predict(wine_x_train)
        wine_em_times.append(time.time()-start)
        silhouette = silhouette_score(wine_x_train, pred, metric="cosine")
        wine_em_silhouettes.append(silhouette)
        print(f"EM for Wine with {i} clusters, silhouette score is {silhouette}")

    plt.plot(indices, wine_knn_silhouettes, label="KNN")
    plt.plot(indices, wine_em_silhouettes, label="EM")
    plt.xticks(indices)
    plt.title("Wine Quality Silhouette scores versus cluster size")
    plt.legend(loc="upper right")
    plt.xlabel("Cluster size")
    plt.ylabel("Scores")
    plt.savefig("plots/wine_clusters.png")
    plt.close()

    plt.plot(indices, wine_knn_times, label="KNN")
    plt.plot(indices, wine_em_times, label="EM")
    plt.xticks(indices)
    plt.title("Wine Quality Time to cluster versus cluster size")
    plt.legend(loc="upper right")
    plt.xlabel("Cluster size")
    plt.ylabel("Time")
    plt.savefig("plots/wine_cluster_time.png")
    plt.close()

    print("")

    print("GRAPHING KNN PUMPKIN SEED RESULTS")
    k = 2
    cluster = KMeans(n_clusters=k, n_init="auto")
    pred = cluster.fit_predict(pumpkin_x_train)
    
    class_x = []
    class_y = []
    for i in range(k):
        class_x.append([])
        class_y.append([])

    for i in range(len(pred)):
        class_x[pred[i]].append(pumpkin_x_train[i][2])
        class_y[pred[i]].append(pumpkin_x_train[i][11])

    for i in range(k):
        plt.scatter(class_x[i], class_y[i])
    plt.title("KNN Pumpkin Seeds with 2 clusters")
    plt.savefig("plots/knn_2_seeds_pred")
    plt.close()

    k = 4
    cluster = KMeans(n_clusters=k, n_init="auto")
    pred = cluster.fit_predict(pumpkin_x_train)
    
    class_x = []
    class_y = []
    for i in range(k):
        class_x.append([])
        class_y.append([])

    for i in range(len(pred)):
        class_x[pred[i]].append(pumpkin_x_train[i][2])
        class_y[pred[i]].append(pumpkin_x_train[i][11])

    for i in range(k):
        plt.scatter(class_x[i], class_y[i])
    plt.title("KNN Pumpkin Seeds with 4 clusters")
    plt.savefig("plots/knn_4_seeds_pred")
    plt.close()

    k = 2
    pred = np.ravel(pumpkin_y_train)
    
    class_x = []
    class_y = []
    for i in range(k):
        class_x.append([])
        class_y.append([])

    for i in range(len(pred)):
        class_x[pred[i]].append(pumpkin_x_train[i][2])
        class_y[pred[i]].append(pumpkin_x_train[i][11])

    for i in range(k):
        plt.scatter(class_x[i], class_y[i])
    plt.title("Actual Pumpkin Seed clusters")
    plt.savefig("plots/seeds_truth")
    plt.close()

    
    print("GRAPHING EM PUMPKIN SEED RESULTS")

    k = 2
    cluster = GaussianMixture(n_components=k)
    pred = cluster.fit_predict(pumpkin_x_train)
    
    class_x = []
    class_y = []
    for i in range(k):
        class_x.append([])
        class_y.append([])

    for i in range(len(pred)):
        class_x[pred[i]].append(pumpkin_x_train[i][2])
        class_y[pred[i]].append(pumpkin_x_train[i][11])

    for i in range(k):
        plt.scatter(class_x[i], class_y[i])
    plt.title("EM Pumpkin Seeds with 2 clusters")
    plt.savefig("plots/em_2_seeds_pred")
    plt.close()

    k = 4
    cluster = GaussianMixture(n_components=k)
    pred = cluster.fit_predict(pumpkin_x_train)
    
    class_x = []
    class_y = []
    for i in range(k):
        class_x.append([])
        class_y.append([])

    for i in range(len(pred)):
        class_x[pred[i]].append(pumpkin_x_train[i][2])
        class_y[pred[i]].append(pumpkin_x_train[i][11])

    for i in range(k):
        plt.scatter(class_x[i], class_y[i])
    plt.title("EM Pumpkin Seeds with 4 clusters")
    plt.savefig("plots/em_4_seeds_pred")
    plt.close()


    print("GRAPHING KNN WINE RESULTS")
    k = 2
    cluster = KMeans(n_clusters=k, n_init="auto")
    pred = cluster.fit_predict(wine_x_train)
    
    class_x = []
    class_y = []
    for i in range(k):
        class_x.append([])
        class_y.append([])

    for i in range(len(pred)):
        class_x[pred[i]].append(wine_x_train[i][0])
        class_y[pred[i]].append(wine_x_train[i][10])

    for i in range(k):
        plt.scatter(class_x[i], class_y[i])
    plt.title("KNN Wine with 2 clusters")
    plt.savefig("plots/knn_2_wine_pred")
    plt.close()

    k = 6
    cluster = KMeans(n_clusters=k, n_init="auto")
    pred = cluster.fit_predict(wine_x_train)
    
    class_x = []
    class_y = []
    for i in range(k):
        class_x.append([])
        class_y.append([])

    for i in range(len(pred)):
        class_x[pred[i]].append(wine_x_train[i][0])
        class_y[pred[i]].append(wine_x_train[i][10])

    for i in range(k):
        plt.scatter(class_x[i], class_y[i])
    plt.title("KNN Wine with 6 clusters")
    plt.savefig("plots/knn_6_wine_pred")
    plt.close()

    k = 10
    pred = np.ravel(wine_y_train)
    
    class_x = []
    class_y = []
    for i in range(k):
        class_x.append([])
        class_y.append([])

    for i in range(len(pred)):
        class_x[pred[i]].append(wine_x_train[i][0])
        class_y[pred[i]].append(wine_x_train[i][10])

    for i in range(k):
        plt.scatter(class_x[i], class_y[i])
    plt.title("Actual Wine Quality clusters")
    plt.savefig("plots/wine_truth")
    plt.close()

    
    print("GRAPHING EM WINE RESULTS")

    k = 2
    cluster = GaussianMixture(n_components=k)
    pred = cluster.fit_predict(wine_x_train)
    
    class_x = []
    class_y = []
    for i in range(k):
        class_x.append([])
        class_y.append([])

    for i in range(len(pred)):
        class_x[pred[i]].append(wine_x_train[i][0])
        class_y[pred[i]].append(wine_x_train[i][10])

    for i in range(k):
        plt.scatter(class_x[i], class_y[i])
    plt.title("EM Wine with 2 clusters")
    plt.savefig("plots/em_2_wine_pred")
    plt.close()

    k = 6
    cluster = GaussianMixture(n_components=k)
    pred = cluster.fit_predict(wine_x_train)
    
    class_x = []
    class_y = []
    for i in range(k):
        class_x.append([])
        class_y.append([])

    for i in range(len(pred)):
        class_x[pred[i]].append(wine_x_train[i][0])
        class_y[pred[i]].append(wine_x_train[i][10])

    for i in range(k):
        plt.scatter(class_x[i], class_y[i])
    plt.title("EM Pumpkin Seeds with 6 clusters")
    plt.savefig("plots/em_6_wine_pred")
    plt.close()

    print("")

    print("KNN PUMPKIN RESULTS")
    cluster = KMeans(n_clusters=2, n_init="auto")
    pred = cluster.fit_predict(pumpkin_x_train)
    print(f"Silhouette Score: {silhouette_score(pumpkin_x_train, pred)}")
    print(f"Homogeneity Score: {homogeneity_score(np.ravel(pumpkin_y_train), pred)}")
    print(f"Completeness Score: {completeness_score(np.ravel(pumpkin_y_train), pred)}")
    print(f"V Measure: {v_measure_score(np.ravel(pumpkin_y_train), pred)}")

    print("")

    print("EM PUMPKIN RESULTS")
    cluster = GaussianMixture(n_components=2)
    pred = cluster.fit_predict(pumpkin_x_train)
    print(f"Silhouette Score: {silhouette_score(pumpkin_x_train, pred)}")
    print(f"Homogeneity Score: {homogeneity_score(np.ravel(pumpkin_y_train), pred)}")
    print(f"Completeness Score: {completeness_score(np.ravel(pumpkin_y_train), pred)}")
    print(f"V Measure: {v_measure_score(np.ravel(pumpkin_y_train), pred)}")

    print("")

    print("KNN WINE RESULTS")
    cluster = KMeans(n_clusters=3, n_init="auto")
    pred = cluster.fit_predict(wine_x_train)
    print(f"Silhouette Score: {silhouette_score(wine_x_train, pred)}")
    print(f"Homogeneity Score: {homogeneity_score(np.ravel(wine_y_train), pred)}")
    print(f"Completeness Score: {completeness_score(np.ravel(wine_y_train), pred)}")
    print(f"V Measure: {v_measure_score(np.ravel(wine_y_train), pred)}")

    print("")

    print("EM WINE RESULTS")
    cluster = GaussianMixture(n_components=5)
    pred = cluster.fit_predict(wine_x_train)
    print(f"Silhouette Score: {silhouette_score(wine_x_train, pred)}")
    print(f"Homogeneity Score: {homogeneity_score(np.ravel(wine_y_train), pred)}")
    print(f"Completeness Score: {completeness_score(np.ravel(wine_y_train), pred)}")
    print(f"V Measure: {v_measure_score(np.ravel(wine_y_train), pred)}")

    print("")

def section_2():
    # PCA

    # Pumpkin

    # https://vitalflux.com/pca-explained-variance-concept-python-example/
    # https://medium.com/luca-chuangs-bapm-notes/principal-component-analysis-pca-using-python-scikit-learn-48c4c13e49af

    pca = PCA()
    pca.fit(pumpkin_x_train)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_))
    plt.title("Cumulative explained variance ratio per principal component for Pumpkin Seeds", wrap=True)
    plt.ylabel("Cumulative explained variance ratio")
    plt.xlabel("Number of principal components")
    plt.savefig("plots/pumpkin_pca.png")
    plt.close()

    # Wine
    pca = PCA()
    pca.fit(wine_x_train)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_))
    plt.title("Cumulative explained variance ratio per principal component for Wine Quality", wrap=True)
    plt.ylabel("Cumulative explained variance ratio")
    plt.xlabel("Number of principal components")
    plt.savefig("plots/wine_pca.png")
    plt.close()

    # ICA https://edstem.org/us/courses/32923/discussion/2767458

    # Pumpkin seeds
    pumpkin_ica = []
    indices = range(1, len(pumpkin_x_train[0] + 1))
    for i in indices:
        ica = FastICA(i, whiten='unit-variance') # supress warning
        out = ica.fit_transform(pumpkin_x_train)
        counter = 0
        kur = 0
        for col in out.T:
            kur += kurtosis(col)
            counter += 1
        pumpkin_ica.append(kur/counter)
    
    plt.plot(indices, pumpkin_ica)
    plt.title("Average Kurtosis per number of components for Pumpkin Seeds", wrap=True)
    plt.ylabel("Average Kurtosis")
    plt.xlabel("Number of components")
    plt.savefig("plots/pumpkin_ica.png")
    plt.close()

    # Wine Quality
    wine_ica = []
    indices = range(1, len(wine_x_train[0] + 1))
    for i in indices:
        ica = FastICA(i, whiten='unit-variance') # supress warning
        out = ica.fit_transform(wine_x_train)
        counter = 0
        kur = 0
        for col in out.T:
            kur += kurtosis(col)
            counter += 1
        wine_ica.append(kur/counter)
    
    plt.plot(indices, wine_ica)
    plt.title("Average Kurtosis per number of components for Wine Quality", wrap=True)
    plt.ylabel("Average Kurtosis")
    plt.xlabel("Number of components")
    plt.savefig("plots/wine_ica.png")
    plt.close()

    # RANDOMIZED PROJECTIONS https://edstem.org/us/courses/32923/discussion/2794176

    # Pumpkin seeds
    pumpkin_rp = []
    indices = range(1, len(pumpkin_x_train[0] + 1))
    for i in indices:
        rp = random_projection.GaussianRandomProjection(n_components = i, compute_inverse_components=True)
        out = rp.fit_transform(pumpkin_x_train)
        back = rp.inverse_transform(out)
        pumpkin_rp.append(np.sum(mean_squared_error(pumpkin_x_train, back)))
    
    plt.plot(indices, pumpkin_rp)
    plt.title("Reconstruction error by number of components for Pumpkin Seeds", wrap=True)
    plt.ylabel("mean squarred error")
    plt.xlabel("Number of components")
    plt.savefig("plots/pumpkin_rp.png")
    plt.close()

    # Wine Quality
    wine_rp = []
    indices = range(1, len(wine_x_train[0] + 1))
    for i in indices:
        rp = random_projection.GaussianRandomProjection(n_components = i, compute_inverse_components=True)
        out = rp.fit_transform(wine_x_train)
        back = rp.inverse_transform(out)
        wine_rp.append(np.sum(mean_squared_error(wine_x_train, back)))
    
    plt.plot(indices, wine_rp)
    plt.title("Reconstruction error by number of components for Wine Quality", wrap=True)
    plt.ylabel("mean squarred error")
    plt.xlabel("Number of components")
    plt.savefig("plots/wine_rp.png")
    plt.close()

    # TREE FEATURE SELECTION

    # PUMPKIN SEEDS
    clf = ExtraTreesClassifier(n_estimators=50)
    clf.fit(pumpkin_x_train, pumpkin_y_train)
    model = SelectFromModel(clf, prefit=True)
    trimmed = model.transform(pumpkin_x_train)

    plt.bar(range(0,len(clf.feature_importances_)), clf.feature_importances_)
    plt.title("Importance of features according to decision trees of Pumpkin Seeds", wrap=True)
    plt.ylabel("Importance")
    plt.xlabel("Feature Number")
    plt.savefig("plots/pumpkin_tfs.png")
    plt.close()

    # WINE
    clf = ExtraTreesClassifier(n_estimators=50)
    clf.fit(wine_x_train, wine_y_train)
    model = SelectFromModel(clf, prefit=True)
    trimmed = model.transform(wine_x_train)

    plt.bar(range(0,len(clf.feature_importances_)), clf.feature_importances_)
    plt.title("Importance of features according to decision trees of Wine Quality", wrap=True)
    plt.ylabel("Importance")
    plt.xlabel("Feature Number")
    plt.savefig("plots/wine_tfs.png")
    plt.close()

def plotting_helper(indices, data):
    knn_silhouettes = []
    em_silhouettes = []
    for i in indices:
        cluster = KMeans(n_clusters=i, n_init="auto")
        pred = cluster.fit_predict(data)
        silhouette = silhouette_score(data, pred, metric="cosine")
        knn_silhouettes.append(silhouette)

        cluster = GaussianMixture(n_components=i)
        pred = cluster.fit_predict(data)
        silhouette = silhouette_score(data, pred, metric="cosine")
        em_silhouettes.append(silhouette)
    
    return (knn_silhouettes, em_silhouettes)

def section_3_part_1():

    # PUMPKIN SEEDS

    indices = range(2, 20)
    start = time.time()
    pca = PCA(n_components = 4)
    data = pca.fit_transform(pumpkin_x_train)
    print(f"Pumpkin PCA fit time: {time.time()-start}")
    (knn_silhouettes, em_silhouettes) = plotting_helper(indices, data)
    plt.suptitle("Pumpkin Seed Silhouette scores given dimensionality reduction algorithms", wrap=True)

    plt.subplot(2,2,1)
    plt.plot(indices, knn_silhouettes, label="KNN")
    plt.plot(indices, em_silhouettes, label="EM")
    plt.xticks([])
    plt.title("PCA")
    plt.legend(loc="upper right")

    indices = range(2, 20)
    start = time.time()
    ica = FastICA(n_components = 9)
    data = ica.fit_transform(pumpkin_x_train)
    print(f"Pumpkin ICA fit time: {time.time()-start}")
    (knn_silhouettes, em_silhouettes) = plotting_helper(indices, data)

    plt.subplot(2,2,2)
    plt.plot(indices, knn_silhouettes, label="KNN")
    plt.plot(indices, em_silhouettes, label="EM")
    plt.xticks([])
    plt.title("ICA")
    plt.legend(loc="upper right")

    start = time.time()
    rp = random_projection.GaussianRandomProjection(n_components = 10, compute_inverse_components=True)
    data = rp.fit_transform(pumpkin_x_train)
    print(f"Pumpkin RP fit time: {time.time()-start}")
    (knn_silhouettes, em_silhouettes) = plotting_helper(indices, data)

    plt.subplot(2,2,3)
    plt.plot(indices, knn_silhouettes, label="KNN")
    plt.plot(indices, em_silhouettes, label="EM")
    plt.title("Random Projection")
    plt.legend(loc="upper right")

    indices = range(2, 20)
    start = time.time()
    clf = ExtraTreesClassifier(n_estimators=50)
    clf.fit(pumpkin_x_train, pumpkin_y_train)
    model = SelectFromModel(clf, prefit=True)
    data = model.transform(pumpkin_x_train)
    print(f"Pumpkin DT fit time: {time.time()-start}")
    (knn_silhouettes, em_silhouettes) = plotting_helper(indices, data)

    plt.subplot(2,2,4)
    plt.plot(indices, knn_silhouettes, label="KNN")
    plt.plot(indices, em_silhouettes, label="EM")
    plt.title("Decision Tree Features")
    plt.legend(loc="upper right")

    plt.savefig("plots/seeds_dim_reduction_clusters.png")
    plt.close()

    # WINE TIME

    indices = range(2, 20)
    start = time.time()
    pca = PCA(n_components = 4)
    data = pca.fit_transform(wine_x_train)
    print(f"Wine PCA fit time: {time.time()-start}")
    (knn_silhouettes, em_silhouettes) = plotting_helper(indices, data)
    plt.suptitle("Wine Quality Silhouette scores given dimensionality reduction algorithms", wrap=True)

    plt.subplot(2,2,1)
    plt.plot(indices, knn_silhouettes, label="KNN")
    plt.plot(indices, em_silhouettes, label="EM")
    plt.xticks([])
    plt.title("PCA")
    plt.legend(loc="upper right")

    indices = range(2, 20)
    start = time.time()
    ica = FastICA(n_components = 9)
    data = ica.fit_transform(wine_x_train)
    print(f"Wine ICA fit time: {time.time()-start}")
    (knn_silhouettes, em_silhouettes) = plotting_helper(indices, data)

    plt.subplot(2,2,2)
    plt.plot(indices, knn_silhouettes, label="KNN")
    plt.plot(indices, em_silhouettes, label="EM")
    plt.xticks([])
    plt.title("ICA")
    plt.legend(loc="upper right")

    start = time.time()
    rp = random_projection.GaussianRandomProjection(n_components = 10, compute_inverse_components=True)
    data = rp.fit_transform(wine_x_train)
    print(f"Wine RP fit time: {time.time()-start}")
    (knn_silhouettes, em_silhouettes) = plotting_helper(indices, data)

    plt.subplot(2,2,3)
    plt.plot(indices, knn_silhouettes, label="KNN")
    plt.plot(indices, em_silhouettes, label="EM")
    plt.title("Random Projection")
    plt.legend(loc="upper right")

    # TAKEN STRAIGHT FROM SCIKITLEARN
    # https://scikit-learn.org/stable/modules/feature_selection.html#tree-based-feature-selection
    indices = range(2, 20)
    start = time.time()
    clf = ExtraTreesClassifier(n_estimators=50)
    clf.fit(wine_x_train, wine_y_train)
    model = SelectFromModel(clf, prefit=True)
    data = model.transform(wine_x_train)
    print(f"Wine DT fit time: {time.time()-start}")
    (knn_silhouettes, em_silhouettes) = plotting_helper(indices, data)

    plt.subplot(2,2,4)
    plt.plot(indices, knn_silhouettes, label="KNN")
    plt.plot(indices, em_silhouettes, label="EM")
    plt.title("Decision Tree Features")
    plt.legend(loc="upper right")

    plt.savefig("plots/wine_dim_reduction_clusters.png")
    plt.close()

def section_3_part_2():
    clf = ExtraTreesClassifier(n_estimators=50)
    clf.fit(pumpkin_x_train, pumpkin_y_train)
    model = SelectFromModel(clf, prefit=True)
    data = model.transform(pumpkin_x_train)

    k = 2
    cluster = KMeans(n_clusters=k, n_init="auto")
    pred = cluster.fit_predict(data)
    
    class_x = []
    class_y = []
    for i in range(k):
        class_x.append([])
        class_y.append([])

    for i in range(len(pred)):
        class_x[pred[i]].append(pumpkin_x_train[i][2])
        class_y[pred[i]].append(pumpkin_x_train[i][11])

    for i in range(k):
        plt.scatter(class_x[i], class_y[i])
    plt.title("KNN Pumpkin Seeds with 2 clusters after Decision Tree Dimensionality Reduction", wrap=True)
    plt.savefig("plots/tfs_knn_2_seeds_pred")
    plt.close()

    
    print("")

    print("KNN PUMPKIN RESULTS")
    print(f"Silhouette Score: {silhouette_score(pumpkin_x_train, pred)}")
    print(f"Homogeneity Score: {homogeneity_score(np.ravel(pumpkin_y_train), pred)}")
    print(f"Completeness Score: {completeness_score(np.ravel(pumpkin_y_train), pred)}")
    print(f"V Measure: {v_measure_score(np.ravel(pumpkin_y_train), pred)}")

    print("")

    clf = ExtraTreesClassifier(n_estimators=50)
    clf.fit(pumpkin_x_train, pumpkin_y_train)
    model = SelectFromModel(clf, prefit=True)
    data = model.transform(pumpkin_x_train)

    k = 2
    cluster = GaussianMixture(n_components=k)
    pred = cluster.fit_predict(data)
    
    class_x = []
    class_y = []
    for i in range(k):
        class_x.append([])
        class_y.append([])

    for i in range(len(pred)):
        class_x[pred[i]].append(pumpkin_x_train[i][2])
        class_y[pred[i]].append(pumpkin_x_train[i][11])

    for i in range(k):
        plt.scatter(class_x[i], class_y[i])
    plt.title("EM Pumpkin Seeds with 2 clusters after Decision Tree Dimensionality Reduction", wrap=True)
    plt.savefig("plots/tfs_em_2_seeds_pred")
    plt.close()

    print("")

    print("EM PUMPKIN RESULTS")
    print(f"Silhouette Score: {silhouette_score(pumpkin_x_train, pred)}")
    print(f"Homogeneity Score: {homogeneity_score(np.ravel(pumpkin_y_train), pred)}")
    print(f"Completeness Score: {completeness_score(np.ravel(pumpkin_y_train), pred)}")
    print(f"V Measure: {v_measure_score(np.ravel(pumpkin_y_train), pred)}")

    print("")

def section_3_part_3():
    pca = PCA(n_components = 4)
    data = pca.fit_transform(wine_x_train)

    k = 4
    cluster = KMeans(n_clusters=k, n_init="auto")
    pred = cluster.fit_predict(data)
    
    class_x = []
    class_y = []
    for i in range(k):
        class_x.append([])
        class_y.append([])

    for i in range(len(pred)):
        class_x[pred[i]].append(wine_x_train[i][0])
        class_y[pred[i]].append(wine_x_train[i][10])

    for i in range(k):
        plt.scatter(class_x[i], class_y[i])
    plt.title("KNN Wine Quality with 4 clusters after PCA")
    plt.savefig("plots/pca_knn_2_wine_pred")
    plt.close()

    
    print("")

    print("KNN PUMPKIN RESULTS")
    print(f"Silhouette Score: {silhouette_score(data, pred)}")
    print(f"Homogeneity Score: {homogeneity_score(np.ravel(wine_y_train), pred)}")
    print(f"Completeness Score: {completeness_score(np.ravel(wine_y_train), pred)}")
    print(f"V Measure: {v_measure_score(np.ravel(wine_y_train), pred)}")

    print("")

    pca = PCA(n_components = 4)
    data = pca.fit_transform(wine_x_train)

    k = 4
    cluster = GaussianMixture(n_components=k)
    pred = cluster.fit_predict(data)
    
    class_x = []
    class_y = []
    for i in range(k):
        class_x.append([])
        class_y.append([])

    for i in range(len(pred)):
        class_x[pred[i]].append(wine_x_train[i][0])
        class_y[pred[i]].append(wine_x_train[i][10])

    for i in range(k):
        plt.scatter(class_x[i], class_y[i])
    plt.title("EM Wine Quality with 4 clusters after PCA")
    plt.savefig("plots/pca_em_2_wine_pred")
    plt.close()

    print("")

    print("EM PUMPKIN RESULTS")
    print(f"Silhouette Score: {silhouette_score(data, pred)}")
    print(f"Homogeneity Score: {homogeneity_score(np.ravel(wine_y_train), pred)}")
    print(f"Completeness Score: {completeness_score(np.ravel(wine_y_train), pred)}")
    print(f"V Measure: {v_measure_score(np.ravel(wine_y_train), pred)}")

    print("")

def print_report(data_set, clf, predicted, actual):
    print(
        f"Classification report for classifier {clf} on {data_set}:\n"
        f"{metrics.classification_report(actual, predicted)}\n"
    )

    disp = metrics.ConfusionMatrixDisplay.from_predictions(actual, predicted)
    disp.figure_.suptitle(f"Confusion Matrix for {clf} on {data_set}")

    plt.savefig(f"plots/Confusion Matrix for {clf} on {data_set}.png")
    plt.close()

def section_4():
    start = time.time()
    clf = MLPClassifier(hidden_layer_sizes = (128), max_iter=100, random_state=0)
    clf.fit(pumpkin_x_train, pumpkin_y_train)
    regularloss = clf.loss_curve_
    pred = clf.predict(pumpkin_x_test)
    print(f"Pumpkin Regular NN train time: {time.time()-start}")
    print_report("pumpkin seeds", "Baseline NN", pred, pumpkin_y_test)

    pca = PCA(n_components = 4)
    x_train = pca.fit_transform(pumpkin_x_train)
    x_test = pca.transform(pumpkin_x_test)
    start = time.time()
    clf = MLPClassifier(hidden_layer_sizes = (128), max_iter=100, random_state=0)
    clf.fit(x_train, pumpkin_y_train)
    pcaloss = clf.loss_curve_
    pred = clf.predict(x_test)
    print(f"Pumpkin PCA NN train time: {time.time()-start}")
    print_report("pumpkin seeds", "PCA NN", pred, pumpkin_y_test)

    ica = FastICA(n_components = 9)
    x_train = ica.fit_transform(pumpkin_x_train)
    x_test = ica.transform(pumpkin_x_test)
    start = time.time()
    clf = MLPClassifier(hidden_layer_sizes = (128), max_iter=100, random_state=0)
    clf.fit(x_train, pumpkin_y_train)
    icaloss = clf.loss_curve_
    pred = clf.predict(x_test)
    print(f"Pumpkin ICA NN train time: {time.time()-start}")
    print_report("pumpkin seeds", "ICA NN", pred, pumpkin_y_test)
    

    rp = random_projection.GaussianRandomProjection(n_components = 10, compute_inverse_components=True)
    x_train = rp.fit_transform(pumpkin_x_train)
    x_test = rp.transform(pumpkin_x_test)
    start = time.time()
    clf = MLPClassifier(hidden_layer_sizes = (128), max_iter=100, random_state=0)
    clf.fit(x_train, pumpkin_y_train)
    rploss = clf.loss_curve_
    pred = clf.predict(x_test)
    print(f"Pumpkin RP NN train time: {time.time()-start}")
    print_report("pumpkin seeds", "RP NN", pred, pumpkin_y_test)
    

    clf = ExtraTreesClassifier(n_estimators=50)
    clf.fit(pumpkin_x_train, pumpkin_y_train)
    model = SelectFromModel(clf, prefit=True)
    x_train = model.transform(pumpkin_x_train)
    x_test = model.transform(pumpkin_x_test)
    start = time.time()
    clf = MLPClassifier(hidden_layer_sizes = (128), max_iter=100, random_state=0)
    clf.fit(x_train, pumpkin_y_train)
    dtloss = clf.loss_curve_
    pred = clf.predict(x_test)
    print(f"Pumpkin DT NN train time: {time.time()-start}")
    print_report("pumpkin seeds", "DT NN", pred, pumpkin_y_test)

    plt.plot(regularloss, label="Regular NN")
    plt.plot(pcaloss, label="PCA NN")
    plt.plot(icaloss, label="ICA NN")
    plt.plot(rploss, label="RP NN")
    plt.plot(dtloss, label="DT NN")
    plt.legend(loc="upper right")
    plt.title("Loss curves for different dimensionality reduction algorithms")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.savefig("plots/loss_curves_dr.png")
    plt.close()

def section_5_part_1():
    clf = MLPClassifier(hidden_layer_sizes = (128), max_iter=100, random_state=0)
    clf.fit(pumpkin_x_train, pumpkin_y_train)
    regularloss = clf.loss_curve_
    pred = clf.predict(pumpkin_x_test)

    pca = PCA(n_components = 4)
    x_train = pca.fit_transform(pumpkin_x_train)
    x_test = pca.transform(pumpkin_x_test)
    cluster = GaussianMixture(n_components=2)
    additional_x_train = cluster.fit_predict(x_train)
    additional_x_test = cluster.predict(x_test)
    x_train = np.column_stack((pumpkin_x_train, additional_x_train))
    x_test = np.column_stack((pumpkin_x_test, additional_x_test))
    clf = MLPClassifier(hidden_layer_sizes = (128), max_iter=100, random_state=0)
    clf.fit(x_train, pumpkin_y_train)
    pcaloss = clf.loss_curve_
    pred = clf.predict(x_test)


    ica = FastICA(n_components = 9)
    x_train = ica.fit_transform(pumpkin_x_train)
    x_test = ica.transform(pumpkin_x_test)
    cluster = GaussianMixture(n_components=2)
    additional_x_train = cluster.fit_predict(x_train)
    additional_x_test = cluster.predict(x_test)
    x_train = np.column_stack((pumpkin_x_train, additional_x_train))
    x_test = np.column_stack((pumpkin_x_test, additional_x_test))
    clf = MLPClassifier(hidden_layer_sizes = (128), max_iter=100, random_state=0)
    clf.fit(x_train, pumpkin_y_train)
    icaloss = clf.loss_curve_
    pred = clf.predict(x_test)
    

    rp = random_projection.GaussianRandomProjection(n_components = 10, compute_inverse_components=True)
    x_train = rp.fit_transform(pumpkin_x_train)
    x_test = rp.transform(pumpkin_x_test)
    cluster = GaussianMixture(n_components=2)
    additional_x_train = cluster.fit_predict(x_train)
    additional_x_test = cluster.predict(x_test)
    x_train = np.column_stack((pumpkin_x_train, additional_x_train))
    x_test = np.column_stack((pumpkin_x_test, additional_x_test))
    clf = MLPClassifier(hidden_layer_sizes = (128), max_iter=100, random_state=0)
    clf.fit(x_train, pumpkin_y_train)
    rploss = clf.loss_curve_
    pred = clf.predict(x_test)
    

    clf = ExtraTreesClassifier(n_estimators=50)
    clf.fit(pumpkin_x_train, pumpkin_y_train)
    model = SelectFromModel(clf, prefit=True)
    x_train = model.transform(pumpkin_x_train)
    x_test = model.transform(pumpkin_x_test)
    cluster = GaussianMixture(n_components=2)
    additional_x_train = cluster.fit_predict(x_train)
    additional_x_test = cluster.predict(x_test)
    x_train = np.column_stack((pumpkin_x_train, additional_x_train))
    x_test = np.column_stack((pumpkin_x_test, additional_x_test))
    clf = MLPClassifier(hidden_layer_sizes = (128), max_iter=100, random_state=0)
    clf.fit(x_train, pumpkin_y_train)
    dtloss = clf.loss_curve_
    pred = clf.predict(x_test)

    plt.plot(regularloss, label="Regular NN")
    plt.plot(pcaloss, label="PCA NN")
    plt.plot(icaloss, label="ICA NN")
    plt.plot(rploss, label="RP NN")
    plt.plot(dtloss, label="DT NN")
    plt.legend(loc="upper right")
    plt.title("Loss curves for additional EM data with different dimensionality reduction algorithms", wrap=True)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.savefig("plots/loss_curves_em.png")
    plt.close()

def section_5_part_2():
    clf = MLPClassifier(hidden_layer_sizes = (128), max_iter=100, random_state=0)
    clf.fit(pumpkin_x_train, pumpkin_y_train)
    regularloss = clf.loss_curve_
    pred = clf.predict(pumpkin_x_test)

    pca = PCA(n_components = 4)
    x_train = pca.fit_transform(pumpkin_x_train)
    x_test = pca.transform(pumpkin_x_test)
    cluster = KMeans(n_clusters=2, n_init="auto")
    additional_x_train = cluster.fit_predict(x_train)
    additional_x_test = cluster.predict(x_test)
    x_train = np.column_stack((pumpkin_x_train, additional_x_train))
    x_test = np.column_stack((pumpkin_x_test, additional_x_test))
    clf = MLPClassifier(hidden_layer_sizes = (128), max_iter=100, random_state=0)
    clf.fit(x_train, pumpkin_y_train)
    pcaloss = clf.loss_curve_
    pred = clf.predict(x_test)


    ica = FastICA(n_components = 9)
    x_train = ica.fit_transform(pumpkin_x_train)
    x_test = ica.transform(pumpkin_x_test)
    cluster = KMeans(n_clusters=2, n_init="auto")
    additional_x_train = cluster.fit_predict(x_train)
    additional_x_test = cluster.predict(x_test)
    x_train = np.column_stack((pumpkin_x_train, additional_x_train))
    x_test = np.column_stack((pumpkin_x_test, additional_x_test))
    clf = MLPClassifier(hidden_layer_sizes = (128), max_iter=100, random_state=0)
    clf.fit(x_train, pumpkin_y_train)
    icaloss = clf.loss_curve_
    pred = clf.predict(x_test)
    

    rp = random_projection.GaussianRandomProjection(n_components = 10, compute_inverse_components=True)
    x_train = rp.fit_transform(pumpkin_x_train)
    x_test = rp.transform(pumpkin_x_test)
    cluster = KMeans(n_clusters=2, n_init="auto")
    additional_x_train = cluster.fit_predict(x_train)
    additional_x_test = cluster.predict(x_test)
    x_train = np.column_stack((pumpkin_x_train, additional_x_train))
    x_test = np.column_stack((pumpkin_x_test, additional_x_test))
    clf = MLPClassifier(hidden_layer_sizes = (128), max_iter=100, random_state=0)
    clf.fit(x_train, pumpkin_y_train)
    rploss = clf.loss_curve_
    pred = clf.predict(x_test)
    

    clf = ExtraTreesClassifier(n_estimators=50)
    clf.fit(pumpkin_x_train, pumpkin_y_train)
    model = SelectFromModel(clf, prefit=True)
    x_train = model.transform(pumpkin_x_train)
    x_test = model.transform(pumpkin_x_test)
    cluster = KMeans(n_clusters=2, n_init="auto")
    additional_x_train = cluster.fit_predict(x_train)
    additional_x_test = cluster.predict(x_test)
    x_train = np.column_stack((pumpkin_x_train, additional_x_train))
    x_test = np.column_stack((pumpkin_x_test, additional_x_test))
    clf = MLPClassifier(hidden_layer_sizes = (128), max_iter=100, random_state=0)
    clf.fit(x_train, pumpkin_y_train)
    dtloss = clf.loss_curve_
    pred = clf.predict(x_test)

    plt.plot(regularloss, label="Regular NN")
    plt.plot(pcaloss, label="PCA NN")
    plt.plot(icaloss, label="ICA NN")
    plt.plot(rploss, label="RP NN")
    plt.plot(dtloss, label="DT NN")
    plt.legend(loc="upper right")
    plt.title("Loss curves for additional KNN data with different dimensionality reduction algorithms", wrap=True)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.savefig("plots/loss_curves_knn.png")
    plt.close()

section_1()
section_2()
section_3_part_1()
section_3_part_2()
section_3_part_3()
section_4()
section_5_part_1()
section_5_part_2()