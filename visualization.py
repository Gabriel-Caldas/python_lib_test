import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import seaborn as sns
from pandas.plotting import scatter_matrix
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

import exploratory as exp
from vindi_sets import VindiColors

def print_test():
    print('SUCESS!')


def plot_corr_matrix(corr):
#     corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(255, 15, s=75, l=40,
                                center="light", as_cmap=True)
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, mask=mask, center=0, annot=True,
                fmt='.2f', square=True, cmap=cmap)

    plt.show()

def plot_scatter_cols_relations(df, y_col, robust=False):
    for x_col in [x_cols for x_cols in df if x_cols != y_col]:
        x = df[x_col]
        y = df[y_col]

        plt.figure(figsize=(10,5))
        plt.title(f"{y_col} distribuition towards {x_col}", y=1.03)

        # Calculate the point density
        xy = np.vstack([x,y])
        z = stats.gaussian_kde(xy)(xy)

        plt.scatter(df[x_col], df[y_col], s=100, c=z)
        plt.vlines(df[x_col].quantile([0, 0.25, 0.5, 0.75, 1]), df[y_col].min(), df[y_col].max(), 
                   colors=VindiColors.orange, label='Quantiles, 0 to 1 at .25 step)')
        plt.colorbar()
        plt.legend()
        plt.xlabel(x_col)

        if robust:
            inf_lim, sup_lim = exp.get_limits(df[x_col])
            inf_outs = (df[x_col] < inf_lim).any()
            sup_outs = (df[x_col] > sup_lim).any()
            
            if inf_outs and sup_outs:
                plt.xlim(left=inf_lim, right=sup_lim)
            elif inf_outs:
                rlim = df[x_col].max() + ((df[x_col].max()-inf_lim)*0.05)
                plt.xlim(left=inf_lim, right=rlim)
            elif sup_outs:
                llim = df[x_col].min() - ((sup_lim-df[x_col].min())*0.05)
                plt.xlim(left=llim, right=sup_lim)
        
        plt.ylabel(y_col)
        plt.grid(alpha=.4)
        plt.show()

def subplots_scatter_cols_relations(df, y_col, robust=False):
    ncols = 3
    nrows = int(np.ceil(df.shape[1]/ncols))
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(12, 4*nrows))
    
    x_cols = [x_cols for x_cols in df if x_cols != y_col]
    idx_col = 0
    
    for row in range(nrows):
        for col in range(ncols):
            if idx_col >= len(x_cols):
                break
            x_col = x_cols[idx_col]
            x = df[x_cols[idx_col]]
            y = df[y_col]
            
            axs[row][col].set_title(f"{x_col}")

            # Calculate the point density
            xy = np.vstack([x,y])
            z = stats.gaussian_kde(xy)(xy)

            axs[row][col].scatter(df[x_col], df[y_col], s=100, c=z)
            axs[row][col].vlines(df[x_col].quantile([0, 0.25, 0.5, 0.75, 1]), df[y_col].min(), df[y_col].max(), 
                       colors=VindiColors.orange, label='Quantiles, 0 to 1 at .25 step)')
            axs[row][col].set_xlabel(x_col)

            if robust:
                inf_lim, sup_lim = exp.get_limits(df[x_col])
                inf_outs = (df[x_col] < inf_lim).any()
                sup_outs = (df[x_col] > sup_lim).any()

                if inf_outs and sup_outs:
                    axs[row][col].set_xlim(left=inf_lim, right=sup_lim)
                elif inf_outs:
                    rlim = df[x_col].max() + ((df[x_col].max()-inf_lim)*0.05)
                    axs[row][col].set_xlim(left=inf_lim, right=rlim)
                elif sup_outs:
                    llim = df[x_col].min() - ((sup_lim-df[x_col].min())*0.05)
                    axs[row][col].set_xlim(left=llim, right=sup_lim)
            
            if col == 0:
                axs[row][col].set_ylabel(y_col)
            axs[row][col].grid(alpha=.4)
            idx_col+=1

    plt.suptitle(f"Correlations with \'{y_col}\' parameter", y=.92)
    plt.subplots_adjust(hspace=0.4)
    plt.show()


def plot_kmeans_inercias_and_silhouettes(df, n_clusters, figsize=(10,4)):
    inertias, silhouette_scores = exp.get_kmeans_inercias_and_silhouettes(df, n_clusters).values()
    
    fig, axs = plt.subplots(2, figsize=figsize, sharex=True)
    axs[0].plot(range(1, n_clusters), inertias, 'o-', color=VindiColors.blue)
    axs[0].set_xticks([*range(1, n_clusters)])
    axs[0].set_ylabel("Inertia", fontsize=14)
    axs[0].set_yticks(np.linspace(0, max(inertias), 5, dtype=int))
    axs[0].grid()
    axs[1].plot(range(2, n_clusters), silhouette_scores, "o-", color=VindiColors.blue)
    axs[1].set_xlabel("$k$", fontsize=14)
    axs[1].set_ylabel("Silhouette score", fontsize=14)
    axs[1].set_yticks(np.linspace(min(silhouette_scores), max(silhouette_scores), 5))
    axs[1].grid()
    plt.show()


def plot_kmeans_silhouette_diagrams(df, ks=[2, 3, 4, 5]):

    kmeans_estimators = [KMeans(n_clusters=k, random_state=42).fit(df) for k in ks]

    ncols = 2
    nrows = int(np.ceil(len(ks)/ncols))
    plt.figure(figsize=(10, 4*nrows))
    for idx, k in enumerate(ks):
        plt.subplot(nrows, ncols, idx+1)

        y_pred = kmeans_estimators[idx].labels_
        silhouette_coefficients = silhouette_samples(df, y_pred)
        silhouette_score = silhouette_coefficients.mean()

        padding = len(df) // 30
        pos = padding
        ticks = []
        for i in range(k):
            coeffs = silhouette_coefficients[y_pred == i]
            coeffs.sort()

            color = mpl.cm.Spectral(i / k)
            plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                              facecolor=color, edgecolor=color, alpha=0.7)
            ticks.append(pos + len(coeffs) // 2)
            pos += len(coeffs) + padding

        plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
        plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))

        if k in (2, 7):
            plt.ylabel("Cluster")

        if k in (7, 10):
            plt.xlabel("Silhouette Coefficient")
        else:
            plt.tick_params(labelbottom=True)

        plt.axvline(x=silhouette_score, color="red", linestyle="--")
        plt.title("$k={}$".format(k), fontsize=16)

    plt.show()