import pandas as pd
import streamlit as st
import random
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd

def make_circle(l,r,color,ax):
    radius = (r-l)/2
    x = [random.randrange(l,r) for i in range(r-l)]
    x_pos = []
    x_neg = []
    y_pos = []
    y_neg = []
    for i in range(len(x)):
        pt = (radius**2)-((x[i]-((l+r)/2))**2)
        pt = pt**(1 / 2)
        if x[i] > (l+r)/2:
            x_pos.append(x[i])
            y_pos.append(pt)

        else:
            x_neg.append(x[i])
            y_neg.append(-pt)


    ax.scatter(x_pos,y_pos,color=color)
    ax.scatter(x_neg,y_neg,color=color)

def generate_data(size,features=2):
    a = [[random.randrange(0,100) for j in range(features)] for i in range(size)]
    df= pd.DataFrame(a)
    return df

#df = generate_data(100)

def point_plot(data,feature1=0,feature2=1):

    fig, ax = plt.subplots()

    ax.scatter(data.iloc[:,feature1],data.iloc[:,feature2])

    plt.xlabel(data.columns[feature1])
    plt.ylabel(data.columns[feature2])
    plt.show()
    return fig

#point_plot(df)


def random_center(n_cluster,data,feature1=0,feature2=1):

    centers = []

    for i in range(n_cluster):
        record = random.randrange(0,len(data))
        center = (data.iloc[record, feature1], data.iloc[record, feature2])
        while center in centers:
            record = random.randrange(0, len(data))
            center = (data.iloc[record, feature1], data.iloc[record, feature2])
        centers.append(center)

    return centers





def n_center_mapping(data,centers,feature1=0,feature2=1,fig=None,ax=None):

    n = len(centers)
    if fig == None and ax== None:
        fig, ax = plt.subplots()

        ax.scatter(data.iloc[:, feature1], data.iloc[:, feature2],color='b')
        colors = ['#000000', '#FF0000', '#800080', '#FFFF00','#194d19', '#a52a2a', '#d2691e', '#dc143c', '#66cc66']
        for center,color in zip(centers,colors):
            ax.scatter(center[0], center[1], color=color, s=150)

        return fig

    ax.scatter(data.iloc[:, feature1], data.iloc[:, feature2], color='b')
    colors = ['#000000', '#FF0000', '#800080', '#FFFF00', '#194d19', '#a52a2a', '#d2691e', '#dc143c', '#66cc66']
    for center, color in zip(centers, colors):
        ax.scatter(center[0], center[1], color=color, s=150)
    plt.xlabel(data.columns[feature1])
    plt.ylabel(data.columns[feature2])
    #plt.show()

#fig = n_center_mapping(df,random_centers)
#plt.show()


def n_cluster_animation(data,slot,centers,feature1=0,feature2=1,max_iter=300,curr_iter=0):

    colors = ['#000000', '#FF0000', '#800080', '#FFFF00','#194d19', '#a52a2a', '#d2691e', '#dc143c', '#66cc66']

    fig, ax = plt.subplots()
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1])
    n_center_mapping(data,centers,fig=fig,ax=ax)
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    dict = {}

    for i in range(len(centers)):
        dict[i] = [colors[i]]

    for i in range(len(data)):
        curr_point = np.array([data.iloc[i,feature1],data.iloc[i,feature2]])
        closest_clust = 10000
        least_dist = float('inf')
        for j in range(len(centers)):


            curr_clust_pts = np.array(centers[j])
            dist = np.linalg.norm(curr_clust_pts - curr_point)

            if dist<least_dist:
                least_dist = dist
                closest_clust = j

        dict[closest_clust].append(curr_point.tolist())
        ax.scatter(data.iloc[i,0],data.iloc[i,1], color=colors[closest_clust])
    slot.pyplot(fig)


    new_center = []

    for i in range(len(centers)):
        x = []
        y= []
        for j in range(1, len(dict[i])):
            x.append(dict[i][j][0])
            y.append(dict[i][j][1])
        centroid = (sum(x) / len(x), sum(y) / len(y))
        new_center.append(centroid)

    if new_center == centers or curr_iter>max_iter:

        return

    n_cluster_animation(data,slot,new_center,curr_iter=(curr_iter+1))