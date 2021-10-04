import pandas as pd
import streamlit as st
import random
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd

def generate_data(size,features=2):
    a = [[random.randrange(0,100) for j in range(features)] for i in range(size)]
    df= pd.DataFrame(a)
    return df

df = generate_data(100)

def point_plot(data,feature1=0,feature2=1):

    fig, ax = plt.subplots()

    ax.scatter(data.iloc[:,feature1],data.iloc[:,feature2])

    plt.xlabel(data.columns[feature1])
    plt.ylabel(data.columns[feature2])
    plt.show()
    return fig

point_plot(df)

def random_center(n_cluster,data,feature1=0,feature2=1):

    centers = []

    for i in range(n_cluster):
        record = random.randrange(0,len(df))
        center = (data.iloc[record, feature1], data.iloc[record, feature2])
        while center in centers:
            record = random.randrange(0, len(df))
            center = (data.iloc[record, 0], data.iloc[record, 1])
        centers.append(center)

    return centers

print(df)

random_centers = random_center(4,df)
print(random_centers)

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
    #plt.xlabel(data.columns[feature1])
    #plt.ylabel(data.columns[feature2])


#n_center_mapping(df,random_centers)

def n_cluster_animation(data,slot,centers=random_centers,feature1=0,feature2=1,max_iter=300,curr_iter=0):

    colors = ['#000000', '#FF0000', '#800080', '#FFFF00','#194d19', '#a52a2a', '#d2691e', '#dc143c', '#66cc66']

    fig, ax = plt.subplots()
    # print("center_red",center_red)
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1])
    n_center_mapping(data,centers,fig=fig,ax=ax)
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    dict = {}

    for i in range(len(centers)):
        dict[i] = [colors[i]]

    for i in range(len(data)):
        curr_point = np.array([data.iloc[i,feature1],data.iloc[i,feature2]])
        #print(curr_point)
        closest_clust = 10000
        least_dist = float('inf')
        for j in range(len(centers)):

            #print(centers[j])
            curr_clust_pts = np.array(centers[j])
            #print('curr_clust_pt',curr_clust_pts)
            dist = np.linalg.norm(curr_clust_pts - curr_point)

            if dist<least_dist:
                least_dist = dist
                closest_clust = j

        dict[closest_clust].append(curr_point.tolist())
        ax.scatter(data.iloc[i,0],data.iloc[i,1], color=colors[closest_clust])
    slot.pyplot(fig)
    print(dict)

    new_center = []

    for i in range(len(centers)):
        x = []
        y= []
        for j in range(1, len(dict[i])):
            # print(dict[0][i][0])
            x.append(dict[i][j][0])
            y.append(dict[i][j][1])
        centroid = (sum(x) / len(x), sum(y) / len(y))
        new_center.append(centroid)
    print(new_center)
    if new_center == centers or curr_iter>max_iter:

        return

    n_cluster_animation(data,slot,new_center,curr_iter=(curr_iter+1))