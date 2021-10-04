import pandas as pd
import streamlit as st
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

a = [[random.randrange(0,100) for j in range(2)] for i in range(400)]
df = pd.DataFrame(a)
print(df)

def centroid_plot(data,center_red,center_green,feature1=0,feature2=1):

    fig, ax = plt.subplots()


    ax.scatter(data.iloc[:,feature1],data.iloc[:,feature2])
    ax.scatter(center_red[0], center_red[1], color='red', s=150)
    ax.scatter(center_green[0], center_green[1], color='green', s=150)
    plt.xlabel(data.columns[feature1])
    plt.ylabel(data.columns[feature2])
    plt.show()
    st.pyplot(fig)

def point_plot(data,feature1=0,feature2=1):

    fig, ax = plt.subplots()

    ax.scatter(data.iloc[:,feature1],data.iloc[:,feature2])

    plt.xlabel(data.columns[feature1])
    plt.ylabel(data.columns[feature2])
    plt.show()
    st.pyplot(fig)

#centroid_plot(df,(35,60),(80,15))
#point_plot(df)

pl = st.empty()
def grouping_animation(data,center_red=None,center_green=None,max_iter=300,curr_iter=0):
    n = len(data)

    if center_red == None and center_green == None:
        record1 = random.randrange(0, n)
        record2 = random.randrange(0, n)
        center_red = np.array([data.iloc[record1,0],data.iloc[record1,1]])
        center_green = np.array([data.iloc[record2,0],data.iloc[record2,1]])
    else:
        print('new-center',center_red)
        center_red = np.array([center_red[0],center_red[1]])
        center_green = np.array([center_green[0],center_green[1]])

    fig, ax = plt.subplots()
    #print("center_red",center_red)
    ax.scatter(data.iloc[:,0],data.iloc[:,1])
    ax.scatter(center_red.tolist()[0], center_red.tolist()[1], color='orange', s=200)
    ax.scatter(center_green.tolist()[0], center_green.tolist()[1], color='cyan', s=200)
    ax.text(100,110,'Generation: {}'.format(curr_iter))
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])

    red_pt = [[],[]]
    green_pt = [[],[]]

    j = 0

    for i in range(n):
        curr_point = np.array([data.iloc[i,0],data.iloc[i,1]])

        dist_red = np.linalg.norm(center_red - curr_point)
        dist_green = np.linalg.norm(center_green - curr_point)

        # if j%10 == 0:
        #     ax.plot((data.iloc[i, 0], center_red.tolist()[0]), (data.iloc[i, 1],center_red.tolist()[1]), color='red')
        #     ax.plot((data.iloc[i, 0], center_green.tolist()[0]), (data.iloc[i, 1],center_green.tolist()[1]), color='green')


        if dist_red<dist_green:

            ax.scatter(data.iloc[i,0],data.iloc[i,1], color='red')

            plt.xlabel(data.columns[0])
            plt.ylabel(data.columns[1])
            red_pt[0].append(curr_point.tolist()[0])
            red_pt[1].append(curr_point.tolist()[1])


        else:
            ax.scatter(data.iloc[i, 0], data.iloc[i, 1], color='green')

            plt.xlabel(data.columns[0])
            plt.ylabel(data.columns[1])
            green_pt.append(curr_point)
            green_pt[0].append(curr_point.tolist()[0])
            green_pt[1].append(curr_point.tolist()[1])

        j += 1
    pl.pyplot(fig)
    center_red = center_red.tolist()
    center_green = center_green.tolist()

    new_red_center = list((sum(red_pt[0]) / len(red_pt[0]), sum(red_pt[1]) / len(red_pt[1])))
    new_green_center = list((sum(green_pt[0]) / len(green_pt[0]), sum(green_pt[1]) / len(green_pt[1])))

    if (center_red == (new_red_center) and center_green == new_green_center) or curr_iter>max_iter:
        st.write('End of Clustering')
        return

    else:

        grouping_animation(data,center_red=new_red_center,center_green=new_green_center,curr_iter=(curr_iter+1))

grouping_animation(df)
