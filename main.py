import streamlit as st
import matplotlib.pyplot as plt
from customized_clustering import point_plot
from customized_clustering import generate_data, n_cluster_animation,random_center,n_center_mapping,make_circle
import base64

st.title('Interactive KMean Clustering')

st.markdown(
    'Since you know how KMeans work, you should do it yourself now. Below is an interactive guide that will '
    'take you through each step but this time you can control the clustering'
)

st.markdown(
    '## Lets Create Our Data First \n'
    'These will be the number of our data points. You can select upt 500 points. In real life scenario you '
    'can go much higher but to save time we will stick to less number of data points'
)


data_size = int(st.number_input('Size of Data',2,500,value=2))

df = generate_data(data_size)

st.write(df)

data_plot = point_plot(df)

st.markdown(
    '## Ploting Data'
    'Lets see how our data is looking on a plot'
)

st.pyplot(data_plot)

st.markdown(
    '## Selecting Number of Clusters \n'
    'You can choose how many clusters you want to form. At this time a random guess would do. In real world '
    'scenario this number is guided by two thing. \n'
    '1. How many groups you want. This can be something like based on shopping record of customers we '
    'categorize them as good customers or bad customers. \n'
    '2. How much inertia your number of clusters have. We will cover this in detail going ahead \n'
    'For now a nice guess would work. You can select between 2 to 9 clusters.'
)

number_of_clust = int(st.number_input('Number of Clusters',2,8,value=2))

st.markdown(
    '## Generating Random Centers'
)

st.markdown(
    'The first step is to pick random data points and assume them to be centers, so lets see how they look'
)

random_center_pts = random_center(number_of_clust,df)

random_center_graph = n_center_mapping(df,random_center_pts)

st.pyplot(random_center_graph)

st.markdown(
    '## Let the algorithm computer the best clustering'
)


pl = st.empty()



a =st.button('Start Simulation')

if a:

    pl = st.empty()

    n_cluster_animation(df,pl,centers=random_center_pts)


st.markdown(
    '## Lets Talk About Inertia \n'
    'Another main component of selecting numbers of cluster is inertia. nertia measures how well a dataset was clustered by K-Means. '
    'It is calculated by measuring the distance between each data point and its centroid, squaring this distance, '
    'and summing these squares across one cluster. (ref- codeacademy.com) \n'
    'Like Everything else it best to show it visually'
)

fig,ax = plt.subplots()
fig.set_size_inches(10,3)
make_circle(20,100,color='r',ax=ax)
make_circle(-100,-20,color='b',ax=ax)
ax.scatter(60,0,color='r',s=150)
ax.scatter(-60,0,color='b',s=150)
ax.scatter(60,40,color='g')
ax.plot([60,60],[40,0])
#ax.plot([60,-60],[40,0])
st.pyplot(fig)

st.markdown(
    'Lets see the example above. There are 2 clusters of data, with their respective centroids. We take a point and we '
    'calculate its distance from the center it belongs to, and Square the distance. Lets say we get the value as 12 '
    'which is our inertia. But what if we only had one center?'
)

fig,ax = plt.subplots()
fig.set_size_inches(10,3)
make_circle(20,100,color='r',ax=ax)
make_circle(-100,-20,color='r',ax=ax)
ax.scatter(0,0,color='r',s=150)
ax.scatter(60,40,color='g')
ax.plot([60,0],[40,0])
st.pyplot(fig)

st.markdown(
    'In case of one cluster we can see the distance is higher from the center that. That means our value of inertia will '
    'be higher than when we had 2 center. Lets say the value is 16.'
)

st.markdown(
    'So we can say, two cluster are better than one, and we can also confirm this visually. But what about 3 cluster? '
    'Visually you can say that 3 clusters does not seem to be the right choice. But how does our algorithm will know this?'
    ' And how will you tell this in case of data that cannot be visually clustered, or data that cannot be plotted? '
    'So what happens when we have 3 clusters?'
)

fig,ax = plt.subplots()
fig.set_size_inches(10,3)
make_circle(20,100,color='r',ax=ax)
make_circle(-100,-20,color='b',ax=ax)
ax.scatter(60,0,color='r',s=150)
ax.scatter(-60,0,color='b',s=150)
ax.scatter(65,35,color='y',s=150)
ax.scatter(60,40,color='g')
ax.plot([60,60],[40,0])
ax.plot([60,65],[40,35])
#ax.plot([60,-60],[40,0])
st.pyplot(fig)

st.markdown(
    'We can see that a cluster close to the data point, will have smaller value of inertia. So that means three cluster '
    'are better than two? Visually it does not seem like that? So if we make the point a cluster itself the inertia value '
    'will be zero, because its distance to itself will be zero. So if we only go by the value of inertia then if we have '
    'as many number of cluster as we have data point then it will be the best clustering, but that itself sounds so wrong. '
    'Why do we even bother clustering than if we are going to say each point is unique. \n'
    'To overcome this issue we plot a graph between number of clusters and value of inertia. The actuall calculation of '
    'inertia is done by taking account of all the data points and all the clusters, but similar to the way we did for '
    'one point. \n'
    'So lets say we do KMean clustering with number of cluster as 3, and after that we get the value of Inertia as 6. '
    'We run the KMean clustering again but this time increasing the number of cluster to 4, we get the Inertia as 2. We '
    'do it again for 5 and got a value of 1.4, for 6 we got 1.2, for 7 cluster we got 1, for 8 cluster we got 0.8.'
)

st.markdown(
    'So we can see upto 4 cluster the value of Inertia dropped dramatically. After that when we increased number of '
    'clusters, the value of Inertia did not go down as dramatically. The point where value of Inertia stopped going '
    'down dramatically, we choose that point as number of clusters. So here the number of clusters is 4'
)

fig,ax = plt.subplots()
fig.set_size_inches(10,3)
make_circle(20,100,color='r',ax=ax)
make_circle(-100,-20,color='b',ax=ax)
ax.scatter(80,20,color='r',s=150)
ax.scatter(40,-20,color='b',s=150)
ax.scatter(-80,-20,color='y',s=150)
ax.scatter(-40,20,color='m',s=150)
#ax.plot([60,60],[40,0])
#ax.plot([60,65],[40,35])
#ax.plot([60,-60],[40,0])
st.pyplot(fig)

st.markdown(
    'So as we initially thought that the data is grouped into 2 clusters, it was even better to group our data in '
    'four clusters. A Inertia Graph is a good way to determine the value of K and it should be selected over our '
    'intutions. We should first cluster the data and then see the properties of the clusters instead of grouping '
    'cluster only upto the results we want. \n'
    'As an example, if you want to group your class into people who are athletic or not athletic, you might be tempted '
    'use number of clusters as 2. But you might find that the way your data is clustered you might find students good '
    'for one specific sports. And the number of students who can be promoted to play other sports at a higher level are '
    'more than you would have gotten out of selecting them on basis of athletic or non atheltic'
)

st.markdown(
    'In a real life scenario the tool you will use for clustering will have a prebuilt function to show you the value '
    'of inertia. Then you can plot it down on a graph. This is called the Elbow Method because the graph looks like '
    'shape of an elbow'
)

file_ = open("elbow_graph.png", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="kmean_gif">',
    unsafe_allow_html=True,
)