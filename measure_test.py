import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools import *
from sklearn.cluster import KMeans

x=range(0,100)
y_sin=pd.DataFrame(np.sin(x))
y_cos=pd.DataFrame(np.cos(x)+0.5)
y_rnd2=pd.DataFrame(np.random.randn(len(x)))
y_rnd3=pd.DataFrame(np.random.randn(len(x)))

def compress(raw,num_clusters):
    classifier=KMeans(n_clusters=num_clusters,max_iter=300,tol=1e-4) # create classifier
    classifier.fit(raw) # perform classification
    return np.take(classifier.cluster_centers_,classifier.labels_) # the cluster center corresponding to each point

def sum_data(dfs):
    tmp=[d.T for d in dfs]
    tmp=pd.concat(tmp)
    return tmp.sum()

def avg_data(dfs):
    tmp=[d.T for d in dfs]
    tmp=pd.concat(tmp)
    return pd.DataFrame(tmp.mean())

def gl_err(raws, cs):
    raw_avg=avg_data(raws)
    c_avg=avg_data(cs)
    return error_formula(raw_avg,c_avg).mean()

def pearson_coeff_sample(c1,c2):
    """
    Compute the pearson coefficient for two samples

    Args:
    c1,c2: the samples, pandas column dataframes
    """
    x=np.asarray(c1.T)
    y=np.asarray(c2.T)
    if len(x)==1:
        x=x[0]
    if len(y)==1:
        y=y[0]
    d_x=x-x.mean()
    d_y=y-y.mean()
    return sum(d_x*d_y)/(np.sqrt(sum(d_x*d_x))*np.sqrt(sum(d_y*d_y)))

def body(num_clusters):
    raws=[y_sin,y_cos,y_rnd2,y_rnd3]
    compressed=[pd.DataFrame(compress(raw,c)) for raw,c in zip(raws,num_clusters)]
    # no grouping
    lerrs=[pearson_coeff_sample(raw,c) for raw,c in zip(raws,compressed)]
    avg_lerr=np.mean(lerrs)

    gerr=pearson_coeff_sample(sum_data(raws),sum_data(compressed))

    # grouping
    #groups=[[0,1],[2,3]]
    groups=np.array([1,1,2,2])
    group_data=[avg_data([c for c,g1 in zip(compressed,groups) if g==g1]) for g in groups]
    g_lerrs=[pearson_coeff_sample(raw,c) for raw,c in zip(raws,group_data)]
    avg_g_lerr=np.mean(g_lerrs)

    g_gerr=pearson_coeff_sample(sum_data(raws),sum_data(group_data))
    return compressed[0],lerrs[0],g_lerrs[0],compressed[1],lerrs[1],g_lerrs[1]

for num_clusters in [[50,50,50,50],[50,5,50,50],[50,2,50,50],[50,1,50,50]]:
    c1,l1,gl1,c2,l2,gl2=body(num_clusters)
    fig,ax=plt.subplots()
    ax.set_ymargin(0.1)        # increase the margin on top of the graph, allows to see points that have a maximum value
    fig.suptitle("Compressing with "+str(num_clusters[0])+" and "+str(num_clusters[1]))
    ax.plot(x,np.asarray(c1.T)[0],color="r")
    ax.plot(x,np.asarray(c2.T)[0],color="b")
    ax2=ax.twiny()
    ax2.axvline(l1,color="r")
    ax2.axvline(l2,color="b")
    ax2.axvline(gl1,color="r",ls="dashed")
    ax2.axvline(gl2,color="b",ls="dashed")
    ax2.legend(["Lerr1","Lerr2","Gerr1","Gerr2"],loc=3)
    fig.savefig("./measure_test_"+str(num_clusters[0])+"_"+str(num_clusters[1])+".pdf",format='pdf')
    plt.close(fig)
