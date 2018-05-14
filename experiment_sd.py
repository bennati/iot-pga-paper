import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tools import *

max_compr=46
min_compr=4

parser = argparse.ArgumentParser(description='reads in parameters')

# Add the arguments for the parser to be passed on the cmd-line
# Defaults could be added default=
parser.add_argument('--data_dir', metavar='data_dir', nargs=1,help='the data directory')
parser.add_argument('--start_compression', nargs=1,type=int,help='the compression level')
parser.add_argument('--group_size', nargs='?',default=1,type=int,help='size of groups of users')
parser.add_argument('--min_user', metavar='min_user', nargs='?',default=1000,type=int,help='index of first user to include in analysis')
parser.add_argument('--max_user', metavar='max_user', nargs='?',default=7444,type=int,help='index of last user to include in analysis')
parser.add_argument('--num_rep', metavar='num_rep', nargs='?',default=1,type=int,help='number of repetitions')
#parser.add_argument('--aggr_fct', metavar='aggr_fct', nargs='?',default="mean",help='The aggregation function, sum or mean. Defaults to mean')

args = parser.parse_args()

def rescale_sample(sample,indexes):
    if sample>=-1 and sample<=1:
        # scale to have range 1
        sample/=2
        # center in 1
        sample+=0.5
        assert(sample>=0 and sample<=1)
        # rescale between min_user and max_user
        rng=max(indexes)-min(indexes)
        sample*=rng
        # cast to integer
        sample=int(sample)
        #translate
        sample+=min(indexes)
        assert(sample>=min(indexes) and sample<=max(indexes))
        return sample
    else:
        return None             # repeat the while loop

def fctn(item):
    key=item[0]
    val=item[1]
    if type(key) != list:
        key=[key]
    if type(val) != list:
        val=[val]
    (data,dummy,dummy2)=generate_daily_data(key,val)
    if data.empty:
        return data
    else:
        data=average_data([data])
        return data[['TIME','RAW','CENTROID','ERROR']]

def get_dict_pairs(dic,keys):
    """
    Returns: a list (k,v) where k is the key or a list of keys, and v is a value or a list of values corresponding to k
    """
    if type(keys) is np.int64 or type(keys) is np.float64 or type(keys) is int or type(keys) is float:
        return [keys,dic[keys]]
    elif type(keys) is np.ndarray or type(keys) is list:
        return [list(keys),[dic[k] for k in keys]]
    else:
        raise TypeError("Warning: unknown type "+str(type(keys)))

def experiment(data_dir,start_compression,min_user,max_user,group_size=1,num_rep=1):
    '''
    - start with the same compression level assigned to each user
    - then at every step redistribute some clusters with gaussian probability
    - compute the standard deviation of the new configuration, and local & global errors
    - repeat, increasing the std

    If group_size is set, group randomly the users
    '''
    indexes=range(min_user,max_user)
    compr_dict=dict([[i,start_compression] for i in indexes])
    compr_tab=pd.DataFrame(index=indexes) # table where to save the distribution of compression level. one row for each user, one column for each SD.
    compr_tab['1']=start_compression                 # initial values
    n_flips=10
    n_clusters=4
    points=[]
    last_std=0
    t=0
    keep=True
    if __name__ == '__main__':
        pool=Pool()
    while keep and t<100000 and last_std<14:
        print "loop "+str(t)
        t+=1
        # Compute the compression levels
        for _ in range(n_flips):
            keys=compr_dict.keys()
            np.random.shuffle(keys)
            try:
                src=next(i for i in keys if compr_dict[i]<=max_compr-n_clusters)
            except:
                keep=False      # stop here
            dest=None
            timeout=0
            while timeout<=100 and (not dest or compr_dict[dest]<=min_compr+n_clusters): # if the source has too few clusters
                timeout+=1
                dest=rescale_sample(np.random.normal(scale=0.35),indexes)
            # update dictionary
            if timeout<=100:
                compr_dict[src]+=n_clusters    # add clusters (decompress)
                compr_dict[dest]-=n_clusters    # remove clusters (compress)
            else:
                keep=False      # stop here
        for k in compr_dict.keys():
            assert(compr_dict[k]<=max_compr and compr_dict[k]>=min_compr)
        # check the standard deviation of the current configuration
        std=int(np.std(compr_dict.values()))
        if std>last_std:        # new point
            last_std=std
            ## save the new distribution
            distr=pd.DataFrame.from_dict(compr_dict,orient='index') # the current distribution
            distr.columns=[std]
            compr_tab=pd.merge(compr_tab,distr,left_index=True,right_index=True,how='outer') # integrate with the old data
            partial=pd.DataFrame()
            for _ in range(num_rep):
                keys=compr_dict.keys()
                keys=group_randomly(keys,group_size,group_size) # group users randomly
                try:
                    items=[get_dict_pairs(compr_dict,group) for group in keys] # collect their compressions
                except ValueError, e:
                    print e.value
                    items=[(i,j) for i,j in compr_dict.iteritems()]
                # execute the compression and compute the errors
                if __name__ == '__main__':
                    ans=pool.map(fctn,items)
                else:
                    ans=map(fctn,items) # compress the user data with the respective compression level
                try:
                    ans=sum_data(ans,counter=True)
                    ans=aggregate_data(ans)
                    ans['GLOBAL_ERROR']=compute_error(ans)      # compute the global error
                except DataFrameEmptyError , e:
                    print e.value
                    ans=pd.DataFrame()
                # ans['COUNTER']=1
                try:
                    partial=sum_data([partial,ans],counter=False)
                except DataFrameEmptyError , e:
                    print e.value
            try:
                ans=average_data([partial])
                ans['GLOBAL_ERROR']=compute_error(ans)      # compute the global error
                points.append([std,ans['ERROR'].mean(),ans['ERROR'].std(),ans['GLOBAL_ERROR'].mean(),ans['GLOBAL_ERROR'].std()]) # save the values
            except DataFrameEmptyError , e:
                print e.value
        else:
            print "skipping"
    points=pd.DataFrame(points,columns=['std','Error','Error std','Global_Error','Global_Error std'])
    points.to_csv(os.path.join(data_dir,"std_errors_group_size_"+str(group_size)+"_"+str(min_user)+"_"+str(max_user)+".csv"),sep=",",index=False)
    compr_tab.to_csv(os.path.join(data_dir,"distrib_group_size_"+str(group_size)+"_"+str(min_user)+"_"+str(max_user)+".csv"),sep=",",index=True)

def experiment2(data_dir,min_user,max_user,num_rep=1):
    '''
    run a simulation with half population having compression min_compr and half having max_compr
    '''
    indexes=range(min_user,max_user)
    compr_dict=dict([[i,min_compr if indexes.index(i) < len(indexes)/2 else max_compr] for i in indexes]) # half and half
    compr_tab=pd.DataFrame(index=indexes) # table where to save the distribution of compression level. one row for each user, one column for each SD.
    compr_tab['1']=[min_compr if indexes.index(i) < len(indexes)/2 else max_compr for i in indexes]                 # initial values
    points=[]
    std=int(np.std(compr_dict.values()))
    if __name__ == '__main__':
        pool=Pool()
    partial=pd.DataFrame()
    for _ in range(num_rep):
        # execute the compression and compute the errors
        if __name__ == '__main__':
            ans=pool.map(fctn,compr_dict.iteritems())
        else:
            ans=map(fctn,compr_dict.iteritems()) # compress the user data with the respective compression level
        try:
            ans=average_data(ans,counter=True)
        except DataFrameEmptyError , e:
            print e.value
            ans=pd.DataFrame()
        # ans['COUNTER']=1
        try:
            partial=sum_data([partial,ans],counter=False)
        except DataFrameEmptyError , e:
            print e.value
    try:
        ans=average_data([partial])
        ans['GLOBAL_ERROR']=compute_error(ans)      # compute the global error
        points.append([std,ans['ERROR'].mean(),ans['GLOBAL_ERROR'].mean()]) # save the values
    except DataFrameEmptyError , e:
        print e.value
    points=pd.DataFrame(points,columns=['std','Error','Global_Error'])
    points.to_csv(os.path.join(data_dir,"std_errors_exp2_"+str(min_user)+"_"+str(max_user)+".csv"),sep=",",index=False)

def read_std_errors(data_dir,group_size):
    data=pd.DataFrame()
    files=[f for f in os.listdir(data_dir) if f.startswith('std_errors_group_size_'+str(group_size)+"_")]
    for f in files:
        temp=pd.read_csv(os.path.join(data_dir,f), delimiter=',')
        data=pd.concat([data,temp])
    # files=[f for f in os.listdir(data_dir) if f.startswith('std_errors_exp2')]
    # if len(files)==1 and group_size==1:
    #     temp=pd.read_csv(os.path.join(data_dir,files[0]), delimiter=',')
    #     data=pd.concat([data,temp])
    return data.groupby(["std"], as_index=False).mean()

def plot(data_dir,group_size=1):
    data=read_std_errors(data_dir,group_size)
    fig, ax1 = plt.subplots()
    fig.suptitle("Effect of sd on local and global error")
    plt.xlabel("Standard Deviation")
    ax1.plot(data['std'],data['Error'],'b')
    ax1.set_ylabel("Error")
    ax1.set_yscale('log')
    #ax1.legend(["Cost Cen","Cost Dec","Priv Cen","Priv Dec"],loc=2)
    # ax2=ax1.twinx()
    ax1.plot(data['std'],data['Global_Error'],'r')
    ## Make the y-axis label and tick labels match the line color.
    # ax2.set_ylabel('Global_Error', color='r')
    # for tl in ax2.get_yticklabels():
    #     tl.set_color('r')
    #ax2.legend(["Score Cen","Score Dec"],loc=1)
    ax1.legend(["Local error","Global error"],loc=2)
    fig.savefig(os.path.join(data_dir,"experiment_sd_size_"+str(group_size)+".pdf"),format='pdf')
    plt.close(fig)
    ## plot distributions
    data=pd.DataFrame()
    files=[f for f in os.listdir(data_dir) if f.startswith('distrib_group_size_'+str(group_size))]
    for f in files:
        temp=pd.read_csv(os.path.join(data_dir,f), delimiter=',')
        data=pd.concat([data,temp])
    data.rename(columns={'Unnamed: 0': 'user', '1': '0', '1.1':'1'}, inplace=True)
    heatmap=np.zeros((49,len(data.columns)))
    for i in range(len(data.columns[1:])): # over all columns
        vals=data[data.columns[1:][i]].value_counts()
        for j in vals.index:
            heatmap[j][i]=vals[j]
    fig=plt.figure()
    fig.suptitle("Distribution of compression levels")
    plt.imshow(heatmap,interpolation='none')
    plt.ylabel("Compression")
    plt.xlabel("Standard deviation")
    plt.xticks(range(heatmap.shape[1]),map(int,data.columns[1:]))
    plt.gca().invert_yaxis()
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_ticks_position('both')
    cbaxes = fig.add_axes([0.08, 0.1, 0.03, 0.8]) # position colorbar on the left
    plt.colorbar(cax=cbaxes)
    fig.savefig(os.path.join(data_dir,"distribution_sd_"+str(group_size)+".pdf"),format='pdf')
    plt.close(fig)

def plot_hmap(heatmap,title,filename,plot_dir,xticks=None):
    fig=plt.figure()
    fig.suptitle(title)
    plt.imshow(heatmap,interpolation='none')
    plt.ylabel("Standard Deviation")
    plt.xlabel("Group size")
    plt.gca().invert_yaxis()
    plt.gca().yaxis.set_ticks_position('both')
    if xticks:
        plt.gca().xaxis.set_ticklabels(xticks)
        plt.gca().xaxis.set_ticks(np.asarray(xticks) - 1)
    plt.colorbar()
    fig.savefig(os.path.join(plot_dir,filename),format='pdf')
    plt.close(fig)

def plot_heatmaps(data_dir,min_group=1,max_group=3):
    groups=range(min_group,max_group+1)
    #data=[[]]*len(groups)
    data=read_std_errors(data_dir,groups[0])
    heat_le=pd.DataFrame(index=range(data.shape[0]),columns=range(len(groups)))
    heat_ge=pd.DataFrame(index=range(data.shape[0]),columns=range(len(groups)))
    for i in range(len(groups)):
        data=read_std_errors(data_dir,groups[i])
        heat_le[i]=data["Error"]
        heat_ge[i]=data["Global_Error"]
    plot_hmap(heat_le,"Local error","std_dev_group_size_heatmap_le.pdf",data_dir,xticks=groups)
    plot_hmap(heat_ge,"Global error","std_dev_group_size_heatmap_ge.pdf",data_dir,xticks=groups)

###########################################################################
# ------------------------------ EXECUTION ------------------------------ #
###########################################################################

experiment(args.data_dir[0],args.start_compression[0], args.min_user, args.max_user,args.group_size,args.num_rep)
