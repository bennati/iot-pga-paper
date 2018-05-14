import argparse
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
import functools
#import random
from tools import *

# data_dir="test"
# compression_levels=[10,10,10]
# num_intervals=20
# num_rep=2
# min_user=1000
# max_user=1010
# group_fraction=1
# grp_size_distr=None
# input_dir="./datasets/ecbt/run_48/output/daily/users/cluster48/"

parser = argparse.ArgumentParser(description='reads in parameters')

# Add the arguments for the parser to be passed on the cmd-line
# Defaults could be added default=
parser.add_argument('--data_dir', metavar='data_dir', nargs=1,help='the output directory')
parser.add_argument('--compression_levels', nargs='+',type=int,help='the compression level')
parser.add_argument('--num_intervals', metavar='num_intervals', nargs='?',default=1,type=int,help='number of subsets in which to divide user data. Defaults to 1')
parser.add_argument('--num_rep', metavar='num_rep', nargs='?',default=1,type=int,help='number of repetitions. Defaults to 1')
parser.add_argument('--group_fraction', metavar='group_fraction', nargs='?',default=1.0,type=float,help='fraction of users to group. Defaults to all.')
parser.add_argument('--min_user', metavar='min_user', nargs='?',default=1000,type=int,help='index of first user to include in analysis')
parser.add_argument('--max_user', metavar='max_user', nargs='?',default=7444,type=int,help='index of last user to include in analysis')
#parser.add_argument('--aggr_fct', metavar='aggr_fct', nargs='?',default="mean",help='The aggregation function, sum or mean. Defaults to mean')
parser.add_argument('--grp_size_distr', metavar='grp_size_distr', nargs='?',default=None,help='The distribution from where to sample the group size. Either randint or randint_power_law. Defaults to deterministic.')
#parser.add_argument('--input_dir', metavar='input_dir', nargs='?',default="./datasets/ecbt/run_48/output/daily/users/cluster48/",help='the directory where to find the data')

args = parser.parse_args()

def main(data_dir,compression_levels,num_intervals,num_rep,min_user,max_user,group_fraction,grp_size_distr):
    #analysis_ecbt(**locals())
    #analysis_nrel(**locals())
    exp_clustering_ecbt(**locals())

def random_summarization_fct():
    np.random.seed()
    return np.random.randint(2,10)

def exp_clustering_ecbt(data_dir,compression_levels,num_intervals,num_rep,min_user,max_user,group_fraction,grp_size_distr):
    num_groups=compression_levels[0]
    print "Grouping users in "+str(num_groups)+" clusters"
    input_dir="../results_ecbt/datasets/ecbt/run_48/output/daily/users/cluster48/"
    files=select_files(input_dir,range(min_user,max_user+1))
    max_std=13
    part_fun=functools.partial(exp_clustering_std,data_dir=data_dir,files=files,num_groups=num_groups,input_dir=input_dir,max_std=max_std)
    # if __name__ == '__main__':
    #     pool=Pool()
    #     print "starting processes"
    #     ans=pool.map(part_fun,range(num_rep))
    #     pool.close()
    #     pool.join()
    # else:
    #     ans=map(part_fun,range(num_rep))
    ans=map(part_fun,range(num_rep))
    hist=pd.concat(ans)
    hist.to_csv(os.path.join(data_dir,"distrib_group_size_"+str(num_groups)+".csv"),sep=",",index=True,compression="bz2")
    ## aggregate results from different simulations
    for fct in ["cl_data","cl_sum","cl_rnd"]:
        for s in range(max_std):
            partial_data=pd.DataFrame()
            filename="exp_clustering_"+str(num_groups)+"_fct_"+str(fct)+"_std_"+str(s)
            for i in range(num_rep):
                try:
                    df=pd.read_csv(os.path.join(data_dir,"indiv","daily_"+filename+"_rep_"+str(i)+".csv.bz2"),compression="bz2")
                    if not df.empty:
                        partial_data=pd.concat([partial_data,df])
                    else:
                        print "File "+"daily_"+filename+"_rep_"+str(i)+" is empty"
                except:
                    print "Skipping file daily_"+filename+"_rep_"+str(i)+": not found"
            if not partial_data.empty:
                partial_data=average_data([partial_data],compute_stds=True)
                partial_data.to_csv(os.path.join(data_dir,"indiv","daily_"+filename+".csv.bz2"),index=False,compression="bz2")
                aggregated=aggregate_daily(partial_data)
                aggregated.to_csv(os.path.join(data_dir,"indiv","aggregated_"+filename+".csv.bz2"),index=False,compression="bz2")
            else:
                print "Skipping "+filename

def analysis_ecbt(data_dir,compression_levels,num_intervals,num_rep,min_user,max_user,group_fraction,grp_size_distr):
    input_dir="./datasets/ecbt/run_48/output/daily/users/cluster48/"
    files=select_files(input_dir,range(min_user,max_user+1))
    analysis_fct=generate_daily_data
    body(data_dir,files,compression_levels,num_intervals,num_rep,group_fraction,grp_size_distr,analysis_fct,input_dir)

def analysis_nrel(data_dir,compression_levels,num_intervals,num_rep,min_user,max_user,group_fraction,grp_size_distr):
    input_dir="./datasets/nrel_southern_nevada/rtcsnv_sorted_by_person/"
    files=select_files_nrel(input_dir)
    files=preprocess_data_nrel(files,compression_levels) # keep only users with enough entries
    analysis_fct=generate_daily_data_nrel
    body(data_dir,files,compression_levels,num_intervals,num_rep,group_fraction,grp_size_distr,analysis_fct,input_dir)

def body(data_dir,files,compression_levels,num_intervals,num_rep,group_fraction,grp_size_distr,analysis_fct,input_dir):
    #random.seed(10) #TODO: CHANGE SEED FOR SERIOUS WORK!!
    nlevs=len(compression_levels)
    assert(nlevs>=1)
    filename=(str(compression_levels[0])
              if len(compression_levels)==1 else
              "_".join(map(str,compression_levels))
    )+(""
       if group_fraction==1.0 else
       "_fract"+str(group_fraction)
    )+".csv"
    partial_data=pd.DataFrame()
    partial_errors=[pd.DataFrame() for _ in range(nlevs)]
    partial_group_errors=[pd.DataFrame() for _ in range(nlevs)]
    partial_corrs=[pd.DataFrame() for _ in range(nlevs)]
    partial_group_corrs=[pd.DataFrame() for _ in range(nlevs)]
    partial_hist=[]
    n1=min(2,nlevs) # 2 or 1 if not grouping
    n2=nlevs
    for r in range(num_rep):    # for each repetition
        print "Repetition "+str(r)
        f=(None if not grp_size_distr else globals()[grp_size_distr]) # if None, the groups are assigned deterministically and of size = n2
        # group users randomly
        intervals,hist=group_data(files,n1,n2,num_intervals,group_fraction,f)
        if len(intervals)==0:
            print "Warning: no data can be processed, exiting"
            exit()
        partial_hist.append(hist)
        ## execute analysis in parallel over the intervals
        part_fun=functools.partial(analysis_fct,
                                   compression_levels=compression_levels)
        if __name__ == '__main__':
            print "starting processes"
            pool=Pool()
            ans=pool.map(part_fun,intervals)
            pool.close()
            pool.join()
        else:
            ans=map(part_fun,intervals)
        dailys,errors,g_errors,corrs,g_corrs=zip(*ans) # the data for each interval
        errors,g_errors,corrs,g_corrs=(zip(*errors),zip(*g_errors),zip(*corrs),zip(*g_corrs)) # one element for each value of compression level, each containing one table for each repetition
        errors,g_errors,corrs,g_corrs=[[average_data(i) for i in e] for e in [errors,g_errors,corrs,g_corrs]] # one dataframe for each compression level
        for d in dailys:
            assert("COUNTER" in d)
        dailys=sum_data(dailys) # compact intervals into one dataframe
        dailys=aggregate_data(dailys) # aggregate data
        dailys['GLOBAL_ERROR']=compute_error(dailys)      # compute the global error
        dailys['GLOBAL_CORR']=compute_correlation(dailys)      # compute the global correlation
        # partial_data=sum_data([partial_data,dailys])
        partial_data=pd.concat([partial_data,dailys])
        if nlevs>1 and group_fraction>0: # aggregate all users that compressed at the same level together
            partial_errors,partial_group_errors,partial_corrs,partial_group_corrs=map(
                lambda partial,current: [p if d.empty else sum_data([p,d]) for (p,d) in zip(partial,current)],
                [partial_errors,partial_group_errors,partial_corrs,partial_group_corrs],
                [errors,g_errors,corrs,g_corrs])
    ### --- average across repetitions ---
    hist=[pd.DataFrame(np.asarray(a).T,columns=["size","count"]) for a in partial_hist]
    hist=pd.concat(hist).groupby(["size"],as_index=False).mean()
    hist.to_csv(os.path.join(data_dir,"group_size_hist_"+filename),sep=",",index=False)
    dailys=average_data([partial_data],counter=True,compute_stds=True)
    dailys.to_csv(os.path.join(data_dir,"daily_"+filename),sep=",",index=False)
    ## aggregate by day
    aggregated=aggregate_daily(dailys)
    aggregated.to_csv(os.path.join(data_dir,"aggregated_"+filename),sep=",",index=False)

    ## average local errors across repetitions
    if nlevs>1 and group_fraction>0:
        for tbl,i_col,o_col,prefix in [[partial_errors,'ERROR','Error','error'],
                                       [partial_group_errors,'GROUP_ERROR','Group_Error',"group_errors"],
                                       [partial_corrs,'CORR','Correlation',"corrs"],
                                       [partial_group_corrs,'GROUP_CORR','Group_Correlation',"group_corrs"]]:
            temp=[np.nan if d.empty else average_data([d]).mean()[i_col] for d in tbl]
            temp=pd.DataFrame(zip(compression_levels,temp),columns=['Compression',o_col])
            temp.to_csv(os.path.join(data_dir,prefix+"_"+filename),sep=",",index=False)

def update_compression_levels(old,last_std,max_compr=46,min_compr=4):
    stop=False
    timeout=1000                # number of moves to perform before giving up (the higher the standard deviation the more moves are needed to increase it)
    timeout_dest=100            # number of times to try finding an appropriate destination
    n_clusters=4                # number of cluster exchanged in each move
    n_flips=10                  # number of moves for every iteration
    compr_dict=dict(old[str(last_std)])
    ret=pd.DataFrame()
    t=0
    while not stop and t<timeout:
        print "loop "+str(t)
        t+=1
        keys=compr_dict.keys()
        # Compute the compression levels
        for _ in range(n_flips):
            np.random.shuffle(keys)
            try:
                src=next(i for i in keys if compr_dict[i]<=max_compr-n_clusters)
            except:
                stop=True      # stop here
            dest=None
            t_dest=0
            while t_dest<=timeout_dest and (not dest or compr_dict[dest]<=min_compr+n_clusters): # if the source has too few clusters
                t_dest+=1
                j=None
                while not j:
                    j=rescale_sample(np.random.normal(scale=0.35),range(old.shape[0])) # sample a row number from a normal distribution
                dest=old.index[j]           # the index does not get reshuffled
            # update dictionary
            if t_dest<=100:                  # successfully found a destination
                compr_dict[src]+=n_clusters    # add clusters (decompress)
                compr_dict[dest]-=n_clusters    # remove clusters (compress)
            else:
                stop=True      # stop here
        for k in compr_dict.keys(): # debugging
            assert(compr_dict[k]<=max_compr and compr_dict[k]>=min_compr)
        # check the standard deviation of the current configuration
        std=int(np.std(compr_dict.values()))
        if std>last_std:
            print "Successfully increased the standard deviation"
            last_std=std
            ret=pd.DataFrame.from_dict(compr_dict,orient="index",dtype=np.int64)
            ret=ret.rename(columns={0:str(std)})
            ret=pd.merge(ret,old,left_index=True,right_index=True) # add column to previous result
            stop=True
    return ret,last_std

def group_clustering_data(t,tbl):
    return list(tbl[[t]].groupby(by=t,as_index=False).apply(lambda x: list(x.index))) # group users according to their group id

def group_clustering_summ(t,k,tbl,std):
    ret=tbl[[str(std)]].copy()
    ret.iloc[np.random.permutation(len(ret))] # shuffle rows
    classifier=KMeans(n_clusters=k) # create classifier
    ret["labels"]=classifier.fit_predict(ret) # perform classification
    return group_clustering_data("labels",ret)

def group_randomly(t,k,tbl):
    ret=np.asarray(tbl.index)
    np.random.shuffle(ret)
    return [ret[i::k] for i in range(k)]

def exp_clustering_std(n,data_dir,files,num_groups,input_dir,start_compression=10,max_std=13):
    np.random.seed(n)
    np.random.shuffle(files)
    files=files[:500]
    groups_tbl,hist,user_data=cluster_users(files,num_groups) # user_data contains the data of each user in files
    #groups_tbl=groups_tbl.ix[:,:205]
    print "Done"
    compr_tbl=pd.DataFrame({"0":start_compression},index=files)
    new_std=0                       # init
    for std in range(max_std):
        print "Thread "+str(n)+" processing std "+str(std)
        if compr_tbl.empty:     # failed, stopping
            print "Stopping at std "+str(std)
            return compr_tbl
        elif std==new_std:      # execute body
            fcts=[["cl_data",functools.partial(group_clustering_data,tbl=groups_tbl)], # group according to similarity in their raw data
                  ["cl_sum",functools.partial(group_clustering_summ,tbl=compr_tbl,std=std,k=num_groups)], # group according to similarity in their compression level
                  ["cl_rnd",functools.partial(group_randomly,tbl=compr_tbl,k=num_groups)]]       # group randomly
            # fcts=[["cl_data",functools.partial(group_clustering_data,tbl=groups_tbl)]]
            for name,fct in fcts:
                print "Thread "+str(n)+" is executing function "+str(name)
                filename="exp_clustering_"+str(num_groups)+"_fct_"+str(name)+"_std_"+str(std)+"_rep_"+str(n)+".csv"
                dailys=pd.DataFrame()
                for i in range(len(groups_tbl.columns)): # for each epoch
                    t=groups_tbl.columns[i]
                    user_groups=fct(t)  # group users for this epoch
                    for g in range(len(user_groups)): # debug
                        tmp=[int(compr_tbl.loc[compr_tbl.index==j,str(std)]) for j in user_groups[g]]
                        print "Function "+str(name)+": group "+str(g)+" has avg "+str(np.mean(tmp))+" and std "+str(np.std(tmp))
                    ## select the data corresponding to the current epoch
                    current_measurements=[[user_data[u][separate_daily_obs(user_data[u]["TIME"])==t].copy() for u in g] for g in user_groups]
                    assert(len(current_measurements)<=num_groups)
                    if len(current_measurements)==0:
                        print "Warning: no data can be processed, skipping epoch "+str(t)
                    else:
                        part_fun=functools.partial(generate_daily_data_clustering,
                                                   cl_tbl=compr_tbl)
                        ans=map(part_fun,current_measurements)                       # process every group separatedly
                        epoch,errors,g_errors,corrs,g_corrs=zip(*ans) # the data for each group
                        epoch=[d for d in epoch if not d.empty] # remove empty tables
                        for d in epoch:
                            assert("COUNTER" in d)
                        epoch=sum_data(epoch) # compact intervals into one dataframe
                        epoch=aggregate_data(epoch) # aggregate data
                        epoch['GLOBAL_ERROR']=compute_error(epoch)      # compute the global error
                        epoch['GLOBAL_CORR']=compute_correlation(epoch)      # compute the global correlation
                        dailys=pd.concat([dailys,epoch])
                ## after processing all timesteps
                dailys.to_csv(os.path.join(data_dir,"daily_"+filename+".bz2"),sep=",",index=False,compression="bz2")
                ## aggregate by day
                aggregated=aggregate_daily(dailys)
                aggregated.to_csv(os.path.join(data_dir,"aggregated_"+filename+".bz2"),sep=",",index=False,compression="bz2")
            ## update the compression levels
            compr_tbl,new_std=update_compression_levels(compr_tbl,std) # update the compression levels for this iteration
        else:
            print "Warning: STD was not increased. Exiting"
            return compr_tbl
    print "Thread "+str(n)+" finished"
    return compr_tbl

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

# def exp_clustering(data_dir,files,num_groups,compression_level_fct,num_intervals,num_rep,input_dir):
#     filename="exp_clustering_"+str(num_groups)+".csv"
#     ### group the users
#     print "Grouping users in "+str(num_groups)+" clusters"
#     np.random.shuffle(files)
#     groups_tbl,hist,user_data=cluster_users(files,num_groups) # user_data contains the data of each user in files
#     print "Done"
#     # hist.to_csv(os.path.join(data_dir,"group_size_hist_"+filename),sep=",",index=False)
#     groups=[]
#     for t in groups_tbl.columns:
#         user_names=list(groups_tbl[[t]].groupby(by=t,as_index=False).apply(lambda x: list(x.index))) # group users according to their group id
#         ## select the data corresponding to the current epoch
#         current_measurements=[[user_data[u][separate_daily_obs(user_data[u]["TIME"])==t].copy() for u in g] for g in user_names]
#         assert(len(current_measurements)<=num_groups)
#         # ## compute intervals
#         # current_measurements=[current_measurements[i::num_intervals] for i in range(num_intervals)] # divide in intervals: random subsets of agents of equal size.
#         # current_measurements=[i for i in current_measurements if i] # remove empty lists, if num_intervals > len(files)
#         if len(current_measurements)==0:
#             print "Warning: no data can be processed, skipping epoch "+str(t)
#         else:
#             groups.append(current_measurements)
#     part_fun=functools.partial(exp_clustering_body,num_rep=num_rep,compression_level_fct=compression_level_fct)
#     if __name__ == '__main__':
#         print "starting processes"
#         pool=Pool()
#         ans=pool.map(part_fun,groups)
#         pool.close()
#         pool.join()
#     else:
#         ans=map(part_fun,groups)
#     dailys=pd.concat(ans)
#     ## after processing all timesteps
#     dailys.to_csv(os.path.join(data_dir,"daily_"+filename),sep=",",index=False)
#     ## aggregate by day
#     aggregated=aggregate_daily(dailys)
#     aggregated.to_csv(os.path.join(data_dir,"aggregated_"+filename),sep=",",index=False)

# def exp_clustering_body(groups,num_rep,compression_level_tbl):
#     """
#     Args:
#     groups: a list of groups containing dataframes with user data
#     num_rep: the number of repetitions

#     Returns:
#     A dataframe containing the average data for the current group
#     """
#     partial_data=pd.DataFrame()
#     partial_hist=[]
#     part_fun=functools.partial(generate_daily_data_clustering,
#                                cl_tbl=compression_level_tbl)
#     for r in range(num_rep):    # for each repetition
#         ans=map(part_fun,groups)                       # process every group separatedly
#         dailys,errors,g_errors,corrs,g_corrs=zip(*ans) # the data for each group
#         dailys=[d for d in dailys if not d.empty] # remove empty tables
#         for d in dailys:
#             assert("COUNTER" in d)
#         dailys=sum_data(dailys) # compact intervals into one dataframe
#         dailys=aggregate_data(dailys) # aggregate data
#         dailys['GLOBAL_ERROR']=compute_error(dailys)      # compute the global error
#         dailys['GLOBAL_CORR']=compute_correlation(dailys)      # compute the global correlation
#         partial_data=pd.concat([partial_data,dailys])
#     ### Compute stds
#     stds=pd.DataFrame()
#     for c in ["RAW","CENTROID","CENTROID_INDIVIDUAL","CORR","GROUP_CORR","ERROR","GROUP_ERROR"]:
#         try:
#             std=std_data([partial_data],column=c)
#             std=std.rename(columns={c:c+"_STD"})
#             if not stds.empty:
#                 stds=pd.merge(stds,std,on="TIME")
#             else:
#                 stds=std.copy()
#         except KeyError:
#             print "Warning: Column "+c+" not found"
#     ## aggregate data
#     partial_data=average_data([partial_data],True) # average the repetitions
#     partial_data=pd.merge(partial_data,stds,on="TIME") # combine the stds
#     return partial_data

def generate_daily_data_clustering(measurements,cl_tbl):
    """
    Randomly group the users and for each group:
    - recompress their raw data according to the given compression levels
    - average the compressed data
    // - compute the local error for both users from the same average compressed data
    - sum their data

    Args:
    measurements: A list containing user data.
    cl_tbl: A table containing the compression level for each file
    """
    loc_errors=[pd.DataFrame() for _ in range(len(measurements))]
    group_errors=[pd.DataFrame() for _ in range(len(measurements))]
    loc_corrs=[pd.DataFrame() for _ in range(len(measurements))]
    group_corrs=[pd.DataFrame() for _ in range(len(measurements))]
    data=pd.DataFrame()
    # for i in range(len(measurements)):
    #     m=measurements[i]
    levs=[int(cl_tbl.loc[cl_tbl.index==i.index.name].values[0][0]) for i in measurements]
        #user_data,errors,g_errors,corrs,g_corrs=process_group(m,levs)
        #n=len(levs)
    user_data,errors,g_errors,corrs,g_corrs=process_group(measurements,levs)
    if not user_data.empty:
        return (user_data,errors,g_errors,corrs,g_corrs)
    else:
        print "User data is empty"
        print str([[k.index.name for k in j] for j in measurements])
        return [pd.DataFrame()]*5

def test_nonuniform_groups_parallel_body(group_sizes,num_rep=1,min_user=3000,max_user=3500,compression_level=10):
    l=range(min_user,max_user+1)
    np.random.shuffle(l)
    spl=np.split(l,group_sizes)[:-1] # remove last empty element
    spl=[i.tolist() for i in spl]
    clevs=[[compression_level]*len(i) for i in spl]
    part_fun=functools.partial(body_parallel,
                               intervals=spl,
                               names=["asd"]*len(group_sizes),
                               compression_levels=clevs,
                               data_dir="./test",
                               num_rep=num_rep,
                               group_fraction=1)
    if __name__ == '__main__':
        print "starting processes"
        pool=Pool()
        ans=pool.map(part_fun,range(len(group_sizes)))
    else:
        ans=map(part_fun,range(len(group_sizes)))
    print "Aggregating data"
    dailys,errors,g_errors,corrs,g_corrs=np.asarray(ans).T
    try:
        tot_data=average_data(dailys,counter=True)
        tot_data['GLOBAL_ERROR']=compute_error(tot_data)      # compute the global error on the partial avg
        tot_data['GLOBAL_CORR']=compute_correlation(tot_data)      # compute the global error on the partial avg
    except DataFrameEmptyError , e:
        print "Error, dataframe is empty"
        print e.value
        tot_data=pd.DataFrame()
    #tot_data.to_csv(os.path.join(data_dir,"daily_"+filename),sep=",",index=False,compression="bz2")
    aggregated=aggregate_daily(tot_data)
    #aggregated.to_csv(os.path.join(data_dir,"aggregated_"+filename),sep=",",index=False,compression="bz2")
    return aggregated["ERROR"].mean(),aggregated["GLOBAL_ERROR"].mean(),aggregated["CORR"].mean(),aggregated["GLOBAL_CORR"].mean()

def test_nonuniform_groups():
    sizes=[[10,10,10],[9,10,11],[8,9,13],[7,9,14],[6,8,16],[6,7,17],[5,7,18],[4,6,20],[3,6,21],[2,5,23],[2,4,24]]
    sizes=[[10,10,10],[6,8,16],[4,6,20],[2,4,24]]
    sizes=[[s1,s1+s2,s1+s2+s3] for s1,s2,s3 in sizes]
    errors=[test_nonuniform_groups_parallel_body(g,num_rep=10) for g in sizes]
    np.savetxt("./nonunif_errors.csv",errors)
    fig, ax = plt.subplots()
    ax.plot(range(len(sizes)),errors)
    fig.savefig("test.pdf",format='pdf')
    plt.close()


## raws.csv
# compression level 20
# user 1044
###########################################################################
# ------------------------------ EXECUTION ------------------------------ #
###########################################################################

def count_user_data(data_dir):
    input_dir="./datasets/nrel_southern_nevada/rtcsnv_sorted_by_person/"
    files=select_files_nrel(input_dir)
    counts=[]
    for c in range(48):         # try different compression levels
        print "counting users for level "+str(c)
        temp=preprocess_data_nrel(files,[c]) # keep only users with enough entries
        counts.append(len(temp))
    data=pd.DataFrame({"COMPRESSION":range(48),"COUNT":counts})
    data.to_csv(os.path.join(data_dir,"data_counts.csv"),index=False)

main(args.data_dir[0],args.compression_levels, args.num_intervals, args.num_rep, args.min_user, args.max_user, args.group_fraction,args.grp_size_distr)
