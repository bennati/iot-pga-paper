import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from itertools import chain
import scipy.stats as stats
import copy

class FileEmptyError(Exception):
    """ Custom exception """
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class DataFrameEmptyError(Exception):
    """ Custom exception """
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

def truncated_power_law(a, m):
    "taken from https://stackoverflow.com/questions/24579269/sample-a-truncated-integer-power-law-in-python"
    x = np.arange(a, m, dtype='float')
    pmf = 1/x**a
    pmf /= pmf.sum()
    return stats.rv_discrete(values=(range(a, m), pmf))

def randint_power_law(n1,n2):
    "Returns an integer between n1 and n2 (excluded) sampled from a power law"
    d=truncated_power_law(a=n1, m=n2)
    return d.rvs(size=1)[0]

def randint(n1,n2):
    "Returns an integer between n1 and n2 (excluded) sampled from a uniform distribution"
    return np.random.randint(n1,n2)

def randint_step(n1,n2):
    """
    Returns an integer between n1 and n2 (excluded) sampled from a step-like distribution:
    with 50% of probability the sampled number is n1, with 50% is n2.
    """
    p=np.random.random()
    if p<=0.5:
        return n1
    else:
        return n2-1

def pearson_coeff_sample(c1,c2):
    """
    Compute the pearson coefficient for two samples

    Args:
    c1,c2: the samples, pandas column dataframes
    """
    if len(c1)==1:               # only one value
        return 0
    else:
        d_x=c1-c1.mean()
        d_y=c2-c2.mean()
        return float((d_x*d_y).sum()/(np.sqrt((d_x*d_x).sum())*np.sqrt((d_y*d_y).sum())))

def error_formula(raw,summarized):
    """
    Computes the error using the formula in the paper.

    Args:
    raw: The raw data, a vector (or pandas DataFrame).
    summarized: The summarized data, a vector (or pandas DataFrame).

    Returns: A vector containing the errors, of the same order as the input data.
    """
    ret=abs(raw-summarized)/abs(raw)
    ret[ret==np.inf]=abs(summarized)     # substitute values where r=0
    return ret

def error_formula_sMAPE(raw,summarized):
    """
    Computes the error using the symmetric mean absolute percentage error.

    Args:
    raw: The raw data, a vector (or pandas DataFrame).
    summarized: The summarized data, a vector (or pandas DataFrame).

    Returns: A vector containing the errors, of the same order as the input data.
    """
    return abs(raw-summarized)/(abs(raw)+abs(summarized))

def separate_daily_obs(timesteps):
    """
    Converts timesteps in day values.
    Timesteps have code XXX[01-48], the last two digits determine the 48 observations in a day.
    Remove the last two digits to obtain the observation number.

    Args:
    timesteps: A vector (or pandas DataFrame) containing the timesteps.

    Returns: A numpy vector containing the day codes
    """
    ### aggregate daily observations:
    return np.floor(timesteps/100.0) # remove the last two digits

def aggregate_daily(data):
    data['TIME']=separate_daily_obs(data['TIME']) # remove the last two digits
    if not data.empty:
        return data.groupby(["TIME"], as_index=False).mean()
    else:
        return pd.DataFrame()

def generate_user_filenames(indexes):
    """ Transforms a list of numbers into filenames """
    return ['user_'+str(i)+'.csv' for i in indexes]

def sum_data(data,counter=False):
    """
    Sum two datasets by the TIME index.

    Args:
    data: A list of Pandas DataFramess, all containing a column named TIME

    Kwargs:
    counter: If true adds a column called COUNTER initialized to 1, used for computing averages.

    Returns: A pandas DataFrame that is the sum of the arguments over the index TIME
    """
    if all([d.empty for d in data]):
        print "Warning, summarizing empty data"
        return pd.DataFrame()
    ret=pd.concat(data)
    if counter:
        ret['COUNTER']=1
    return ret.groupby(["TIME"], as_index=False).sum()

def std_data(data,column="ERROR"):
    """
    Computes the standard deviation of values in a list of databases

    Args:
    data: A list of Pandas DataFramess, all containing a column named TIME

    Kwargs:
    column: the identifier of the data

    Returns: A pandas DataFrame containing the std of the data, aggregated over time
    """
    if all([d.empty for d in data]):
        print "Warning, summarizing empty data"
        return pd.DataFrame()
    ret=pd.concat(data)
    return ret.groupby(["TIME"], as_index=False).agg({column:np.std})[["TIME",column]]

def average_data(data,counter=False,compute_stds=False,debug=False):
    """
    Averages the data

    Args:
    data: A list of pandas DataFrames

    Kwargs:
    counter: If true reset or create the counter column

    Returns: A pandas DataFrame that contains the average of the original data
    """
    ret=sum_data(data,counter=counter)                          # create a support column to count occurrences, needed for averaging
    if 'COUNTER' not in ret:
        if ret.empty:
            print "Warning, averaging empty data"
            return pd.DataFrame()
        else:
            print "Warning, missing column COUNTER"
            ret=sum_data(data,counter=True)                      # force creating the column COUNTER
    cols=ret.columns != "TIME"
    ret.ix[:,cols]=ret.ix[:,cols].div(ret.COUNTER, axis='index') # don't divide the TIME
    #assert all(ret['COUNTER']==1)
    #ret=ret.drop('COUNTER',1)                                # remove the extra column
    if compute_stds:
        stds=pd.DataFrame()
        for c in ["RAW","CENTROID","CENTROID_INDIVIDUAL","CORR","GROUP_CORR","ERROR","GROUP_ERROR"]:
            try:
                std=std_data(data,column=c)
                std=std.rename(columns={c:c+"_STD"})
                if not stds.empty:
                    stds=pd.merge(stds,std,on="TIME")
                else:
                    stds=std.copy()
            except KeyError:
                if debug:
                    print "Warning: Column "+c+" not found"
        ret=pd.merge(ret,stds,on="TIME")
    return ret

def aggregate_data(data):
    """
    Aggregates the data using the function specified in f

    Args:
    data: A pandas DataFrames

    Returns: A pandas DataFrame that contains the aggregated original data

    Raises: DataFrameEmptyError, if the input is an empty dataframe
    """
    ret=data.copy()
    if ret.empty:
        raise DataFrameEmptyError("Warning, aggregating empty data")
    if 'COUNTER' in ret:    # more than one datapoints, compute the average
        # if f=="mean":                                                   # normalize by counter
        #     cols=ret.columns != "TIME"
        # elif f=="sum":
        #     # else, we want to compute the sum. The data is a sum already so no action is needed.
        #     cols=[col for col in ret.columns if col not in ["TIME","RAW","CENTROID"]] # average only the error and counter
        # else:
        #     print "Warning, unknown aggregation function. Using default: mean."
        #     cols=ret.columns != "TIME"
        cols=ret.columns != "TIME"
        ret.ix[:,cols]=ret.ix[:,cols].div(ret.COUNTER, axis='index') # divide the chosen columns by the value in COUNTER
    else:
        print "Warning, could not find column COUNTER. Aggregation not performed."
    ret["COUNTER"]=1                                                 # reset the counter
    return ret

def select_files(input_dir,indexes=None):
    """
    Select the files in the folder that are present in the list.

    Args:
    input_dir: The folder in which to read the files.
    Indexes: A list containing the codes corresponding to users to be considered.

    Returns: A list of filenames that are both present in the directory and in the input list.
    """
    files=[f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if indexes:
        users=generate_user_filenames(indexes)
        files=list(set(files).intersection(users)) # select only files that are contained into valid users
    files=[os.path.join(input_dir,i) for i in files]
    return files

def summarize_data(filename,compression_level,update_error=True):
    """
    Summarize data.

    Args:
    filename: The path to the file containing the data to summarize, or a pandas dataframe.
    compression_level: The number of clusters to use for the compression

    Kwargs:
    update_error: If true, update the error column after summarizing. Defaults to True

    Returns: A pandas DataFrame containing the recompressed data.
    """
    if isinstance(filename,pd.DataFrame):
        data=filename
        if data.empty:
            print "Warning: summarizing empty dataset"
    elif isinstance(filename,str):
        data = pd.read_csv(filename, delimiter=',')
    else:
        print "warning: unrecognized input format in summarize_data. Exiting"
        exit(1)
    if len(data.shape)==1:
        raise FileEmptyError("Warning: compression "+str(compression_level)+", file "+str(f1)+" is empty")
    ## add a column containing the counters
    data['CENTROID']=compress_data(data,separate_daily_obs(data['TIME']),int(compression_level),colname='RAW') # regenerate the data
    if update_error:
        data['ERROR']=compute_error(data) # update local error
    return data

def merge_group_data(data,data_avg):
    """
    Updates a dataframe with the compressed data obtained by averaging it with another user's data.

    Args:
    data: The user's data.
    data_avg: The average data of this and another user.

    Returns: A pandas DataFrame that differs from the original data in the values of compressed data and local error.
    """
    ## Substitute the centroids with the newly computed values
    ret=data.rename(index=str,columns={"CENTROID":"CENTROID_INDIVIDUAL"})
    ret=pd.merge(ret,data_avg[['TIME','CENTROID']],on='TIME')
    # for i in ret['TIME']:
    #     assert(data[data['TIME'==i]]['RAW']==ret[ret['TIME'==i]]['RAW'])
    #     assert(data_avg[data_avg['TIME'==i]]['CENTROID']==ret[ret['TIME'==i]]['CENTROID'])
    if 'ERROR' in ret:
        ret=ret.drop("ERROR",1) # error is outdated
    ## Add columns with the measures
    ret=pd.concat([ret,pd.DataFrame({
        'ERROR':compute_error(ret),
        'CORR':compute_correlation(ret),
        'GROUP_ERROR':compute_error(ret,raw="CENTROID_INDIVIDUAL"),
        'GROUP_CORR':compute_correlation(ret,raw="CENTROID_INDIVIDUAL")
    })],axis=1)
    return ret

def compress_data(data,days,num_clusters,colname='CENTROID',max_iter=300,tol=1e-4):
    """
    In the NREL dataset, each user has a variable number of trips on each day. Days are already defined.
    Summarize the data of each day.

    Args:
    data: the dataset to summarize
    days: the groups to summarize, an array where each element corresponds to a row in data and each value correspond to a group. All data with the same value is summarized together
    num_clusters: the summarization level to use.

    Kwargs:
    colname: The column where to find the values to summarize (raw data). Defaults to CENTROID
    max_iter: parameter of kmeans
    tol: parameter of kmeans
    """
    ret=[]
    for t in np.unique(days): # all values of time
        raw=data[days==t][colname] # isolate entries that correspond to this day
        raw = np.asarray(raw)[np.newaxis].T # transpose the already-compressed data, needed for classification
        if len(raw)<num_clusters:
            print "Warning, not enough points ("+str(len(raw))+") at time "+str(t)
        try:
            ### create a new classifier to avoid bias from previous iteration
            classifier=KMeans(n_clusters=num_clusters,max_iter=max_iter,tol=tol) # create classifier
            classifier.fit(raw) # perform classification
            ret.extend(np.take(classifier.cluster_centers_,classifier.labels_)) # the cluster center corresponding to each point
        except ValueError:
            print "Warning, skipping day "+str(t)+" because has too few entries: "+str(len(raw))
            ret.extend(raw) # append the raw data (error = 0)
        except:
            print "Warning while processing day "+str(t)
    if len(ret)!=len(data):
        print "Warning summarize"
    ret=np.asarray(ret).astype(np.float64) # make it possible to perform math ops
    return ret

def compute_error(data,raw="RAW",summarized="CENTROID"):
    """ compute the error """
    err=error_formula_sMAPE(data[raw],data[summarized])
    ## cap errors to one
    err[err>1]=1
    return err # add the three columns containing raw sum, cluster sum and global error

def compute_correlation(data,raw="RAW",summarized="CENTROID"):
    """ compute the correlation
    The pearson correlation returns a value for two time series, so the correlation is computet between raw and summarized of each day.
    The result vector will contain the correlation values repeated as many times as the number of row for that day
    """
    assert(all(data["TIME"]==sorted(data["TIME"]))) # increasing order
    days=separate_daily_obs(data["TIME"])
    corr=[]
    for d in np.unique(days):
        temp=data[days==d]
        corr.extend([pearson_coeff_sample(temp[raw],temp[summarized])]*temp.shape[0])
    return corr # add the three columns containing raw sum, cluster sum and global error

# def group_randomly(values,n,fraction=1):
#     '''
#     Shuffle the array and reshape it to have rows of lenght n

#     Args:
#     values: the array to group
#     n: the group size.

#     Kwargs:
#     fraction: the fraction of the vector to group, if <1 it is considered to be in the range 0,1

#     Returns:
#     A list whose (initial fraction*values.size) elements are lists of length n (the rest are lists of lenght 1: not grouped)

#     Raises:
#     ValueError: if fraction is greater than 100
#     '''
#     if fraction>100 or fraction<0:
#         raise ValueError("Wrong percentage value")
#     if fraction >1 and fraction <=100:
#         fraction/=100
#     assert(n>0)
#     ret=np.asarray(values)
#     np.random.shuffle(ret)    # shuffle files
#     idx=len(ret)*fraction
#     spl=np.split(ret,[idx])     # split in two, the first contains the 'fraction' of the population
#     rest=len(spl[0])%n
#     if rest:
#         spl[0]=spl[0][:-rest]        # reduce to even lenght
#     spl[0]=spl[0].reshape((-1,n))       # group users in groups of size n
#     spl[1]=spl[1].reshape((-1,1))       # group users individually
#     ch=chain.from_iterable(spl)      # concat the arrays
#     ret=[a.tolist() for a in ch]    # convert to lists
#     return ret

def group_randomly(values,n1,n2,fraction=1,f=None):
    '''
    Shuffle the array and reshape it to have rows of lenght n

    Args:
    values: the array to group
    n1: the min group size.
    n2: the max group size.

    Kwargs:
    fraction: the fraction of the vector to group, if <1 it is considered to be in the range 0,1
    f: a function that generates random numbers, used to generate the group sizes. Default is none. If none, groups are of deterministic size = n2.

    Returns:
    A list whose (initial fraction*values.size) elements are lists of length n (the rest are lists of lenght 1: not grouped). A numpy histogram (two numpy arrays) indicating the unique group sizes and the correspondent occurrences.

    Raises:
    ValueError: if fraction is greater than 100
    '''
    if fraction>100 or fraction<0:
        raise ValueError("Wrong percentage value")
    assert(n1<=n2)
    assert(n1>0)
    if fraction >1 and fraction <=100:
        fraction/=100
    ret=np.asarray(values)
    np.random.shuffle(ret)    # shuffle files
    idx=int(len(ret)*fraction)
    if n1==n2 or not f:
        idx-=idx%n2        # reduce to a lenght that is divisible by n2
        spl=np.split(ret,[idx]) # split in two, the first contains the 'fraction' of the population
        spl[0]=spl[0].reshape((-1,n2))       # group users in groups of size n
        spl[1]=spl[1].reshape((-1,1))       # group users individually
        spl=chain.from_iterable(spl)      # concat the arrays
    else:
        idxs=[]
        s=0
        while s<idx-n2:
            n=f(n1,n2+1) # find a random group size
            s+=n
            idxs.append(s)
        idxs.extend(range(idx,len(ret))) # fill up with individuals
        spl=np.split(ret,idxs)           # split to the indexes
    ret=[a.tolist() for a in spl]    # convert to lists
    lens=[len(a) for a in ret]
    if int(np.version.version.split(".")[1])>=9: # >= 1.9
        lens=np.unique(lens,return_counts=True)
    else:
        if len(lens)>0:
            lens=zip(*stats.itemfreq(lens)) # compatibility for older numpy versions
        else:
            lens=[[],[]]
    return ret,lens

def cluster_users(files,k,max_iter=300,tol=1e-4,column="RAW"):
    """
    Cluster the users in k clusters, according to their data.
    Clustering differs every epoch (day)

    Args:
    files: the array to group
    k: the number of clusters

    Returns:
    a table including the label (cluster) of each user for each epoch, a histogram with the group sizes and a dictionary containing the data for each user.
    """
    #extract_name_from_file=lambda n: n.split("/")[-1].split(".")[0]
    extract_name_from_file=lambda n: n
    ids=[]
    user_data=pd.DataFrame()
    dic={}
    for f in range(len(files)):
        tbl=pd.read_csv(files[f])
        if not tbl.empty:
            tbl.index.name=files[f]
            identifier=extract_name_from_file(files[f])
            ids.append(identifier)
            dic.update({identifier:tbl}) # save the full data
            tbl=tbl[["TIME",column]]      # keep only the raw data, used by clustering
            tbl=tbl.rename(columns={column:identifier}) # rename with the username
            if user_data.empty:
                user_data=tbl.copy()
            else:
                user_data=pd.merge(user_data,tbl,on="TIME",how='outer')
        else:
            print "file "+str(files[f])+" is empty"
    user_data["TIME"]=separate_daily_obs(user_data['TIME'])
    days=np.unique(user_data["TIME"])
    # Create the return table, containing a value for each epoch and user
    ret=pd.DataFrame(index=ids)
    # perform the clustering
    for d in days:
        tmp=user_data[user_data["TIME"]==d]
        tmp.index=range(len(tmp)) # reindex to avoid problems with column names later on
        tmp=tmp.dropna(axis=1,how='any') # drop columns with any empty value, cannot cluster with NAs. TODO: see https://stackoverflow.com/questions/35611465/python-scikit-learn-clustering-with-missing-data
        tmp=tmp.drop("TIME",1).T # features in the columns, users in the rows
        ### create a new classifier to avoid bias from previous iteration
        ncols=tmp.shape[0]
        if ncols<=2:
            print "Warning: less than 2 users at epoch "+str(d)
        else:
            nclusters=k
            if ncols<nclusters:
                print "warning, less points than clusters. Reducing cluster number to "+str(ncols)
                nclusters=ncols
            classifier=KMeans(n_clusters=nclusters,max_iter=max_iter,tol=tol) # create classifier
            tmp[d]=classifier.fit_predict(tmp) # perform classification
            ret=pd.merge(ret,tmp[[d]],left_index=True,right_index=True) # join on index
    ## compute histogram: size of groups
    hist=pd.DataFrame()
    for t in ret.columns:
        hist=pd.concat([hist,ret.groupby(by=t,as_index=False).size()])
    if int(np.version.version.split(".")[1])>=9: # >= 1.9
        hist=np.unique(hist,return_counts=True)
    else:
        if len(hist)>0:
            hist=zip(*stats.itemfreq(hist)) # compatibility for older numpy versions
        else:
            hist=[[],[]]
    return ret,hist,dic

# def generate_daily_data(indexes,compression_levels,fraction=1.0,data_dir="./datasets/ecbt/run_48/output/daily/users/cluster48/"):
#     """
#     Randomly group the users and for each group:
#     - recompress their raw data according to the given compression levels
#     - average the compressed data
#     - compute the local error for both users from the same average compressed data
#     - compute the similarity measure for both users from the same average compressed data
#     - sum their data

#     Args:
#     indexes: A list containing user codes. If None, all users are kept.
#     compression_levels: A list containing the number of clusters to be used for recompression of each agent in the group. Its length determines the size of groups.

#     Kwargs:
#     fraction: the fraction of the population to group
#     data_dir: The directory where to find the data

#     Returns: A tuple (d,e). 'd' is the sum of the data of all users and 'e' is the sum of the local errors for each compression level.
#     """
#     n=len(compression_levels)
#     loc_errors=[pd.DataFrame() for _ in range(n)]
#     loc_sims=[pd.DataFrame() for _ in range(n)]
#     data=pd.DataFrame()
#     ### global error: computed across all users.
#     ### To avoid having to store all data and process it afterwards, compute sums while reading data
#     files=select_files(data_dir,indexes) # convert indexes to file names
#     files=group_randomly(files,n,fraction) # generate random groups of size n
#     for flist in files:
#         levs=compression_levels if len(flist)==n else [np.random.choice(compression_levels)] # if the group size is smaller than n (or individual), choose a value randomly
#         user_data,errors,similarities=process_group(flist,levs,data_dir) # perform the analysis
#         if not user_data.empty:
#             data=sum_data([user_data,data]) # sum the partial data. data already contains a counter, do not reset it
#         else:
#             print "User data is empty"
#         if len(flist)==n:       # we are grouping
#             loc_errors=[le if e.empty else sum_data([le,e]) for (le,e) in zip(loc_errors,errors)] # Save the errors separatedly for different compression levels
#             loc_sims=[ls if s.empty else sum_data([ls,s]) for (ls,s) in zip(loc_sims,similarities)] # Save the errors separatedly for different compression levels
#     return (data,loc_errors,loc_sims)

def generate_daily_data(files,compression_levels):
    """
    Randomly group the users and for each group:
    - recompress their raw data according to the given compression levels
    - average the compressed data
    - compute the local error for both users from the same average compressed data
    - sum their data

    Args:
    files: A list containing user data.
    compression_levels: A list containing the number of clusters to be used for recompression of each agent in the group. Its length determines the size of groups.


    Returns: A tuple (d,e,gc,gc). 'd' is the sum of the data of all users and 'e' is the sum of the local errors for each compression level, 'g' is the sum of the group errors for each compression level, 'c' is the sum of the correlations for each compression level, 'gc' is the sum of the group correlations for each compression level.

    e,g,c and gc are ordered according to the order of compression_levels. Each vector has the same lenght as compression_levels and it is positionally consistent with compression_levels:
    If compression_levels=[2,3,5], the first element of the results will contain data of agents compressing with level 2, the second with level 3 and the third with level 5
    If compression_levels contains duplicate values e.g. [2,3,3] the users compressing with the duplicate value will be positioned randomly.

    If the files are less in number than the compression levels, it might happen that some elements of the outputs remain empty
    """
    loc_errors=[pd.DataFrame() for _ in range(len(compression_levels))]
    group_errors=[pd.DataFrame() for _ in range(len(compression_levels))]
    loc_corrs=[pd.DataFrame() for _ in range(len(compression_levels))]
    group_corrs=[pd.DataFrame() for _ in range(len(compression_levels))]
    data=pd.DataFrame()
    ### global error: computed across all users.
    ### To avoid having to store all data and process it afterwards, compute sums while reading data
    for flist in files:
        clevs=copy.copy(compression_levels)
        levs=np.random.permutation(clevs)[:len(flist)] # choose randomly the compression levels, use all compression levels if the group is full sized
        user_data,errors,g_errors,corrs,g_corrs=process_group(flist,levs)
        if not user_data.empty:
            data=sum_data([user_data,data]) # data already contains a counter, do not reset it
        else:
            print "User data is empty"
        if len(flist)>1:       # we are grouping
            # reorder according to the original compression levels
            idxs=[]
            for a in levs:
                idx=np.where(np.in1d(clevs,a))[0][0] # the idexes corresponding to the position of the chosen levels in the initial array
                idxs.append(idx)
                clevs[idx]=np.nan # remove that element from list, to deal with duplicate values
            for i,e,ge,c,gc in zip(idxs,errors,g_errors,corrs,g_corrs): # update array
                loc_errors[i]=(e if loc_errors[i].empty else sum_data([loc_errors[i],e]))
                group_errors[i]=(ge if group_errors[i].empty else sum_data([group_errors[i],ge]))
                loc_corrs[i]=(c if loc_corrs[i].empty else sum_data([loc_corrs[i],c]))
                group_corrs[i]=(gc if group_corrs[i].empty else sum_data([group_corrs[i],gc]))
    return (data,loc_errors,group_errors,loc_corrs,group_corrs)

def process_group(flist,clevs):
    """
    Produce the summarized data of a user or a group of users.
    Read the raw data of the users in flist and summarize it with the values in clevs.

    Args:
    flist: a list of files to read
    clevs: a list of compression levels, each number corresponds positionally to a file. it must be of the same size of flist.

    Returns: tuple (data,error,group_error), data contains the sum of all users, including a column counter
    """
    n=len(clevs)
    assert(len(flist)==n)
    # if isinstance(flist[0],str):
    #     print str([str.split(f,"/")[-1:][0] for f in flist])+" "+str(clevs)
    # elif isinstance(flist[0],pd.DataFrame):
    #     print str([str.split(f.index.name,"/")[-1:][0] for f in flist])+" "+str(clevs)
    # else:
    #     print str(clevs)
    user_data=pd.DataFrame()
    loc_errors=[pd.DataFrame() for _ in range(n)]
    group_errors=[pd.DataFrame() for _ in range(n)]
    loc_corrs=[pd.DataFrame() for _ in range(n)]
    group_corrs=[pd.DataFrame() for _ in range(n)]
    try:
        if n>1:
            data=[summarize_data(f,c,update_error=False) for (f,c) in zip(flist,clevs)]
            ## Users might have missing data, merge always on the time to make sure data is consistent
            data_avg=average_data(data,counter=True) # average the compressed data, reset the counter as each row represents one record
            ## Substitute the centroids with the newly computed values
            data=[merge_group_data(d,data_avg) for d in data]
            loc_errors,group_errors,loc_corrs,group_corrs=map(
                lambda c: [sum_data([d[['TIME',c]]],counter=True)  # equivalent to data but with the column COUNTER
                           for d in data],# Save the errors separatedly for different compression levels
                ["ERROR","GROUP_ERROR","CORR","GROUP_CORR"])
            user_data=sum_data(data,counter=True) # data contains data from separate users, so reset the counter before summing it to the previous data
        else:
            # WARNING: the table does not include the columns containing the group measures, the counter might not be precise
            user_data=summarize_data(flist[0],clevs[0])
            user_data["COUNTER"]=1
        ## return the sum of the data
    except FileEmptyError , e:
        print "Warning, empty file in "+str(flist)
        print e.value
    except DataFrameEmptyError, e:
        print "Warning, empty dataframe in files "+str(flist)
        print e.value
    except:
        print "Warning, undefined error when processing files "+str(flist)
    finally:
        return (user_data,loc_errors,group_errors,loc_corrs,group_corrs)

def group_data(files,n1,n2,num_intervals,group_fraction=1,grp_fct=None):
    """
    reads the data corresponding to the users defined in the argument and groups them randomly.
    The users are first grouped, according to the grouping function f, then subdivided in intervals that can be processed in parallel.

    Args:
    files: a list with the files to be read
    n1: the minimum group size
    n2: the maximum group size
    num_intervals: the number of intervals to subdivide the data in, each interval can be processed in parallel.

    Kwargs:
    group_fraction: the fraction of users to group, e.g. 0.7 mean 70% of users will be grouped and 30% will be individuals.
    grp_fct: the grouping function. Defaults to None, which is deterministic equal to n2.

    Returns: the intervals, list of lists containing filenames, and the histogram with the frequency of group sizes
    """
    np.random.shuffle(files)
    files,hist=group_randomly(files,n1,n2,group_fraction,f=grp_fct)
    intervals=[files[i::num_intervals] for i in range(num_intervals)] # divide in intervals: random subsets of agents of equal size.
    intervals=[i for i in intervals if i]                             # remove empty lists, if num_intervals > len(files)
    return intervals,hist

######################################################################################################################
# -------------------------------------------------- NREL Dataset -------------------------------------------------- #
######################################################################################################################
def select_files_nrel(input_dir):
    """
    Select the files in the folder that are present in the list.

    Args:
    input_dir: The folder in which to read the files.

    Returns: A list of lists of filenames.
    """
    files=[os.path.join(f,"gps_trips.csv") for f in os.listdir(input_dir) if f.startswith('person_')] # directories
    files=[os.path.join(input_dir,i) for i in files]
    return files

def summarize_data_nrel(filename,compression_level,update_error=True):
    """
    Summarize data.

    Args:
    filename: The file containing the data to summarize.
    compression_level: The number of clusters to use for the compression

    Kwargs:
    update_error: If true, update the error column after summarizing. Defaults to True

    Returns: A pandas DataFrame containing the recompressed data.
    """
    data = pd.read_csv(filename, delimiter=',',usecols=["gpstravdaytripid","gpstravdayid","avg_speed_mph"])
    if len(data.shape)==1:
        raise FileEmptyError("Warning: compression "+str(compression_level)+", file "+str(f1)+" is empty")
    data=data.rename(columns={"gpstravdaytripid":"TIME","gpstravdayid":"DAY","avg_speed_mph":"RAW"})
    data['CENTROID']=compress_data(data,data["DAY"],int(compression_level),colname='RAW') # compress the data of each day
    if update_error:
        data['ERROR']=compute_error(data) # update local error
    data['TIME']=data['DAY']*100+data['TIME'] # update the TIME to be in the same format as the other dataset (DDDNN: D day, N measurement number)
    data=data.drop("DAY",1)
    return data

def generate_daily_data_nrel(files,compression_levels):
    """
    Randomly group the users and for each group:
    - recompress their raw data according to the given compression levels
    - average the compressed data
    - compute the local error for both users from the same average compressed data
    - sum their data

    Args:
    files: A list containing user data.
    compression_levels: A list containing the number of clusters to be used for recompression of each agent in the group. Its length determines the size of groups.

    Returns: A tuple (d,e,gc,gc). 'd' is the sum of the data of all users and 'e' is the sum of the local errors for each compression level, 'g' is the sum of the group errors for each compression level, 'c' is the sum of the correlations for each compression level, 'gc' is the sum of the group correlations for each compression level.

    e,g,c and gc are ordered according to the order of compression_levels. Each vector has the same lenght as compression_levels and it is positionally consistent with compression_levels:
    If compression_levels=[2,3,5], the first element of the results will contain data of agents compressing with level 2, the second with level 3 and the third with level 5
    If compression_levels contains duplicate values e.g. [2,3,3] the users compressing with the duplicate value will be positioned randomly.

    If the files are less in number than the compression levels, it might happen that some elements of the outputs remain empty
    """
    loc_errors=[pd.DataFrame() for _ in range(len(compression_levels))]
    group_errors=[pd.DataFrame() for _ in range(len(compression_levels))]
    loc_corrs=[pd.DataFrame() for _ in range(len(compression_levels))]
    group_corrs=[pd.DataFrame() for _ in range(len(compression_levels))]
    data=pd.DataFrame()
    ### global error: computed across all users.
    ### To avoid having to store all data and process it afterwards, compute sums while reading data
    for flist in files:
        clevs=copy.copy(compression_levels)
        levs=np.random.permutation(clevs)[:len(flist)] # choose randomly the compression levels, use all compression levels if the group is full sized
        user_data,errors,g_errors,corrs,g_corrs=process_group_nrel(flist,levs)
        if not user_data.empty:
            data=sum_data([user_data,data]) # data already contains a counter, do not reset it
        else:
            print "User data is empty"
        if len(flist)>1:       # we are grouping
            # reorder according to the original compression levels
            idxs=[]
            for a in levs:
                idx=np.where(np.in1d(clevs,a))[0][0] # the idexes corresponding to the position of the chosen levels in the initial array
                idxs.append(idx)
                clevs[idx]=np.nan # remove that element from list, to deal with duplicate values
            for i,e,ge,c,gc in zip(idxs,errors,g_errors,corrs,g_corrs): # update array
                loc_errors[i]=(e if loc_errors[i].empty else sum_data([loc_errors[i],e]))
                group_errors[i]=(ge if group_errors[i].empty else sum_data([group_errors[i],ge]))
                loc_corrs[i]=(c if loc_corrs[i].empty else sum_data([loc_corrs[i],c]))
                group_corrs[i]=(gc if group_corrs[i].empty else sum_data([group_corrs[i],gc]))
    return (data,loc_errors,group_errors,loc_corrs,group_corrs)

def process_group_nrel(flist,clevs):
    """
    Produce the summarized data of a user or a group of users.
    Read the raw data of the users in flist and summarize it with the values in clevs.

    Args:
    flist: a list of files to read
    clevs: a list of compression levels, each number corresponds positionally to a file. it must be of the same size of flist.

    Returns: tuple (data,error,group_error), data contains the sum of all users, including a column counter
    """
    n=len(clevs)
    assert(len(flist)==n)
    print str([str.split(f,"/")[-2:][0] for f in flist])+" "+str(clevs)
    user_data=pd.DataFrame()
    loc_errors=[pd.DataFrame() for _ in range(n)]
    group_errors=[pd.DataFrame() for _ in range(n)]
    loc_corrs=[pd.DataFrame() for _ in range(n)]
    group_corrs=[pd.DataFrame() for _ in range(n)]
    try:
        if n>1:
            data=[summarize_data_nrel(f,c,update_error=True) for (f,c) in zip(flist,clevs)]
            ## Users might have missing data, merge always on the time to make sure data is consistent
            data_avg=average_data(data,counter=True) # average the compressed data, reset the counter as each row represents one record
            ## Substitute the centroids with the newly computed values
            data=[merge_group_data(d,data_avg) for d in data]
            loc_errors,group_errors,loc_corrs,group_corrs=map(
                lambda c: [sum_data([d[['TIME',c]]],counter=True)  # equivalent to data but with the column COUNTER
                           for d in data],# Save the errors separatedly for different compression levels
                ["ERROR","GROUP_ERROR","CORR","GROUP_CORR"])
            user_data=sum_data(data,counter=True) # data contains data from separate users, so reset the counter before summing it to the previous data
        else:
            user_data=summarize_data_nrel(flist[0],clevs[0])
            user_data["COUNTER"]=1
        ## return the sum of the data
    except FileEmptyError , e:
        print e.value
    except DataFrameEmptyError, e:
        print e.value
    except e:
        print e.value
    finally:
        return (user_data,loc_errors,group_errors,loc_corrs,group_corrs)

def preprocess_data_nrel(files,compression_levels):
    """
    Subset the data and keep only users that have enough trips (more than the compression level) for each day
    """
    max_lev=max(compression_levels)
    ret=[]
    for i in files:
        temp=pd.read_csv(i)
        if "gpstravdayid" in temp.columns:
            nrows=[len(temp[temp["gpstravdayid"]==j]) for j in np.unique(temp["gpstravdayid"])] # number of entries for each day
            if min(nrows)>=max_lev: # there are enough entries
                ret.append(i)
        else:
            print "Could not find required data for user "+str(i)
    return ret
