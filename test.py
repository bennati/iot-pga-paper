def test():
# check that by not recompressing the raw values, the mean error is greater than when recompressing them

    err1=0
    err2=0
    err3=0
    err4=0
    err5=0
    min_user=1000
    max_user=7444
    users=np.random.randint(min_user,high=max_user,size=10)
    files=['user_'+str(i)+'.csv' for i in users]
    for f in files:
        # --------------------------------------------------------------------------------
        print f
        # 48 -> RAW -> 40
        # --------------------------------------------------------------------------------
        print "loading data with 40 clusters"
        compression_level=40
        data_dir="./run_"+str(compression_level)+"/output/daily/users/cluster"+str(compression_level)+"/"
        d_40 = pd.read_csv(os.path.join(data_dir,f), delimiter=',')
        # --------------------------------------------------------------------------------
        # 48 -> NEW -> 40
        # --------------------------------------------------------------------------------
        print "recompress with same compression level, to test that compression works"
        d_48_40 = pd.read_csv(os.path.join(data_dir,f), delimiter=',')
        d_48_40['CENTROID']=summarize_data(d_48_40,40,colname='RAW') # data with the target compression level
        d_48_40['ERROR']=compute_error(d_48_40['RAW'],d_48_40['CENTROID'])
        try:
            assert max(d_40['RAW']-d_48_40['RAW'])==0
            assert max(d_40['CENTROID']-d_48_40['CENTROID'])<10e-15 # same as in original data
            assert max(d_40['ERROR']-d_48_40['ERROR'])<10e-15
            print str(len(np.unique(d_40['CENTROID'][:48])))+" - "+str(len(np.unique(d_48_r['CENTROID'][:48])))
            print "everything ok"
        except:
            print "something went wrong"

        # --------------------------------------------------------------------------------
        # 48 -> NEW -> 40 -> NEW -> 20
        # --------------------------------------------------------------------------------
        print  "recompress 48 clusters to 40 and to 20 clusters"
        d_48_40_20 = pd.read_csv(os.path.join(data_dir,f), delimiter=',')
        d_48_40_20['CENTROID']=summarize_data(d_48_40_20,40,colname='RAW') # data with the target compression level
        d_48_40_20['CENTROID']=summarize_data(d_48_40_20,20,colname='CENTROID') # data with the target compression level
        d_48_40_20['ERROR']=compute_error(d_48_40_20['RAW'],d_48_40_20['CENTROID'])
        try:
            assert max(d_40['RAW']-d_48_40_20['RAW'])==0 # same original data
            print "everything ok"
        except:
            print "something went wrong"

        # --------------------------------------------------------------------------------
        # 48 -> OLD -> 40 -> NEW -> 20
        # --------------------------------------------------------------------------------
        print  "recompress 40 clusters to 20 clusters"
        d_40_20 = pd.read_csv(os.path.join(data_dir,f), delimiter=',')
        d_40_20['CENTROID']=summarize_data(d_40_20,20,colname='CENTROID') # data with the target compression level
        d_40_20['ERROR']=compute_error(d_40_20['RAW'],d_40_20['CENTROID'])
        try:
            assert max(d_40['RAW']-d_40_20['RAW'])==0 # same original data
            print "everything ok"
        except:
            print "something went wrong"

        # --------------------------------------------------------------------------------
        # 48 -> NEW -> 20
        # --------------------------------------------------------------------------------
        print  "recompress raw data to 20 clusters"
        d_48_20 = pd.read_csv(os.path.join(data_dir,f), delimiter=',')
        d_48_20['CENTROID']=summarize_data(d_48_20,20,colname='RAW') # data with the target compression level
        d_48_20['ERROR']=compute_error(d_48_20['RAW'],d_48_20['CENTROID'])
        try:
            assert max(d_40['RAW']-d_48_20['RAW'])==0 # same original data
            print "everything ok"
        except:
            print "something went wrong"

        # --------------------------------------------------------------------------------
        # 48 -> OLD -> 20
        # --------------------------------------------------------------------------------

        print "load precompressed data at 20 clusters"
        compression_level=20
        data_dir="./run_"+str(compression_level)+"/output/daily/users/cluster"+str(compression_level)+"/"
        d_20 = pd.read_csv(os.path.join(data_dir,f), delimiter=',')
        try:
            assert max(d_20['RAW']-d_40['RAW'])<10e-15 # same original data
            print "everything ok"
        except:
            print "something went wrong"

        # --------------------------------------------------------------------------------
        print  "48 -> OLD -> 20 vs. 48 -> NEW -> 20"
        try:
            assert max(d_20['RAW']-d_48_20['RAW'])==0
            assert max(d_20['CENTROID']-d_48_20['CENTROID'])<10e-15 # have the same centroids
            assert max(d_20['ERROR']-d_48_20['ERROR'])<10e-15
            print str(len(np.unique(d_20['CENTROID'][:48])))+" - "+str(len(np.unique(d_48_20['CENTROID'][:48])))
            print "everything ok"
        except:
            print "The two algorithms give different results"
            print "Mean error "+str((d_20['ERROR']-d_48_20['ERROR']).mean())

        # --------------------------------------------------------------------------------
        print  "48 -> OLD -> 20 vs. 48 -> OLD -> 40 -> NEW -> 20"
        try:
            assert max(d_20['RAW']-d_40_20['RAW'])==0
            assert max(d_20['CENTROID']-d_40_20['CENTROID'])<10e-15 # have the same centroids
            assert max(d_20['ERROR']-d_40_20['ERROR'])<10e-15
            print str(len(np.unique(d_20['CENTROID'][:48])))+" - "+str(len(np.unique(d_40_20['CENTROID'][:48])))
            print "everything ok"
        except:
            print "The two algorithms give different results"
            print "Mean error "+str((d_20['ERROR']-d_40_20['ERROR']).mean())

        # --------------------------------------------------------------------------------

        print "48 -> NEW -> 20 vs. 48 -> OLD -> 40 -> NEW -> 20"
        try:
            assert max(d_48_20['CENTROID']-d_40_20['CENTROID'])<10e-15 # same as in original data
            assert max(d_48_20['ERROR']-d_40_20['ERROR'])<10e-15
            print "everything ok"
        except:
            print "The two algorithms give different results"
            print "Mean error "+str((d_48_20['ERROR']-d_40_20['ERROR']).mean())

        # --------------------------------------------------------------------------------

        print  "48 -> NEW -> 40 -> NEW -> 20 vs. 48 -> OLD -> 40 -> NEW -> 20"
        try:
            assert max(d_48_40_20['RAW']-d_40_20['RAW'])==0
            assert max(d_48_40_20['CENTROID']-d_40_20['CENTROID'])<10e-15 # have the same centroids
            assert max(d_48_40_20['ERROR']-d_40_20['ERROR'])<10e-15
            print str(len(np.unique(d_48_40_20['CENTROID'][:48])))+" - "+str(len(np.unique(d_40_20['CENTROID'][:48])))
            print "everything ok"
        except:
            print "The two algorithms give different results"
            print "Mean error "+str((d_48_40_20['ERROR']-d_40_20['ERROR']).mean())

        # --------------------------------------------------------------------------------

        print  "48 -> NEW -> 40 -> NEW -> 20 vs. 48 -> NEW -> 20"
        try:
            assert max(d_48_40_20['RAW']-d_48_20['RAW'])==0
            assert max(d_48_40_20['CENTROID']-d_48_20['CENTROID'])<10e-15 # have the same centroids
            assert max(d_48_40_20['ERROR']-d_48_20['ERROR'])<10e-15
            print str(len(np.unique(d_48_40_20['CENTROID'][:48])))+" - "+str(len(np.unique(d_48_20['CENTROID'][:48])))
            print "everything ok"
        except:
            print "The two algorithms give different results"
            print "Mean error "+str((d_48_40_20['ERROR']-d_48_20['ERROR']).mean())

        # --------------------------------------------------------------------------------
        # save the errors
        err1+=(d_20['ERROR']-d_48_20['ERROR']).mean()
        err2+=(d_20['ERROR']-d_40_20['ERROR']).mean()
        err3+=(d_48_20['ERROR']-d_40_20['ERROR']).mean()
        err4+=(d_48_40_20['ERROR']-d_40_20['ERROR']).mean()
        err5+=(d_48_40_20['ERROR']-d_48_20['ERROR']).mean()

    print "printing the avg errors"
    print  "48 -> OLD -> 20 vs. 48 -> NEW -> 20: "+str(err1/float(len(files)))
    print  "48 -> OLD -> 20 vs. 48 -> OLD -> 40 -> NEW -> 20: "+str(err2/float(len(files)))
    print "48 -> NEW -> 20 vs. 48 -> OLD -> 40 -> NEW -> 20: "+str(err3/float(len(files)))
    print  "48 -> NEW -> 40 -> NEW -> 20 vs. 48 -> OLD -> 40 -> NEW -> 20: "+str(err4/float(len(files)))
    print  "48 -> NEW -> 40 -> NEW -> 20 vs. 48 -> NEW -> 20: "+str(err5/float(len(files)))
