import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from scipy import stats as sts
from itertools import tee, izip, combinations_with_replacement
from scipy import interpolate
import pandas as pd

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

def find_segment_containing(y,curve):
    if y<min(curve) or y>max(curve):
        return [False]*(len(curve)-1)
    else:
        return [True if np.sign(y-y1) != np.sign(y-y2) else False # at point 1 y is bigger than the curve and at point 2 y becomes smaller than the curve
                for (y1,y2) in pairwise(curve)]

def compute_intersection_segments(X1,X2,Y1,Y2,Y3,Y4):
    # line is y=A*x+b
    A1 = (Y1-Y2)/(X1-X2)
    A2 = (Y3-Y4)/(X1-X2)
    b1 = Y1-A1*X1
    b2 = Y3-A2*X1
    return (b2 - b1) / (A1 - A2) # intersection

def compute_intersection_segment_axis_x(X1,X2,Y1,Y2,X):
    A1 = (Y1-Y2)/(X1-X2)
    b1 = Y1-A1*X1
    return A1*X+b1

def compute_intersection_segment_axis_y(X1,X2,Y1,Y2,Y):
    return compute_intersection_segment_axis_x(Y1,Y2,X1,X2,Y)

# def compute_intersection_axis_x(x,xs,ys):
#     idx=find_segment_containing(x,xs).index(True)
#     return compute_intersection_segment_axis_x(xs[idx],xs[idx+1],ys[idx],ys[idx+1],x)

# def compute_intersection_axis_y(y,xs,ys):
#     idx=find_segment_containing(y,ys).index(True)
#     return compute_intersection_segment_axis_y(xs[idx],xs[idx+1],ys[idx],ys[idx+1],y)

def find_intersection_segments(xs,curve1,curve2):
    # we assume both curves have the same x axis
    # find the segment where they intersect
    intersections=[True if np.sign(r1-s1) != np.sign(r2-s2) else False # at point 1 r is bigger than s and at point 2 r becomes smaller than s
                   for ((r1,r2),(s1,s2)) in zip(pairwise(curve1),pairwise(curve2))] # ignore the first point of the curve (they generally always overlap at x=1
    indexes = [i for i,j in enumerate(intersections) if j == True] # find the indexes where the value is True

    return [compute_intersection_segments(xs[i],xs[i+1],curve1[i],curve1[i+1],curve2[i],curve2[i+1]) for i in indexes]

def coord_at_x(x,xs,ys):
    ret=None
    # tck = interpolate.splrep(xs, ys, s=1)
    # ans=interpolate.splev(x, tck, der=0)
    # if ans.size==1:         # could find a root
    #     ret=float(ans)
    # else:
    #     #fallback to segment intersection
    #     print "Warning, could not interpolate y value, falling back to segment method"
    ans=None
    try:
        idx=find_segment_containing(x,xs).index(True)
        ans=compute_intersection_segment_axis_x(xs[idx],xs[idx+1],ys[idx],ys[idx+1],x)
    except:
        None
        # print "Warning, segment method failed"
    if ans:
        ret=ans
    else:
        # print "Warning, could not finding intersections, trying comparing the points"
        ## check if the required point is close to one of the points in the line
        ans=[True if abs(i-x)<0.0001 else False for i in xs]
        if any(ans):
            return ys[ans.index(True)]
    return ret


def coord_at_y(y,xs,ys):
    "Returns a float identifying the x coordinate at which the curve 'ys' takes value 'y'"
    ## try with interpolating the spline
    ret=None
    # tck = interpolate.splrep(xs, np.subtract(ys,y), s=1)
    # ans=interpolate.sproot(tck)
    # if ans.size==1:         # could find a root
    #     ret=ans[0]
    # else:                   # fallback to segment intersection
        #print "Warning, could not interpolate x value, falling back to segment method"
    ans=None
    try:
        idx=find_segment_containing(y,ys).index(True)
        ans=compute_intersection_segment_axis_y(xs[idx],xs[idx+1],ys[idx],ys[idx+1],y)
    except:
        None
        # print "Warning, segment method failed"
    if ans:
        ret=ans
    else:
        # print "Warning, could not finding intersections, trying comparing the points"
        ## check if the required point is close to one of the points in the line
        ans=[True if abs(i-y)<0.0001 else False for i in ys]
        if any(ans):
            ret=xs[ans.index(True)]
    return ret


def find_intersection(xs,curve1,curve2):
    # we assume both curves have the same x axis
    ## check if the points of the curves are similar
    ret=find_intersection_segments(xs,curve1,curve2)
    if len(ret)==0:              # if nothing worked, try with interpolation
        print "Warning, could not interpolate intersections, trying with splines"
        # interpolate their difference
        tck=interpolate.splrep(xs, np.subtract(curve1,curve2), s=0) # don't use smoothing to avoid false positives when the two curves are almost parallel
        # find the roots of the spline
        ans=interpolate.sproot(tck)
        if ans.size!=0:         # could find a root
            ret.extend(ans.tolist())
        else:                   # fallback to segment intersection
            print "Warning, could not interpolate intersections, comparing the points"
            ans=[True if abs(i-j)<0.0001 else False for (i,j) in zip(curve1,curve2)][1:-1]  # ignore intersections at the extremes
            if any(ans):
                ret.extend([xs[i] for i,j in enumerate(ans) if j == True]) # find the indexes where the value is True
    return ret

def find_local_from_global(y,xs,g,l):
    "Compute the y value of 'l' that corresponds to the compression level where 'g' has value 'y'"
    intercept=coord_at_y(y,xs,g)
    if intercept:
        ## find the value of local error at the intersection points
        return coord_at_x(intercept,xs,l)
    else:
        print "Warning: couldn't find intersection"
        return np.nan

def find_global_from_local(y,xs,g,l):
    "Compute the y value of 'l' that corresponds to the compression level where 'g' has value 'y'"
    return find_local_from_global(y,xs,l,g)

def find_point_equivalence(xs,lerr,lerr_rec,gerr,gerr_rec,ymin,ymax):
    "Find the value of accuracy (Y) such that global error and local error are the same for both the normal and recompressed transmission"
    thresh=0.001
    res=None
    yavg=None
    ## reduce the interval by half until an intercept is found or the difference is lower than thresh
    while (ymax-ymin > thresh):
        ## find the x coordinates at which global error intersects the extremes
        y_l_min=find_local_from_global(ymin,xs,gerr,lerr)
        y_l_r_min=find_local_from_global(ymin,xs,gerr_rec,lerr_rec)
        y_l_max=find_local_from_global(ymax,xs,gerr,lerr)
        y_l_r_max=find_local_from_global(ymax,xs,gerr_rec,lerr_rec)

        ## find the value of the mean value
        yavg=(ymin+ymax)/2.0
        y_l_avg=find_local_from_global(yavg,xs,gerr,lerr)
        y_l_r_avg=find_local_from_global(yavg,xs,gerr_rec,lerr_rec)
        #print str(y_l_min)+" "+str(y_l_r_min)

        if abs(y_l_avg-y_l_r_avg)<thresh: # if the two values of local error are close enough
            res=float((y_l_avg+y_l_r_avg)/2.0) # return the average value
            break
        # restrict the domain and repeat
        if np.sign(y_l_max-y_l_r_max) != np.sign(y_l_avg-y_l_r_avg): # the two curves intersect between avg and max
            ymin=yavg           # use avg as new minimum
        elif np.sign(y_l_min-y_l_r_min) != np.sign(y_l_avg-y_l_r_avg): # the two curves intersect between avg and min
            ymax=yavg           # use avg as new maximum
        else:
            break
    return (yavg,res)

def find_point_equivalence_diff(xs,lerr,lerr_rec,gerr,gerr_rec,ys,diff):
    ret=[]
    idx=find_segment_containing(0,diff)
    idx=[i for (b,i) in zip(idx,range(len(idx))) if b and b!=np.nan] # convert to numeric indexes
    ret_gerr=[compute_intersection_segment_axis_y(ys[i],ys[i+1],diff[i],diff[i+1],0) for i in idx] # find the coord for which the difference is zero

    ## find the equivalent y coordinate for lerr
    if ret_gerr:
        ret_lerr=[find_local_from_global(y,xs,gerr,lerr) for y in ret_gerr]
        return [(g,l) for (g,l) in zip(ret_gerr,ret_lerr) if l and not np.isnan(l)] # remove nans
    else:
        return []

def find_points_equivalence(xs,lerr,lerr_rec,gerr,gerr_rec):
    "find as many equivalence points as there are"
    # find the value of local error for each point where the global errors are the same
    ys,diff=compute_method_efficiency(xs,lerr,lerr_rec,gerr,gerr_rec)
    return find_point_equivalence_diff(xs,lerr,lerr_rec,gerr,gerr_rec,ys,diff)

def compute_method_efficiency(xs,lerr,lerr_rec,gerr,gerr_rec):
    "Compute the difference in privacy between the normal method and the recompression"
    ymin=max(min(gerr),min(gerr_rec))
    ymax=min(max(gerr),max(gerr_rec))
    ## generate the new x axis taking the values that gerr and gerr_rec take
    ys=np.logspace(np.log10(ymin),np.log10(ymax),num=100).tolist()
    ys=[i for i in ys if i>=ymin and i<=ymax]
    ## add the already existing y values
    ys=ys+[i for i in gerr+gerr_rec if i>=ymin and i<=ymax]
    ys.sort()
    ## for each of the points generated find the corresponding x coordinate at which gerr takes that value
    curve=[find_local_from_global(y,xs,gerr,lerr) for y in ys]
    curve_rec=[find_local_from_global(y,xs,gerr_rec,lerr_rec) for y in ys]
    return (ys,np.subtract(curve_rec,curve))

def main_old(data_dir,plot_dir,percentage=100):
    ### plot aggregated data
    compr_levels=range(1,49)
    start_levels=range(5,49)
    heatmap_efficiency=np.zeros((49,49))
    loc_errors=[]
    gl_errors=[]
    delta_intersects=[]
    for start in start_levels:
        print "starting from "+str(start)
        ### read the datasets and store values of local and global error
        (local_error,global_error)=read_datasets(data_dir,compr_levels,percentage)
        compr_levels_rec=[x for x in compr_levels if x<=start] # all compression higher than the starting compression
        (local_error_rec,global_error_rec)=read_datasets(data_dir,compr_levels_rec,percentage,start)

        if compr_levels_rec and any(global_error) and any(global_error_rec) and any(local_error) and any(local_error_rec):
            print "Plotting start level "+str(start)
            ## shorten the data to the size of compr_levels_rec
            xs=compr_levels_rec
            lerr=local_error[:len(xs)]
            lerr_rec=local_error_rec[:len(xs)]
            gerr=global_error[:len(xs)]
            gerr_rec=global_error_rec[:len(xs)]
            ## save the current errors for later analysis
            loc_errors.append({'start':start,'xs':xs,'err':lerr_rec})
            gl_errors.append({'start':start,'xs':xs,'err':gerr_rec})

            ##################################################################################################################################
            # This plot compares the local and global errors of the standard transmission and transmission with recompression                #
            # A higher local error means higher privacy, a lower global error means better accuracy.                                         #
            # We are interested in the regime where there is an increase in privacy (local error) and an increase in accuracy (global error) #
            ##################################################################################################################################
            fig=plt.figure()
            fig.suptitle("Recompressing data with "+str(start)+" clusters")
            # ax = fig.add_subplot(1,1,1)
            ax = plt.subplot2grid((1, 10), (0, 0), colspan=9)
            ax.set_yscale('log')
            ax.invert_xaxis()
            ax.plot(xs,lerr,'b--')
            ax.plot(xs,gerr,'r--')
            ax.plot(xs,lerr_rec[:len(xs)],"b")
            ax.plot(xs,gerr_rec,"r")
            plt.xlim(xmax=1.0)
            plt.ylim(ymax=1.0,ymin=min(min(gerr),min(gerr_rec)))
            ax.legend(['Local','Global','Local Rec.','Global Rec.'],prop={'size':15},loc=2)

            ################################################################################################################################################
            # Plot vertical lines showing the intersection between the global error, with and without recompression, and the local error, with and without #
            # The intersection between the local errors "xl" denotes the point where the system keeps the same privacy as the simple transmission.         #
            # The intersection between the global errors "xg" denotes the point where the system keeps the same accuracy as the simple transmission        #
            # The area to the left of xg is the regime where the recompression bings an advantage at the system level, increasing accuracy                 #
            # The area to the right of xl is the regime where the system bring an advantage at the local level, increasong privacy                         #
            ################################################################################################################################################

            Xa=find_intersection(xs,lerr,lerr_rec)
            Xa=[i for i in Xa if i>=2 and i<=start-1]
            for i in Xa:
                ax.axvline(i,linewidth=1,color='b',ls="dashed")
                ax.text(i,1.2,round(i,2)) # write the x coordinate above the plot
                # intr=coord_at_x(i,xs,gerr)
                # xmin=1-i/plt.xlim()[0]
                # ax.axhline(y=intr,xmin=xmin, xmax=1,linewidth=1, color = 'g')
            Xb=find_intersection(xs,gerr,gerr_rec)
            Xb=[i for i in Xb if i>=2 and i<=start-1]
            for i in Xb:
                ax.axvline(i,linewidth=1,color='r',ls="dashed")
                ax.text(i,1.2,round(i,2)) # write the x coordinate above the plot

            ## compute and save the difference between the two coordinates
            if Xa and Xb:
                deltas=[i-j for (i,j) in zip(Xa,Xb)]
                deltas=np.mean(deltas)
                delta_intersects.append(deltas)
            else:
                delta_intersects.append(None)

            ## fill the heatmap: count combinations of recompressions that create the good regime (between Xa and Xb)
            if Xa and Xb:
                recompr=[x for x in xs if x>Xb[0] and x < Xa[0]]
                for i in recompr:
                    idx=xs.index(i)
                    heatmap_efficiency[i,start]+=lerr_rec[idx]-lerr[idx]+gerr[idx]-gerr_rec[idx] # add the difference between the two curves, we want the global error to be reduced and the local error to be increased
            elif not Xa and Xb:
                recompr=[x for x in xs if x>Xb[0]]
                for i in recompr:
                    idx=xs.index(i)
                    heatmap_efficiency[i,start]+=lerr_rec[idx]-lerr[idx]+gerr[idx]-gerr_rec[idx]

            ## find the point where the algorithm is indifferent: local and local_rec are equal, as well as global and global_rec
            ## plot horizontal line where at the boundaries where the system changes from efficient to unefficient
            # points_eq=find_points_equivalence(xs,lerr,lerr_rec,gerr,gerr_rec)
            # for (y_eq_g,y_eq_l) in points_eq:
            #     ax.axhline(y_eq_l,color="#99cccc",ls="dotted")
            #     ax.text(-0.5,y_eq_l,round(y_eq_l,2)) # write the y coordinate above the plot
            #     ax.axhline(y_eq_g,color="#BF4040",ls="dotted")
            #     ax.text(-0.5,y_eq_g,round(y_eq_g,2)) # write the y coordinate above the plot
            ## plot colored background to identity the regions of efficiency
            # boundaries=[min(max(gerr),max(gerr_rec))]
            # boundaries.extend([a[0] for a in points_eq]) # only for global error
            # boundaries.extend([max(min(gerr),min(gerr_rec))])
            # colors=[]
            # for avg in [(a+b)/2.0 for (a,b) in pairwise(boundaries)]: # skip the first value that is above the curves
            #     y_l=find_local_from_global(x_g,xs,gerr,lerr)
            #     y_l_r=find_local_from_global(x_g_r,xs,gerr_rec,lerr_rec)
            #     colors.extend(["yellow" if np.sign(y_l_r-y_l)>0 else "grey"])
            # for ((e,s),c) in zip(pairwise(boundaries),colors):
            #     ax.axhspan(s,e,facecolor=c,alpha=0.3)

            #########################################################################################################
            # The side graph shows the gain in privacy that the system brings for any accuracy requirement.         #
            # Given an accuracy requirement y, what privacy does the user gain by turning on the system?            #
            # For each y take x and x', the coordinates at which the global errors take that y value.               #
            # Get the value that the local errors take at the respective x coordinate and compute their difference. #
            #########################################################################################################
            try:
                (ys_new,perf)=compute_method_efficiency(xs,lerr,lerr_rec,gerr,gerr_rec)
                ax2 = plt.subplot2grid((1, 10), (0, 9))
                plt.ylim(ymax=1.0,ymin=min(min(gerr),min(gerr_rec)))
                plt.xlim(xmin=min(perf),xmax=max(perf)+0.01)
                # hide labels
                ax2.axes.xaxis.set_ticklabels([])
                ax2.set_yscale('log')
                ax2.axes.yaxis.set_ticklabels([])
                ax2.plot(perf,ys_new,color='k')
                ## draw line at 0
                ax2.axvline(0,ls='dashed')
                ## draw horizontal lines at intersection points
                for i in Xa:
                    intr=coord_at_x(i,xs,gerr)
                    xmax=coord_at_x(intr,ys_new,perf)
                    ax2.axhline(y=intr,xmin=0, xmax=xmax/plt.xlim()[1] if xmax else 0,
                                linewidth=1, color = 'k',ls='dashed')
                    intr=coord_at_x(i,xs,gerr_rec)
                    xmax=coord_at_x(intr,ys_new,perf)
                    ax2.axhline(y=intr,xmin=0, xmax=xmax/plt.xlim()[1] if xmax else 0,
                                linewidth=1, color = 'k',ls='dashed')
                for i in Xb:
                    intr=coord_at_x(i,xs,gerr)
                    xmax=coord_at_x(intr,ys_new,perf)
                    ax2.axhline(y=intr,xmin=0, xmax=xmax/plt.xlim()[1] if xmax else 0,
                                linewidth=1, color = 'k',ls='dashed')
                ## draw horizontal lines where the performance goes to 0
                points_eq=find_point_equivalence_diff(xs,lerr,lerr_rec,gerr,gerr_rec,ys_new,perf)
                for (y_eq_g,y_eq_l) in points_eq:
                    ax2.axhline(y_eq_g,color="#BF4040",ls="dotted")
                    ax2.text(max(perf)+0.02,y_eq_g,round(y_eq_g,2)) # write the y coordinate above the plot
            except:
                print "failed plotting the side graph"

            fig.savefig(os.path.join(plot_dir,"plot_start_"+str(start)+"_percent_"+str(percentage)+".pdf"),format='pdf')
            plt.close(fig)

            ## plot the relation between the difference in local and global error
            plot_errors_diff(os.path.join(plot_dir,"diffs_start_"+str(start)+"_percent_"+str(percentage)+".pdf"),
                             start,percentage,xs,gerr,gerr_rec,lerr,lerr_rec)


    ## end for

    ## save the curves of local and global errors
    curves_dir="./curves/"
    if not os.path.exists(curves_dir):
        os.makedirs(curves_dir)
    for start in start_levels:
        res=[i['err'] for i in loc_errors if i['start']==start]
        res=res+[i['err'] for i in gl_errors if i['start']==start]
        np.savetxt(os.path.join(curves_dir,"curves_"+str(start)+"_percent_"+str(percentage)+".csv"),res)

    ## plot the heatmap
    plot_heatmap(heatmap_efficiency,os.path.join(plot_dir,"heatmap.pdf"))

    ## plot the difference between the intersections
    fig=plt.figure()
    fig.suptitle("Delta intersections")
    ax = fig.add_subplot(1,1,1)
    try:
        ax.plot(start_levels,delta_intersects)
        ax.plot(start_levels,[d/s if d else None for (d,s) in zip(delta_intersects,start_levels)],color='r')
        ax.legend(['Absolute','Relative'],prop={'size':15},loc=2)
        fig.savefig(os.path.join(plot_dir,"delta_intersects"+"_percent_"+str(percentage)+".pdf"),format='pdf')
        plt.close(fig)
    except:
        plt.close(fig)

    ########################################################################################################################################################################
    # This plot shows what combinations of starting points and recompressions will give results similar to the privacy of the standard method at a given compression level #
    # Since the compression values are discrete, we plot discrete values, we consider them equal if they differ less than 5% from the initial value                        #
    ########################################################################################################################################################################
    for y in lerr[1:-1]:
        std_compression=lerr.index(y)
        disc_pairs=[[(l['start'],l['err'].index(i)) # return the index at which the value is found
                for i in l['err'] # loop across the values of local error
                if abs(i-y)<=i*0.05] # keep it only if it within 5% of the original value
               for l in loc_errors]
        disc_pairs=[i for i in disc_pairs if i] # remove empty lists
        pairs=[[(l['start'],coord_at_y(y,l['xs'],l['err']))] # return the index at which the value is found
               for l in loc_errors]
        pairs=[i for i in pairs if i] # remove empty lists
        if pairs and disc_pairs:
            pairs=[i[0] for i in pairs]   # flatten
            pairs=np.asarray(pairs).T     # first vector contains starting points, second vector contains recompressions
            disc_pairs=[i[0] for i in disc_pairs]   # flatten
            disc_pairs=np.asarray(disc_pairs).T     # first vector contains starting points, second vector contains recompressions
            fig=plt.figure()
            fig.suptitle("Performance at "+str(std_compression))
            plt.xlabel("Starting compression")
            plt.ylabel("Recompression value")
            ax = fig.add_subplot(1,1,1)
            ax.scatter(disc_pairs[0],disc_pairs[1],color="grey")
            ax.plot(pairs[0],pairs[1],color="black")
            fig.savefig(os.path.join(plot_dir,"equiv_"+str(std_compression)+"_percent_"+str(percentage)+".pdf"),format='pdf')
            plt.close(fig)


    ###############################################################################################################################################
    # benefit a the individual level:                                                                                                             #
    # for different values of global error (requirement) show the privacy of the normal method and the privacy for each pair start-recompression. #
    # 3D graph for each accuracy value, x=start, y=recompression z=privacy (is triangolar) + line on the empty side showing the standard privacy  #
    ###############################################################################################################################################

    plot_privacy_heatmap(plot_dir,"heatmap_priv",compr_levels,global_error,local_error,gl_errors,loc_errors)
    plot_accuracy_heatmap(plot_dir,"heatmap_acc",compr_levels,global_error,local_error,gl_errors,loc_errors)
