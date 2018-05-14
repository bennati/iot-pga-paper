import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import argparse
import os
from scipy import stats as sts
from itertools import tee, izip, combinations_with_replacement
from scipy import interpolate
import pandas as pd
from plot_tools import *
import itertools

#set math mode font to default font
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

min_compr=1
max_compr=10

def read_file(filename):
    local_error=None
    stds=None
    global_error=None
    try:
        data=pd.read_csv(filename,delimiter=',')
        local_error=data.mean()['ERROR']
        stds=data.mean()['ERROR_STD']
        global_error=data.mean()['GLOBAL_ERROR']
    except:
        print "Warning, file "+str(filename)+" not found"
    return (local_error,stds,global_error)

def plot_errors_diff_vert(start,percentage,xs,gerr,gerr_rec,lerr,lerr_rec):
    "Computes the difference between the two local errors and the two global errors and plots them side by side"
    fig=plt.figure()
    fig.suptitle("Difference in local and global error, start "+str(start))
    # global error
    ax = fig.add_subplot(121)
    ax.set_ylabel("Recompression")
    ax.set_xlabel("Global difference")
    ax.axvline(0,color="grey",ls='dashed')
    gerr_diff=np.subtract(gerr_rec,gerr)
    ax.plot(gerr_diff,xs)
    # local error
    ax2 = fig.add_subplot(122)
    ax2.set_ylabel("Recompression")
    ax2.set_xlabel("Local difference")
    ax2.axvline(0,color="grey",ls='dashed')
    lerr_diff=np.subtract(lerr_rec,lerr)
    ax2.plot(lerr_diff,xs)
    fig.savefig(os.path.join(plot_dir,"diffs_start_"+str(start)+"_percent_"+str(percentage)+".pdf"),format='pdf')
    plt.close(fig)

def plot_errors_diff(name,start,percentage,xs,gerr,gerr_rec,lerr,lerr_rec):
    "Computes the difference between the two local errors and the two global errors and plots them side by side"
    ######################################################################################################################################
    # Given a starting compression, how much privacy does one gain by recompressing the data?                                            #
    # For each possible recompression, plot the difference between the global errors (negative is good: more accuracy),                  #
    # and the difference between the local errors (positive is good: more privacy). Finally plot the difference of these two quantities. #
    # A positive value means that by recompressing at that level the system gains in privacy and accuracy                                #
    ######################################################################################################################################
    fig=plt.figure()
    fig.suptitle("Difference in local and global error, start "+str(start))
    # global error
    ax = plt.subplot2grid((9, 1), (0, 0), rowspan=3)
    ax.set_xlabel("")
    ax.set_ylabel("Global difference")
    ax.axhline(0,color="grey",ls='dashed')
    ax.invert_yaxis()           # we are interested at the negative region
    ax.axes.get_xaxis().set_ticklabels([])
    gerr_diff=np.subtract(gerr_rec,gerr)
    ax.axvline(xs[np.argmin(gerr_diff)],color="grey",ls='dashed') # the point where it in minimal
    ax.plot(xs,gerr_diff)
    # local error
    ax2 = plt.subplot2grid((9, 1), (3, 0), rowspan=3)
    ax2.set_xlabel("Recompression")
    ax2.set_ylabel("Local difference")
    ax2.axhline(0,color="grey",ls='dashed')
    ax2.axes.get_xaxis().set_ticklabels([])
    lerr_diff=np.subtract(lerr_rec,lerr)
    ax2.plot(xs,lerr_diff)
    # sum
    ax3 = plt.subplot2grid((9, 1), (6, 0), rowspan=3)
    ax3.set_xlabel("Recompression")
    ax3.set_ylabel("Sum of differences")
    ax3.axhline(0,color="grey",ls='dashed')
    ax3.plot(xs,lerr_diff-gerr_diff)
    fig.savefig(name,format='pdf')
    plt.close(fig)

def plot_heatmap(matrix,name,**args):
    "Plot a heatmap from a matrix"
    fig=plt.figure()
    fig.suptitle("Difference in local and global errors")
    plt.imshow(matrix,interpolation='none',**args)
    plt.ylabel("Recompression")
    plt.xlabel("Start")
    plt.gca().invert_yaxis()
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_ticks_position('both')
    cbaxes = fig.add_axes([0.08, 0.1, 0.03, 0.8]) # position colorbar on the left
    plt.colorbar(cax=cbaxes)
    fig.savefig(name,format='pdf')
    plt.close(fig)

def plot_hmap(heatmap,title,filename,plot_dir,xlab="compr2",xlab2="",ylab="compr",xlim=None,ylim=None,ticks=None,ticklabs=None,scale_percent=False,ticklabs_2x=None,ticklabs_2y=None,show_contour=False,font_size=16,cmap=None,num_decimals_legend=2,display_text=False):
    fig,ax=plt.subplots()
    fig.suptitle(title,fontsize=font_size)
    masked_array = np.ma.array (heatmap, mask=np.isnan(heatmap))
    if cmap==None:
        cmap = matplotlib.cm.jet
    cmap.set_bad('white',1.)
    if scale_percent:
        plt.imshow(masked_array, interpolation='nearest', cmap=cmap,vmin=0,vmax=1)
        cbar=plt.colorbar()
        t=np.arange(0,1.01,0.2)
        cbar.set_ticks(t)
        cbar.set_ticklabels([str(int(i*100))+"%" for i in t])
        cbar.ax.tick_params(labelsize=font_size)
    else:
        plt.imshow(masked_array, interpolation='nearest', cmap=cmap, aspect='auto')
        vmax=round(np.max(masked_array),num_decimals_legend)
        vmin=round(np.min(masked_array),num_decimals_legend)
        term=False
        step=10**(-num_decimals_legend)
        while not term:
            cbar_ticks=np.arange(vmin,vmax+step,step)
            if len(cbar_ticks)<10:
                term=True
            else:
                step*=2
        # cbar=plt.colorbar(format="%."+str(num_decimals_legend)+"f")
        cbar=plt.colorbar()
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticks)
        cbar.ax.tick_params(labelsize=font_size)
    if show_contour:
        CS = plt.contour(masked_array,colors='k')
        plt.clabel(CS, inline=1, fontsize=font_size)
    ax.set_ylabel(ylab,fontsize=font_size)
    ax.set_xlabel(xlab,fontsize=font_size)
    plt.gca().invert_yaxis()
    if not ticks==None:
        if len(ticks)==2:
            xticks=ticks[0]
            yticks=ticks[1]
        else:
            xticks=ticks
            yticks=ticks
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
    if not ticklabs==None:
        if len(ticklabs)==2:
            xticks_l=ticklabs[0]
            yticks_l=ticklabs[1]
        else:
            xticks_l=ticklabs
            yticks_l=ticklabs
        ax.set_xticklabels(xticks_l)
        ax.set_yticklabels(yticks_l)
    ax.tick_params(labelsize=font_size)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if not ticklabs_2x==None:
        ax2=ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(xticks)
        ax2.set_xlabel(xlab2,fontsize=font_size)
        ax2.tick_params(labelsize=font_size)
        ax2.set_xticklabels(ticklabs_2x)
    if not ticklabs_2y==None:
        ax2y=ax.twinx()
        ax2y.set_ylim(ax.get_ylim())
        ax2y.set_yticks(yticks)
        plt.setp(ax2y.yaxis.get_majorticklabels(),rotation=-90)
        ax2y.set_yticklabels(ticklabs_2y)
        #ax2y.tick_params(labelsize=16)
    if display_text:
        for j,i in apply(itertools.product,[range(x) for x in heatmap.shape]): # every cell in the matrix
            if (not xlim or (i>=xlim[0] and i<=xlim[1])) and (not ylim or (j>=ylim[0] and j<=ylim[1])):
                ax.text(i,j,round(heatmap[j,i],3), va='center', ha='center',color='y')
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.savefig(os.path.join(plot_dir,filename),format='pdf')
    plt.close(fig)

def fill_matrix(data_dir,var,name,compr_levels,compr3=None,abs_values=False):
    if abs_values:
        f=lambda x,y: abs(x-y)
    else:
        f=lambda x,y: x-y
    heatmap_d12=np.empty((max(compr_levels)-min(compr_levels)+1,max(compr_levels)-min(compr_levels)+1))
    heatmap_d12[:]=np.nan
    heatmap_u1=np.empty((max(compr_levels)-min(compr_levels)+1,max(compr_levels)-min(compr_levels)+1))
    heatmap_u1[:]=np.nan
    heatmap_u2=np.empty((max(compr_levels)-min(compr_levels)+1,max(compr_levels)-min(compr_levels)+1))
    heatmap_u2[:]=np.nan
    if compr3:
        heatmap_u3=np.empty((max(compr_levels)-min(compr_levels)+1,max(compr_levels)-min(compr_levels)+1))
        heatmap_u3[:]=np.nan
        heatmap_d13=np.empty((max(compr_levels)-min(compr_levels)+1,max(compr_levels)-min(compr_levels)+1))
        heatmap_d13[:]=np.nan
        heatmap_d23=np.empty((max(compr_levels)-min(compr_levels)+1,max(compr_levels)-min(compr_levels)+1))
        heatmap_d23[:]=np.nan
    else:
        heatmap_u3=None
        heatmap_d13=None
        heatmap_d23=None
    for compr in compr_levels:
        for compr2 in compr_levels:
            try:
                data=pd.read_csv(os.path.join(data_dir,name+"_"+str(compr)+"_"+str(compr2)+("_"+str(compr3) if compr3 else "")+".csv.gz"),delimiter=",",index_col="Compression")
                print "processing compression "+str(compr)+" "+str(compr2)
                if not data.empty:
                    if compr==compr2:
                        d=np.asarray(data[var])
                        heatmap_d12[compr,compr2]=f(d[0],d[1])
                        heatmap_u1[compr,compr2]=d[0]
                        heatmap_u2[compr,compr2]=d[1]
                        if compr3:
                            heatmap_d13[compr,compr2]=f(d[0],d[2])
                            heatmap_d23[compr,compr2]=f(d[1],d[2])
                            heatmap_u3[compr,compr2]=d[2]
                    else:
                        heatmap_d12[compr,compr2]=f(data.ix[compr],data.ix[compr2])
                        heatmap_u1[compr,compr2]=data.ix[compr]
                        heatmap_u2[compr,compr2]=data.ix[compr2]
                        if compr3:
                            heatmap_u3[compr,compr2]=data.ix[compr3]
                            heatmap_d13[compr,compr2]=f(data.ix[compr],data.ix[compr3])
                            heatmap_d23[compr,compr2]=f(data.ix[compr2],data.ix[compr3])

            except:
                print "skip"
    return heatmap_d12,heatmap_u1,heatmap_u2,heatmap_u3,heatmap_d13,heatmap_d23

def read_individual_errors(data_dir,levels,prefix="aggregated_"):
    lerr=[]
    stds=[]
    gerr=[]
    corr=[]
    for l in levels:
        try:
            data=pd.read_csv(os.path.join(data_dir,prefix+str(l)+".csv.gz"),delimiter=",")
            lerr.append(data["ERROR"].mean())
            stds.append(data["ERROR_STD"].mean())
            gerr.append(data["GLOBAL_ERROR"].mean())
            #corr.append(data["GLOBAL_CORR"].mean())
        except:
            print "skipping "+str(l)
            lerr.append(np.nan)
            stds.append(np.nan)
            gerr.append(np.nan)
            #corr.append(np.nan)
    return lerr,stds,gerr,corr

def plot_individual_errors(data_dirs,plot_dir,counters_loc=False,legend_labs=None,legend_styles=["solid"],nlevels=48,title=""):
    levels=range(nlevels)
    try:
        os.mkdir(plot_dir)
    except OSError:
        print("Directory already exists")
    if counters_loc and os.path.isfile(counters_loc):
        counts=pd.read_csv(counters_loc)
        count_labs=[int(counts[counts["COMPRESSION"]==i]["COUNT"]) for i in levels]
    fig,ax=plt.subplots(figsize=(10,5))
    fig.suptitle(title)
    ax.set_xlabel("Summarization level",fontsize=16)
    xticks=[l for l in levels if l%5==0]
    ax.set_xticks(xticks)
    ax.set_xticklabels(["1/"+str(l) for l in xticks])
    ax.set_ylabel("Error",fontsize=16)
    ax.set_yscale("log", nonposy='clip')
    plt.gca().invert_xaxis()
    for d in range(len(data_dirs)):
        lerr,stds,gerr,corr=read_individual_errors(data_dirs[d],levels)
        if legend_labs:
            ax.plot(levels,lerr,color="b",label=legend_labs[d]+" Local Group Error",linestyle=legend_styles[d])
            ax.plot(levels,gerr,color="r",label=legend_labs[d]+" Global Error",linestyle=legend_styles[d])
            #ax.plot(levels,corr,color="g",label="Global Correlation "+legend_labs[d],linestyle=legend_styles[d])
        else:
            ax.plot(levels,lerr,color="b",label="Local Group Error",linestyle=legend_styles[0])
            ax.plot(levels,gerr,color="r",label="Global error",linestyle=legend_styles[0])
            #ax.plot(levels,corr,color="g",label="Global Correlation",linestyle=legend_styles[0])
    ax.fill_between(levels,np.asarray(lerr)-np.asarray(stds),np.asarray(lerr)+np.asarray(stds),alpha=0.2,linestyle="-",facecolor="b")
    ax.legend(loc=4,fontsize=16)
    ## add user counts
    if counters_loc:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(levels)
        plt.setp(ax2.xaxis.get_majorticklabels(),rotation=70)
        ax2.set_xticklabels(count_labs)
    ax.tick_params(labelsize=16)
    plt.tight_layout(pad=1.5)
    fig.savefig(os.path.join(plot_dir,"individuals.pdf"),format='pdf')
    plt.close(fig)

def plot_group_heatmaps(min_compr,max_compr,compr3,cmap=None,font_size=None,title=""):
    compr_levels=range(min_compr,max_compr+1)
    heatmap_gerr=np.empty((max_compr+1,max_compr+1))
    heatmap_gerr[:]=np.nan
    heatmap_lerr=np.empty((max_compr+1,max_compr+1))
    heatmap_lerr[:]=np.nan
    heatmap_group_err=np.empty((max_compr+1,max_compr+1))
    heatmap_group_err[:]=np.nan
    heatmap_gcorr=np.empty((max_compr+1,max_compr+1))
    heatmap_gcorr[:]=np.nan
    heatmap_lcorr=np.empty((max_compr+1,max_compr+1))
    heatmap_lcorr[:]=np.nan
    heatmap_group_corr=np.empty((max_compr+1,max_compr+1))
    heatmap_group_corr[:]=np.nan
    ## Find the errors for pairs using these compression levels
    abs_values=True
    heatmap_d12,heatmap_u1,heatmap_u2,heatmap_u3,heatmap_d13,heatmap_d23=fill_matrix(data_dir,"Error","error",compr_levels,compr3,abs_values=abs_values)
    heatmap_g_d12,heatmap_g_u1,heatmap_g_u2,heatmap_g_u3,heatmap_g_d13,heatmap_g_d23=fill_matrix(data_dir,"Group_Error","group_errors",compr_levels,compr3,abs_values=abs_values)
    heatmap_c_d12,heatmap_c_u1,heatmap_c_u2,heatmap_c_u3,heatmap_c_d13,heatmap_c_d23=fill_matrix(data_dir,"Correlation","corrs",compr_levels,compr3,abs_values=abs_values)
    heatmap_c_d12=1-heatmap_c_d12
    heatmap_c_u1=1-heatmap_c_u1
    heatmap_c_u2=1-heatmap_c_u2
    heatmap_g_c_d12,heatmap_g_c_u1,heatmap_g_c_u2,heatmap_g_c_u3,heatmap_g_c_d13,heatmap_g_c_d23=fill_matrix(data_dir,"Group_Correlation","group_corrs",compr_levels,compr3,abs_values=abs_values)
    heatmap_g_c_d12=1-heatmap_g_c_d12
    heatmap_g_c_u1=1-heatmap_g_c_u1
    heatmap_g_c_u2=1-heatmap_g_c_u2
    if compr3:
        heatmap_c_d13=1-heatmap_c_d13
        heatmap_c_d23=1-heatmap_c_d23
        heatmap_c_u3=1-heatmap_c_u3
        heatmap_g_c_d13=1-heatmap_g_c_d13
        heatmap_g_c_d23=1-heatmap_g_c_d23
        heatmap_g_c_u3=1-heatmap_g_c_u3
    for compr in compr_levels:
        for compr2 in compr_levels:
            try:
                data=pd.read_csv(os.path.join(data_dir,"aggregated_"+str(compr)+"_"+str(compr2)+("_"+str(compr3) if compr3 else "")+".csv.gz"),delimiter=",")
                if not data.empty:
                    heatmap_lerr[compr,compr2]=data["ERROR"].mean()
                    heatmap_group_err[compr,compr2]=data["GROUP_ERROR"].mean()
                    heatmap_gerr[compr,compr2]=data["GLOBAL_ERROR"].mean()
                    heatmap_lcorr[compr,compr2]=1-data["CORR"].mean()
                    heatmap_group_corr[compr,compr2]=1-data["GROUP_CORR"].mean()
                    heatmap_gcorr[compr,compr2]=1-data["GLOBAL_CORR"].mean()
            except:
                print "skip"
    ticks=compr_levels[:-1]
    ticklabs=["1/"+str(i) if i>1 else "1" for i in compr_levels[:-1]]
    measure="Privacy-correlation"
    contour=False
    ## plot the heatmap
    plot_hmap(heatmap_d12,title,"error_d12.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",cmap=cmap,font_size=font_size)
    plot_hmap(heatmap_u1,title,"error_u1.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",show_contour=contour,cmap=cmap,font_size=font_size)
    plot_hmap(heatmap_u2,title,"error_u2.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",show_contour=contour,cmap=cmap,font_size=font_size)
    plot_hmap(heatmap_g_d12,title,"group_error_d12.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",cmap=cmap,num_decimals_legend=3,font_size=font_size)
    plot_hmap(heatmap_g_u1,title,"group_error_u1.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",show_contour=contour,cmap=cmap,font_size=font_size)
    plot_hmap(heatmap_g_u2,title,"group_error_u2.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",show_contour=contour,cmap=cmap,font_size=font_size)
    plot_hmap(heatmap_c_d12,title,"corr_d12.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",cmap=cmap,scale_percent=True,font_size=font_size)
    plot_hmap(heatmap_c_u1,title,"corr_u1.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",scale_percent=True,show_contour=contour,cmap=cmap,font_size=font_size)
    plot_hmap(heatmap_c_u2,title,"corr_u2.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",scale_percent=True,show_contour=contour,cmap=cmap,font_size=font_size)
    plot_hmap(heatmap_g_c_d12,title,"group_corr_d12.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",cmap=cmap,scale_percent=True,font_size=font_size)
    plot_hmap(heatmap_g_c_u1,title,"group_corr_u1.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",scale_percent=True,show_contour=contour,cmap=cmap,font_size=font_size)
    plot_hmap(heatmap_g_c_u2,title,"group_corr_u2.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",scale_percent=True,show_contour=contour,cmap=cmap,font_size=font_size)
    if compr3:
        plot_hmap(heatmap_d13,title,"error_d13.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",cmap=cmap,font_size=font_size)
        plot_hmap(heatmap_d23,title,"error_d23.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",cmap=cmap,font_size=font_size)
        plot_hmap(heatmap_u3,title,"error_u3.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",show_contour=contour,cmap=cmap,font_size=font_size)
        plot_hmap(heatmap_g_d13,title,"group_error_d13.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",cmap=cmap,num_decimals_legend=3,font_size=font_size)
        plot_hmap(heatmap_g_d23,title,"group_error_d23.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",cmap=cmap,num_decimals_legend=3,font_size=font_size)
        plot_hmap(heatmap_g_u3,title,"group_error_u3.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",show_contour=contour,cmap=cmap,font_size=font_size)
        plot_hmap(heatmap_c_d13,title,"corr_d13.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",cmap=cmap,scale_percent=True,font_size=font_size)
        plot_hmap(heatmap_c_d23,title,"corr_d23.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",scale_percent=True,cmap=cmap,font_size=font_size)
        plot_hmap(heatmap_c_u3,title,"corr_u3.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",scale_percent=True,show_contour=contour,cmap=cmap,font_size=font_size)
        plot_hmap(heatmap_g_c_d13,title,"group_corr_d13.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",cmap=cmap,scale_percent=True,font_size=font_size)
        plot_hmap(heatmap_g_c_d23,title,"group_corr_d23.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",scale_percent=True,cmap=cmap,font_size=font_size)
        plot_hmap(heatmap_g_c_u3,title,"group_corr_u3.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",scale_percent=True,show_contour=contour,cmap=cmap,font_size=font_size)
    plot_hmap(heatmap_lerr,title,"mean_lerr.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",show_contour=contour,cmap=cmap,font_size=font_size)
    plot_hmap(heatmap_group_err,title,"mean_group_err.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",show_contour=contour,cmap=cmap,font_size=font_size)
    plot_hmap(heatmap_gerr,title,"mean_gerr.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",show_contour=contour,cmap=cmap,font_size=font_size)
    plot_hmap(heatmap_lcorr,title,"mean_lcorr.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",scale_percent=True,show_contour=contour,cmap=cmap,font_size=font_size)
    plot_hmap(heatmap_group_corr,title,"mean_group_corr.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",scale_percent=True,show_contour=contour,cmap=cmap,font_size=font_size)
    plot_hmap(heatmap_gcorr,title,"mean_gcorr.pdf",plot_dir,xlim=[min_compr-0.5,max_compr-0.5],ylim=[min_compr-0.5,max_compr-0.5],ticks=ticks,ticklabs=ticklabs,xlab="Summarization $a_2$",ylab="Summarization $a_1$",scale_percent=True,show_contour=contour,cmap=cmap,font_size=font_size)

def plot_group_lines(x,local_errors,stds,global_errors,filename,xlab="",xlab2="",aspect_ratio=(10,10),font_size=16,grp_counts=None,tit="",ylim=None,ylim2=None):
    local_errors=[i for i in local_errors if not np.isnan(i)]
    stds=[i for i in stds if not np.isnan(i)]
    global_errors=[i for i in global_errors if not np.isnan(i)]
    fig, ax = plt.subplots(figsize=aspect_ratio)
    fig.suptitle("")
    ax.set_xlim([min(x),max(x)])
    if ylim!=None:
        ax.set_ylim(ylim)
    ax.set_xlabel(xlab,fontsize=font_size)
    #plt.ylim(0,max(max(score_dec[0]),max(cost_dec[0]),max(priv_dec[0]),max(score_cen),max(cost_cen),max(priv_cen)))
    ax.plot(x,local_errors,color='b',label="Local Group Error",linewidth=2)#,yerr=stds)
    ax.fill_between(x,np.asarray(local_errors)-np.asarray(stds),np.asarray(local_errors)+np.asarray(stds),alpha=0.2,linestyle="-",facecolor="b")
    ax.set_ylabel("Local Group Error",fontsize=font_size)
    if grp_counts:
        ax2=ax.twiny()
        ax2.set_xlabel(xlab2,fontsize=font_size)
        ax2.set_xlim([min(x),max(x)])
        ax.axes.get_xaxis().set_ticks([i for i in x if i%5==0])
        ax2.axes.get_xaxis().set_ticks([i for i in x if i%5==0])
        #ax.axes.get_xaxis().set_ticklabels([str(i)+" ("+str(int(j))+")" for i,j in zip(x,grp_counts) if i%5==0])
        ax.axes.get_xaxis().set_ticklabels([str(i) for i in x if i%5==0])
        ax2.axes.get_xaxis().set_ticklabels([str(int(j)) for i,j in zip(x,grp_counts) if i%5==0])
        ax.xaxis.grid() # vertical lines
        ax2.tick_params(labelsize=font_size)
    ax.tick_params(labelsize=font_size)
    ax1=ax.twinx()
    m=np.mean([g for g in global_errors if g])
    ax1.set_xlim([min(x),max(x)])
    if ylim2!=None:
        ax1.set_ylim(ylim2)
    else:
        ax1.set_ylim([m-0.01,m+0.01])
    ax1.plot(x,global_errors,'r',label="Global Error",linewidth=2)
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('Global error', color='r',fontsize=font_size)
    ax1.tick_params(labelsize=font_size)
    for tl in ax1.get_yticklabels():
        tl.set_color('r')
    # add legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax1.get_legend_handles_labels()
    legend=ax.legend(h1+h2, l1+l2, loc=4,fontsize=font_size,title=tit)
    plt.setp(legend.get_title(),fontsize=font_size)
    plt.tight_layout(pad=1.5)
    fig.savefig(filename,format='pdf')
    plt.close(fig)

def plot_lines_and_histograms(dirs,group_sizes,summ_levs,ratios=[None],sizes=[None],count_labs=None,plot_hist=True,cmap=None,ylim=None,ylim2=None):
    if not any(sizes):
        print "default sizes"
        sizes=[16 for _ in dirs] # default font size
    for data_dir,aspect_ratio,font_size in zip(dirs,ratios,sizes):
        plot_dir=os.path.join(data_dir,"plots")
        try:
            os.mkdir(plot_dir)
        except OSError:
            print("Directory already exists")
        for n in range(len(summ_levs)):
            summ_level=summ_levs[n]
            (local_error,std,global_error)=read_file(os.path.join(data_dir,"aggregated_"+str(summ_level)+".csv.gz"))
            if local_error==None:
                print "Skipping compression "+str(summ_level)
            else:
                local_errors=[np.nan,local_error]
                stds=[np.nan,std]
                global_errors=[np.nan,global_error]
                for l in group_sizes[2:]:    # start from 2
                    filename=os.path.join(data_dir,"aggregated_"+str("_").join([str(summ_level)]*(l))+".csv.gz")
                    (local_error_pair,stds_pair,global_error_pair)=read_file(filename)
                    local_errors.append(local_error_pair if local_error_pair else np.nan)
                    stds.append(stds_pair if stds_pair else np.nan)
                    global_errors.append(global_error_pair if global_error_pair else np.nan)
                x=np.where(~np.isnan(local_errors))[0]
                counts=[]
                histogram=np.empty((max(x)+1,max(x)+1))
                histogram[:]=np.nan
                for l in x:
                    filename=os.path.join(data_dir,"group_size_hist_"+str("_").join([str(summ_level)]*(l))+".csv.gz")
                    try:
                        data=pd.read_csv(filename)
                        norm=float(data[data['size']>1].sum()["count"])
                        counts.append(norm)
                        for i in data.index:
                            histogram[int(data[data.index==i]["size"]),l]=int(data[data.index==i]["count"])/norm
                    except:
                        print("File "+filename+" does not exist")
                        counts.append(np.nan)

                plot_group_lines(x,local_errors,stds,global_errors,os.path.join(plot_dir,"groups_"+str(summ_level)+".pdf"),xlab="Group size",xlab2="Number of groups",aspect_ratio=aspect_ratio,font_size=font_size,grp_counts=False,tit=("No. of users: "+str(count_labs[n]) if count_labs else ""),ylim=ylim,ylim2=ylim2)
                # plot group histograms
                if plot_hist:
                    tks=[i for i in x if i%5==0]
                    tks_l=[str(i) for i in x if i%5==0]
                    tks_l2=[str(int(j)) for i,j in zip(x,counts)  if i%5==0 and i>0]
                    plot_hmap(histogram,"","group_hist_"+str(summ_level)+".pdf",plot_dir,xlim=(1.5,max(x)+0.5),xlab="Group size",xlab2="Number of groups",ylab="Actual group size",ylim=(0.5,max(x)+0.5),font_size=font_size,cmap=cmap,scale_percent=True,ticks=tks,ticklabs=[tks_l,tks],ticklabs_2x=tks_l2)

def plot_compare_lines(root_dir,dirs,labels,group_sizes,summ_levs,aspect_ratio=None,font_size=16,count_labs=None,xlab="",ylab="",xlab2=""):
    plot_dir=os.path.join(root_dir,"plots")
    try:
        os.mkdir(plot_dir)
    except OSError:
        print("Directory already exists")
    cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=range(len(dirs))[-1])
    cmap = matplotlib.cm.jet
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)
    for summ_level in summ_levs:
        fig, ax = plt.subplots(figsize=aspect_ratio)
        fig.suptitle("")
        plt.xlabel(xlab,fontsize=font_size)
        #plt.ylim(0,max(max(score_dec[0]),max(cost_dec[0]),max(priv_dec[0]),max(score_cen),max(cost_cen),max(priv_cen)))
        ax.set_xlim([1,max(group_sizes)])
        ax.set_ylabel("Local Group Error",fontsize=font_size)
        ax.tick_params(labelsize=font_size)

        fig1, ax1 = plt.subplots(figsize=aspect_ratio)
        fig1.suptitle("")
        plt.xlabel(xlab,fontsize=font_size)
        #plt.ylim(0,max(max(score_dec[0]),max(cost_dec[0]),max(priv_dec[0]),max(score_cen),max(cost_cen),max(priv_cen)))
        ax1.set_xlim([1,max(group_sizes)])
        ax1.set_ylabel("Local Group Error",fontsize=font_size)
        ax1.tick_params(labelsize=font_size)

        plt.tight_layout(pad=1.0)

        avg_counts=None
        for d,lab in zip(range(len(dirs)),labels):
            data_dir=os.path.join(root_dir,dirs[d])
            # summ_level=summ_levs[n]
            (local_error,std,global_error)=read_file(os.path.join(data_dir,"aggregated_"+str(summ_level)+".csv.gz"))
            if local_error==None:
                print "Skipping compression "+str(summ_level)
            else:
                local_errors=[np.nan,local_error]
                stds=[np.nan,std]
                global_errors=[np.nan,global_error]
                for l in group_sizes[2:]:    # start from 2
                    filename=os.path.join(data_dir,"aggregated_"+str("_").join([str(summ_level)]*(l))+".csv.gz")
                    (local_error_pair,stds_pair,global_error_pair)=read_file(filename)
                    local_errors.append(local_error_pair if local_error_pair else np.nan)
                    stds.append(stds_pair if stds_pair else np.nan)
                    global_errors.append(global_error_pair if global_error_pair else np.nan)
                x=np.where(~np.isnan(local_errors))[0]
                counts=[]
                for l in x:
                    filename=os.path.join(data_dir,"group_size_hist_"+str("_").join([str(summ_level)]*(l))+".csv.gz")
                    try:
                        data=pd.read_csv(filename)
                        norm=float(data.sum()["count"])
                        counts.append(norm)
                    except:
                        print("File "+filename+" does not exist")
                        counts.append(np.nan)
                ax.plot(x,np.asarray(local_errors)[x],label=str(lab),linewidth=2,color=scalarMap.to_rgba(d))#,yerr=stds)
                ax.fill_between(x,np.asarray(local_errors)[x]-np.asarray(stds)[x],np.asarray(local_errors)[x]+np.asarray(stds)[x],alpha=0.2,linestyle="-",color=scalarMap.to_rgba(d))
                ax.legend(loc=4,fontsize=font_size)
                # m=np.mean([g for g in global_errors if g])
                # ax1.set_ylim([m-0.01,m+0.01])
                ax1.plot(x,np.asarray(global_errors)[x],label=str(lab),linewidth=2,color=scalarMap.to_rgba(d))
                ax1.legend(loc=4,fontsize=font_size)
                if avg_counts!=None:
                    avg_counts+=np.asarray(counts)
                else:
                    avg_counts=np.asarray(counts)
        avg_counts/=len(dirs)   # average
        ticks=[1]+[i for i in group_sizes if i%5==0 and i!=0]
        ax.set_xticks(ticks)
        ax2=ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.tick_params(labelsize=font_size)
        ax2.set_xlabel(xlab2,fontsize=font_size)
        ax2.set_xticks(ax.get_xticks())
        ax2.set_xticklabels(avg_counts.astype(np.int))
        ax.xaxis.grid() # vertical lines

        ax1.set_xticks(ticks)
        ax3=ax1.twiny()
        ax3.set_xlim(ax1.get_xlim())
        ax3.tick_params(labelsize=font_size)
        ax3.set_xlabel(xlab2,fontsize=font_size)
        ax3.set_xticks(ax1.get_xticks())
        ax3.set_xticklabels(avg_counts.astype(np.int))
        ax1.xaxis.grid() # vertical lines

        fig.savefig(os.path.join(plot_dir,"group_hist_le_"+str(summ_level)+".pdf"),format='pdf')
        plt.close(fig)
        fig1.savefig(os.path.join(plot_dir,"group_hist_ge_"+str(summ_level)+".pdf"),format='pdf')
        plt.close(fig1)

def groups_distr(dirs,group_sizes,summ_levs,count_labs=None,sizes=[1,5,10],font_sizes=None,cmap=None,ylim=None):
    if font_sizes==None:
        font_sizes=[20 for _ in dirs]
    for data_dir,font_size in zip(dirs,font_sizes):
        plot_dir=os.path.join(data_dir,"plots")
        try:
            os.mkdir(plot_dir)
        except OSError:
            print("Directory already exists")
        ## plot surface with local error varying with compression level and group size
        heatmap_lerr=np.empty((len(summ_levs),max(group_sizes)+1))
        heatmap_lerr[:]=np.nan
        heatmap_lerr_std=np.empty((len(summ_levs),max(group_sizes)+1))
        heatmap_lerr_std[:]=np.nan
        heatmap_counts=np.empty((len(summ_levs),max(group_sizes)+1))
        heatmap_counts[:]=np.nan
        for i in range(len(summ_levs)):
            for l in group_sizes:
                filename=os.path.join(data_dir,"aggregated_"+str("_").join([str(summ_levs[i])]*(l))+".csv.gz")
                if os.path.isfile(filename):
                    data=pd.read_csv(filename)
                    heatmap_lerr[i][l]=data["ERROR"].mean()
                    heatmap_lerr_std[i][l]=data["ERROR_STD"].mean()
                else:
                    print "skipping group size "+str(l)+" and summarization "+str(summ_levs[i])
                filename=os.path.join(data_dir,"group_size_hist_"+str("_").join([str(summ_levs[i])]*(l))+".csv.gz")
                if os.path.isfile(filename):
                    data=pd.read_csv(filename)
                    norm=float(data.sum()["count"])
                    heatmap_counts[i][l]=norm
                else:
                    print "skipping histogram for group size "+str(l)+" and summarization "+str(summ_levs[i])
        counts=np.nanmean(heatmap_counts,axis=0) # average columns, across summarization lvels
        plot_hmap(heatmap_lerr,"","lerr_heatmap.pdf",plot_dir,xlim=[min(group_sizes)+0.5,max(group_sizes)-0.5],xlab="Group size",xlab2="Number of groups",ylab="Summarization",ticks=[group_sizes,range(len(summ_levs))],ticklabs=[[str(l) if l%5==0 and l>0 else "" for l in group_sizes],["1/"+str(i) for i in summ_levs]],ticklabs_2x=[str(int(c)) if l%5==0 and l>0 else "" for l,c in zip(group_sizes,counts)],font_size=font_size,cmap=cmap)
        ## plot overlapping lines
        fig, ax = plt.subplots()
        #fig.suptitle("Local Error",fontsize=font_size)
        plt.xlabel("Summarization",fontsize=font_size)
        plt.ylabel("Local Group Error",fontsize=font_size)
        plt.gca().xaxis.set_ticks(summ_levs)
        plt.gca().xaxis.set_ticklabels(["1/"+str(i) for i in summ_levs])
        plt.gca().invert_xaxis()
        ax.tick_params(labelsize=font_size)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.2,
                          box.width, box.height * 0.75])
        if ylim!=None:
            ax.set_ylim(ylim)
        #cmap=None
        if cmap!=None:
            cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=range(len(sizes)+1)[-1])
            scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)
        for i in range(len(sizes)):
            l=sizes[i]
            if cmap!=None:
                colorVal = scalarMap.to_rgba(i+1) # skip first color = white
            else:
                colorVal=None
            y=heatmap_lerr[:,l]
            s=heatmap_lerr_std[:,l]
            ymask=np.isfinite(y)
            smask=np.isfinite(s)
            if any(ymask):
                (_,caps,_)=ax.errorbar(np.asarray(summ_levs)[ymask],y[ymask],yerr=s[ymask],color=colorVal,linewidth=3,elinewidth=2,capsize=5)
                for cap in caps:
                    cap.set_markeredgewidth(2)
                #ax.fill_between(np.asarray(summ_levs)[ymask & smask],y[ymask & smask]-s[ymask & smask],y[ymask & smask]+s[ymask & smask],alpha=0.2,linestyle="-")
        if cmap!=None:
            colorVal = scalarMap.to_rgba(1)
        else:
            colorVal=None
        ax.axhline(heatmap_lerr[0,1],ls="dashed",color=colorVal,linewidth=3)
        if count_labs:
            # add counts on top
            ax2=ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            box = ax2.get_position()
            ax2.set_position([box.x0, box.y0 + box.height * 0.2,
                              box.width, box.height * 0.75])
            ax2.set_xticks(np.asarray(summ_levs)[ymask])
            plt.setp(ax2.xaxis.get_majorticklabels(),rotation=-90,fontsize=font_size)
            ax2.set_xticklabels(np.array(count_labs)[ymask])
        counts_l=[c for s,c in zip(group_sizes,counts) if s in sizes]
        # Put a legend below current axis
        legend=ax.legend(["Baseline"]+[str(i) for i in sizes],loc='upper center', bbox_to_anchor=(0.5, -0.15), #[str(i)+" ("+str(int(c))+")" for i,c in zip(sizes,counts_l)]
                          fancybox=True, shadow=True, ncol=3,fontsize=font_size)
        #ax.legend([str(i)+" ("+str(int(c))+")" for i,c in zip(sizes,counts_l)], loc=3,fontsize=font_size)
        fig.savefig(os.path.join(plot_dir,"strategic.pdf"),format='pdf')
        plt.close(fig)

def group_fractions(plot_dir,ns):
    try:
        os.mkdir(plot_dir)
    except OSError:
        print("Directory already exists")
    for n in ns:
        x=[0]
        (local_errors,stds,global_errors)=read_file(os.path.join(data_dir,"aggregated_"+str(n)+"_"+str(n)+".csv.gz"))
        local_errors=[local_errors]
        global_errors=[global_errors]
        stds=[stds]
        for fract in np.arange(0.1,1.1,0.1):
            filename=os.path.join(data_dir,"aggregated_"+str(n)+"_"+str(n)+"_fract"+str(fract)+".csv.gz")
            if os.path.isfile(filename): # read pre-existing file
                x.append(fract)
                (local_error_pair,std_pair,global_error_pair)=read_file(filename)
                local_errors.append(local_error_pair)
                stds.append(std_pair)
                global_errors.append(global_error_pair)
        fig, ax = plt.subplots()
        fig.suptitle("Error for group fraction")
        plt.xlabel("Population grouped")
        #plt.ylim(0,max(max(score_dec[0]),max(cost_dec[0]),max(priv_dec[0]),max(score_cen),max(cost_cen),max(priv_cen)))
        ax.plot(x,local_errors,'b')
        ax.set_ylabel("Local error")
        ax1=ax.twinx()
        ax1.set_ylim([0,0.01])
        ax1.plot(x,global_errors,'r')
        # Make the y-axis label and tick labels match the line color.
        ax1.set_ylabel('Global error', color='r')
        for tl in ax1.get_yticklabels():
            tl.set_color('r')
        fig.savefig(os.path.join(plot_dir,"group_fractions_"+str(n)+".pdf"),format='pdf')
        plt.close(fig)

def plot_group_stats(matrix,plot_dir,filename):
    fig,ax=plt.subplots(figsize=(10,5))
    ax.set_xlabel("Group summarization",fontsize=16)
    ax.set_ylabel("Measure",fontsize=16)
    ax.tick_params(labelsize=16)
    for col in ["Error","Correlation","Group_Error","Group_Correlation"]:
        mask=np.isfinite(matrix[col])
        x=matrix["Compression"][mask]
        y=matrix[col][mask]
        yerr=matrix[col+"_std"][mask]
        ax.errorbar(x,y,yerr=yerr,label=col)
    ax.legend(loc=2)
    fig.savefig(os.path.join(plot_dir,filename+".pdf"),format='pdf')
    plt.close(fig)

def main(data_dir,plot_dir,percentage=None):
    matplotlib.style.use('classic')
    ## plot bar histogram with nrel user counts for each summarization level
    root_dir="./results_nrel/"
    counts=pd.read_csv(os.path.join(root_dir,"data_counts.csv.gz"))
    levels=range(20)
    fig,ax=plt.subplots(figsize=(10,5))
    ax.set_xlabel("Summarization Level",fontsize=16)
    ax.set_ylabel("Users considered",fontsize=16)
    ax.tick_params(labelsize=16)
    y=[int(counts[counts["COMPRESSION"]==i]["COUNT"]) for i in levels]
    plt.gca().xaxis.set_ticks([i-0.6 for i in levels+np.asarray(1) if i%5==0])
    plt.gca().xaxis.set_ticklabels(["1/"+str(i) for i in levels+np.asarray(1) if i%5==0])
    ax.set_xlim([-0.2,max(levels)+1])
    plt.gca().invert_xaxis()
    ax.bar(levels,y,color="b")
    fig.tight_layout()
    fig.savefig(os.path.join(root_dir,"data_counts.pdf"),format='pdf')
    plt.close(fig)
    ## replicate plots of pournaras 2016, nrel
    root_dir="./results_nrel/"
    data_dir=os.path.join(root_dir,"indiv")
    plot_dir=os.path.join(data_dir,"plots")
    counters_loc=os.path.join(root_dir,"data_counts.csv.gz")
    ## replicate plots of pournaras 2016, ecbt vs ecbt with sMAPE
    plot_individual_errors([data_dir],plot_dir,counters_loc,nlevels=38)
    root_dirs=["./results_ecbt_original/","./results_ecbt/"]
    data_dirs=[os.path.join(r,"indiv") for r in root_dirs]
    plot_dir=os.path.join(data_dirs[1],"plots")
    plot_individual_errors(data_dirs,plot_dir,legend_labs=["Original","Symmetric"],legend_styles=["dashed","solid"])

    ## plot experiment with clustering agents in k groups
    # root_dir="./results_clustering/"
    # data_dir=os.path.join(root_dir,"indiv_det")
    # plot_dir=os.path.join(data_dir,"plots")
    # try:
    #     os.mkdir(plot_dir)
    # except OSError:
    #     print("Directory already exists")
    # #plot_lines_and_histograms([data_dir],range(1,50),
    # summ_level=None
    # num_users=None
    # local_errors=[np.nan]
    # stds=[np.nan]
    # global_errors=[np.nan]
    # for l in range(1,50):
    #     filename=os.path.join(data_dir,"aggregated_exp_clustering_"+str(l)+".csv")
    #     (local_error_pair,stds_pair,global_error_pair)=read_file(filename)
    #     local_errors.append(local_error_pair if local_error_pair else np.nan)
    #     stds.append(stds_pair if stds_pair else np.nan)
    #     global_errors.append(global_error_pair if global_error_pair else np.nan)
    # x=np.where(~np.isnan(local_errors))[0]
    # plot_group_lines(x,local_errors,stds,global_errors,os.path.join(plot_dir,"groups_nums.pdf"),xlab="Number of groups")

    cx1 = plt.get_cmap('cubehelix_r')
    ### plot aggregated data
    for root_dir,tit in [["./results_nrel/","NREL dataset"],["./results_ecbt/","ECBT dataset"]]:
        for compr3 in [None]:#,10]:
            if compr3:
                data_dir=os.path.join(root_dir,"groups3")
                plot_dir=os.path.join(data_dir,"./plot_three_"+str(compr3))
            else:
                data_dir=os.path.join(root_dir,"groups2")
                plot_dir=os.path.join(data_dir,"./plot")
            try:
                os.mkdir(plot_dir)
            except OSError:
                print("Directory already exists")
            if os.path.exists(data_dir):
                plot_group_heatmaps(min_compr,max_compr,compr3,cx1,font_size=20,title=tit)
    # plot_heatmap(heatmap_global,os.path.join(plot_dir,"heatmap_global.pdf"))


    # plot the difference between the errors of all pairs of numbers that sum up to the name number
    # vec=[]
    # for c in compr_levels:
    #     combs=[comb for comb in combinations_with_replacement(compr_levels, 2) if sum(comb) == c]
    #     vec.append([local_error[compr_levels.index(c)],
    #                 [heatmap_local[x,y] for (x,y) in combs]])
    # mlen=max([len(a[1]) for a in vec])
    # colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(vec)))
    # fig, ax = plt.subplots()
    # for (a,c) in zip(vec,colors):
    #     y=a[1]+[np.nan]*(mlen-len(a[1])) # pad the list
    #     print y
    #     ax.plot(range(mlen),y,color=c)
    # fig.savefig(os.path.join(plot_dir,"test_.pdf"),format='pdf')
    # plt.close()

    ## plot macro level graphs
    cx1 = plt.get_cmap('cubehelix_r')
    ylim3=[0,0.5]
    for root_dir,plot_num_users,ylim,ylim2 in [["./results_nrel/",True,[0,0.3],[-0.015,0.015]],["./results_ecbt/",False,[0,0.45],[-0.015,0.015]]]:
        group_sizes=range(21)
        summ_levs=[1,2,3,4,5,6,7,8,9,10]
        dirs=[["groups",(8,6),20,"Deterministic"],["groups_uniform",(8,6),20,"Uniform"],["groups_powerlaw",(8,6),20,"Power law"],["groups_step",(8,6),20,"Step function"]]
        sizes=[1,2,5,10,20]
        if plot_num_users and os.path.isfile(os.path.join(root_dir,"data_counts.csv.gz")):
            counts=pd.read_csv(os.path.join(root_dir,"data_counts.csv.gz"))
            count_labs=[int(counts[counts["COMPRESSION"]==i]["COUNT"]) for i in summ_levs]
        else:
            count_labs=None
        groups_distr([os.path.join(root_dir,d) for d,r,s,l in dirs],group_sizes,summ_levs,count_labs,sizes,cmap=cx1,ylim=ylim3)
        ## plot individual lines
        plot_lines_and_histograms([os.path.join(root_dir,d) for d,r,s,l in dirs],group_sizes,summ_levs,count_labs=count_labs,ratios=[r for d,r,s,l in dirs],sizes=[s for d,r,s,l in dirs],cmap=cx1,ylim=ylim,ylim2=ylim2)
        plot_compare_lines(root_dir,[d for d,r,s,l in dirs],[l for d,r,s,l in dirs],group_sizes,summ_levs,count_labs=count_labs,aspect_ratio=(8,6),font_size=20)


    ## fractions: generate groups of size 2 and group only a fraction of the population
    # resulting graphs similar to those with varying group size
    data_dir="./results_nrel/groups_fractions"
    plot_dir=os.path.join(data_dir,"plots")
    group_fractions(plot_dir,[10,20])

    ## plot experiment with clustering agents in k groups
    cx1 = plt.get_cmap('cubehelix_r')
    root_dir="./experiment_grouping/"
    data_dir=os.path.join(root_dir,"results")
    plot_dir=os.path.join(data_dir,"plots")
    try:
        os.mkdir(plot_dir)
    except OSError:
        print("Directory already exists")
    max_std=10
    stds=range(max_std)
    max_group=61
    font_size=20
    fcts=[["cl_data","Data proximity"],["cl_rnd","Random grouping"],["cl_sum","Summarization proximity"]]
    matrix_le=np.empty([len(fcts),len(stds),max_group])
    matrix_le[:]=np.nan
    matrix_std=np.empty([len(fcts),len(stds),max_group])
    matrix_std[:]=np.nan
    matrix_ge=np.empty([len(fcts),len(stds),max_group])
    matrix_ge[:]=np.nan
    for n in range(len(fcts)): # functions
        for g in range(max_group):
            for s in range(len(stds)):
                filename=os.path.join(data_dir,"aggregated_exp_clustering_"+str(g)+"_fct_"+fcts[n][0]+"_std_"+str(stds[s])+".csv.gz")
                if os.path.isfile(filename):
                    (local_error_pair,stds_pair,global_error_pair)=read_file(filename)
                    matrix_le[n,s,g]=local_error_pair
                    matrix_std[n,s,g]=stds_pair
                    matrix_ge[n,s,g]=global_error_pair
                else:
                    print "File "+filename+" not found"
    # ### plot the group statistics: measures aggregated by group, depending on the average summarization level in the group
    # matrix_group_stats=pd.DataFrame()
    # for n in range(len(fcts)): # functions
    #     matrix_group_stats_by_fct=pd.DataFrame()
    #     ### aggregate by std first, then by group size
    #     for g in range(len(groups)):
    #         matrix_group_stats_by_fct_grp=pd.DataFrame()
    #         for s in range(len(stds)):
    #             filename=os.path.join(data_dir,"group_stats_exp_clustering_"+str(groups[g])+"_fct_"+fcts[n][0]+"_std_"+str(stds[s])+".csv")
    #             if os.path.isfile(filename):
    #                 group_stats=pd.read_csv(filename)
    #                 if "COUNTER" not in group_stats.columns:
    #                     print filename
    #                 if "Unnamed: 0" in group_stats.columns:
    #                     group_stats=group_stats.drop(["Unnamed: 0"],1)
    #                 matrix_group_stats_by_fct_grp=matrix_group_stats_by_fct_grp.add(group_stats,fill_value=0)
    #             else:
    #                 print "File "+filename+" not found"
    #         matrix_group_stats_by_fct=matrix_group_stats_by_fct.add(matrix_group_stats_by_fct_grp,fill_value=0)
    #         matrix_group_stats_by_fct_grp=matrix_group_stats_by_fct_grp.div(matrix_group_stats_by_fct_grp.COUNTER, axis='index') # don't divide the TIME
    #         matrix_group_stats_by_fct_grp["Compression"]=matrix_group_stats_by_fct_grp.index
    #         if not matrix_group_stats_by_fct_grp.empty:
    #             plot_group_stats(matrix_group_stats_by_fct_grp,plot_dir,"group_stats_comparison_fct"+str(fcts[n][0])+"_grp"+str(groups[g]))
    #         else:
    #             print "empty g"+str(groups[g])+" s "+str(stds[s])
    #     matrix_group_stats=matrix_group_stats.add(matrix_group_stats_by_fct,fill_value=0)
    #     matrix_group_stats_by_fct=matrix_group_stats_by_fct.div(matrix_group_stats_by_fct.COUNTER, axis='index') # don't divide the TIME
    #     matrix_group_stats_by_fct["Compression"]=matrix_group_stats_by_fct.index
    #     plot_group_stats(matrix_group_stats_by_fct,plot_dir,"group_stats_comparison_fct"+str(fcts[n][0]))
    #     ### aggregate by group size first, then by std
    #     for s in range(len(stds)):
    #         matrix_group_stats_by_fct_std=pd.DataFrame()
    #         for g in range(len(groups)):
    #             filename=os.path.join(data_dir,"group_stats_exp_clustering_"+str(groups[g])+"_fct_"+fcts[n][0]+"_std_"+str(stds[s])+".csv")
    #             if os.path.isfile(filename):
    #                 group_stats=pd.read_csv(filename)
    #                 if "COUNTER" not in group_stats.columns:
    #                     print filename
    #                 if "Unnamed: 0" in group_stats.columns:
    #                     group_stats=group_stats.drop(["Unnamed: 0"],1)
    #                 matrix_group_stats_by_fct_std=matrix_group_stats_by_fct_std.add(group_stats,fill_value=0)
    #             else:
    #                 print "File "+filename+" not found"
    #         matrix_group_stats_by_fct=matrix_group_stats_by_fct.add(matrix_group_stats_by_fct_std,fill_value=0)
    #         matrix_group_stats_by_fct_std=matrix_group_stats_by_fct_std.div(matrix_group_stats_by_fct_std.COUNTER, axis='index') # don't divide the TIME
    #         matrix_group_stats_by_fct_std["Compression"]=matrix_group_stats_by_fct_std.index
    #         plot_group_stats(matrix_group_stats_by_fct_std,plot_dir,"group_stats_comparison_fct"+str(fcts[n][0])+"_std"+str(stds[s]))
    #     matrix_group_stats=matrix_group_stats.add(matrix_group_stats_by_fct,fill_value=0)
    #     matrix_group_stats_by_fct=matrix_group_stats_by_fct.div(matrix_group_stats_by_fct.COUNTER, axis='index') # don't divide the TIME
    #     matrix_group_stats_by_fct["Compression"]=matrix_group_stats_by_fct.index
    #     plot_group_stats(matrix_group_stats_by_fct,plot_dir,"group_stats_comparison_fct"+str(fcts[n][0]))
    # matrix_group_stats=matrix_group_stats.div(matrix_group_stats.COUNTER, axis='index') # don't divide the TIME
    # matrix_group_stats["Compression"]=matrix_group_stats.index
    # plot_group_stats(matrix_group_stats,plot_dir,"group_stats_comparison_fct")
    cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=len(fcts)+1)
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cx1)
    colors=[scalarMap.to_rgba(i) for i in range(1,len(fcts)+1)]
    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%.2f' % height,
                    ha='center', va='bottom')
    for s in range(1,max_std): # stds
        local_errors=matrix_le[:,s,:]
        stds=matrix_std[:,s,:]
        global_errors=matrix_ge[:,s,:]
        aspect_ratio=[10,5]
        xlab="Number of groups"
        ylab="Local Group Error"
        ## plot lines local error
        filename=os.path.join(plot_dir,"comparison_std_"+str(s)+".pdf")
        fig, ax = plt.subplots(figsize=aspect_ratio)
        fig.suptitle("")
        ax.set_xlabel(xlab,fontsize=font_size)
        ax.set_ylabel(ylab,fontsize=font_size)
        for le,std,ge,col,name in zip(local_errors,stds,global_errors,colors,zip(*fcts)[0]):
            ax.errorbar(range(max_group),le,yerr=std,color=col,label=name,linewidth=3)#,yerr=stds)
        ax.tick_params(labelsize=font_size)
        ax.legend(loc=3,ncol=3)
        plt.tight_layout()
        fig.savefig(filename,format='pdf')
        plt.close(fig)
        ## plot barcharts
        filename=os.path.join(plot_dir,"comparison_std_bar_"+str(s)+".pdf")
        fig, ax = plt.subplots(figsize=aspect_ratio)
        fig.suptitle("")
        ax.set_xlabel(xlab,fontsize=font_size)
        ax.set_ylabel(ylab,fontsize=font_size)
        x=[1,10,20,40,60]
        width=0.25
        rects=[]
        for i,le,std,ge,col,name in zip(range(len(fcts)),local_errors,stds,global_errors,colors,zip(*fcts)[1]):
            rects.append(ax.bar(range(len(x))+np.asarray(i*width),le[x],width,yerr=std[x],color=col,label=name,error_kw=dict(ecolor='darkgray', lw=2, capsize=5, capthick=2)))
        plt.gca().xaxis.set_ticks(range(len(x))+np.asarray(len(fcts)*width/2.0))
        plt.gca().xaxis.set_ticklabels(x)
        ax.tick_params(labelsize=font_size)
        ax.set_ylim([0.3,0.5])
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0  + box.height * 0.1,
                         box.width, box.height * 0.85])
        # Put a legend below current axis
        legend=ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
                         fancybox=True, shadow=True, ncol=3)#,fontsize=font_size)
        #plt.tight_layout()
        for r in rects:
            autolabel(r)
        fig.savefig(filename,format='pdf')
        plt.close(fig)
        ## plot barcharts gerr
        filename=os.path.join(plot_dir,"comparison_std_bar_ge_"+str(s)+".pdf")
        fig, ax = plt.subplots(figsize=aspect_ratio)
        fig.suptitle("")
        ax.set_xlabel(xlab,fontsize=font_size)
        ax.set_ylabel(ylab,fontsize=font_size)
        x=[1,10,20,40,60]
        width=0.25
        rects=[]
        for i,le,std,ge,col,name in zip(range(len(fcts)),local_errors,stds,global_errors,colors,zip(*fcts)[1]):
            rects.append(ax.bar(range(len(x))+np.asarray(i*width),ge[x],width,color=col,label=name))
        plt.gca().xaxis.set_ticks(range(len(x))+np.asarray(len(fcts)*width/2.0))
        plt.gca().xaxis.set_ticklabels(x)
        ax.tick_params(labelsize=font_size)
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0  + box.height * 0.1,
                         box.width, box.height * 0.85])
        # Put a legend below current axis
        legend=ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
                         fancybox=True, shadow=True, ncol=3)#,fontsize=font_size)
        fig.savefig(filename,format='pdf')
        plt.close(fig)
    ## plot heatmaps
    max_group=10
    max_std=9
    for n in range(len(fcts)): # functions
        heatmap_le=matrix_le[n,:,:]
        plot_hmap(heatmap_le[:,:max_group],"Local Group Error","heatmap_lerr_"+str(fcts[n][0])+".pdf",plot_dir,xlab="Number of groups",ylab="Standard Deviation",xlim=[1-0.5,max_group-0.5],ylim=[1-0.5,max_std-0.5],ticks=[range(1,max_group),range(1,max_std)],ticklabs=[range(1,max_group),range(1,max_std)],cmap=cx1,num_decimals_legend=3,display_text=False,font_size=font_size)
        heatmap_ge=matrix_ge[n,:,:]
        plot_hmap(heatmap_ge[:,:max_group],"Global Error","heatmap_gerr_"+str(fcts[n][0])+".pdf",plot_dir,xlab="Number of groups",ylab="Standard Deviation",xlim=[1-0.5,max_group-0.5],ylim=[1-0.5,max_std-0.5],ticks=[range(1,max_group),range(1,max_std)],ticklabs=[range(1,max_group),range(1,max_std)],cmap=cx1,num_decimals_legend=3,display_text=False,font_size=font_size)

    ## figure explaining privacy-correlation measure
    root_dir="./results_ecbt"
    plot_dir=os.path.join(root_dir,"plots")
    try:
        os.mkdir(plot_dir)
    except OSError:
        print("Directory already exists")
    x=np.linspace(0,6*np.pi,100)
    cos_mean=2
    sin_mean=4
    y_cos=np.cos(x)+cos_mean
    y_sin=np.sin(x)+sin_mean
    from sklearn.cluster import KMeans
    classifier=KMeans(n_clusters=5) # create classifier
    y_cos_summ=classifier.fit_predict(y_cos[np.newaxis].T) # perform classification
    y_cos_summ=np.asarray([round(classifier.cluster_centers_[i][0],2) for i in y_cos_summ])

    classifier=KMeans(n_clusters=1) # create classifier
    y_sin_summ=classifier.fit_predict(y_sin[np.newaxis].T) # perform classification
    y_sin_summ=np.asarray([round(classifier.cluster_centers_[i][0],2) for i in y_sin_summ])

    y_avg=np.mean([y_cos_summ,y_sin_summ],axis=0)
    y_err=abs(y_cos_summ-y_avg)
    y_err_tr=abs((y_cos_summ-cos_mean)-(y_avg-np.mean([cos_mean,sin_mean])))

    from tools import pearson_coeff_sample
    y_err_t_s=round(1-pearson_coeff_sample(y_avg,y_cos_summ),2)

    font_size=16
    line_width=3
    fig,ax=plt.subplots(figsize=(10,2))
    plt.gca().xaxis.set_ticks([])
    plt.gca().xaxis.set_ticklabels([])
    ax.tick_params(labelsize=font_size)
    ax.plot(x,y_cos,linewidth=line_width)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir,"priv_corr_1.pdf"),format='pdf')
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,2))
    plt.gca().xaxis.set_ticks([])
    plt.gca().xaxis.set_ticklabels([])
    ax.tick_params(labelsize=font_size)
    ax.plot(x,y_cos_summ,linewidth=line_width)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir,"priv_corr_2.pdf"),format='pdf')
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,2))
    plt.gca().xaxis.set_ticks([])
    plt.gca().xaxis.set_ticklabels([])
    ax.tick_params(labelsize=font_size)
    ax.plot(x,y_sin,color='r',linewidth=line_width)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir,"priv_corr_3.pdf"),format='pdf')
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,2))
    plt.gca().xaxis.set_ticks([])
    plt.gca().xaxis.set_ticklabels([])
    ax.tick_params(labelsize=font_size)
    ax.plot(x,y_sin_summ,color='r',linewidth=line_width)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir,"priv_corr_4.pdf"),format='pdf')
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,3))
    plt.gca().xaxis.set_ticks([])
    plt.gca().xaxis.set_ticklabels([])
    ax.tick_params(labelsize=font_size)
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                      box.width, box.height * 0.9])
    # Put a legend below current axis
    ax.plot(x,y_cos,color='b',linewidth=line_width,linestyle='dashed',label="Raw")
    ax.plot(x,y_cos_summ,color='b',linewidth=line_width,label="Summarized")
    legend=ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0),
                      fancybox=True, shadow=True, ncol=2,fontsize=font_size)
    #fig.tight_layout()
    fig.savefig(os.path.join(plot_dir,"priv_corr_1a.pdf"),format='pdf')
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,3))
    plt.gca().xaxis.set_ticks([])
    plt.gca().xaxis.set_ticklabels([])
    ax.tick_params(labelsize=font_size)
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                      box.width, box.height * 0.9])
    # Put a legend below current axis
    ax.plot(x,y_sin,color='r',linewidth=line_width,linestyle='dashed',label="Raw")
    ax.plot(x,y_sin_summ,color='r',linewidth=line_width,label="Summarized")
    legend=ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0),
                      fancybox=True, shadow=True, ncol=2,fontsize=font_size)
    #fig.tight_layout()
    fig.savefig(os.path.join(plot_dir,"priv_corr_2a.pdf"),format='pdf')
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5))
    plt.gca().xaxis.set_ticks([])
    plt.gca().xaxis.set_ticklabels([])
    ax.tick_params(labelsize=font_size)
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                      box.width, box.height * 0.9])
    # Put a legend below current axis
    ax.plot(x,y_avg,color='violet',label="Average of $a_1$ and $a_2$",linewidth=line_width)
    ax.axhline(np.mean(y_avg),ls="dashed",linewidth=line_width,color='violet')
    ax.plot(x,y_cos_summ,color='b',label="Summarized data of $a_2$",linewidth=line_width)
    ax.axhline(np.mean(y_cos_summ),ls="dashed",linewidth=line_width,color='b')
    legend=ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0),
                      fancybox=True, shadow=True, ncol=2,fontsize=font_size)
    #fig.tight_layout()
    fig.savefig(os.path.join(plot_dir,"priv_corr_5.pdf"),format='pdf')
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5))
    plt.gca().xaxis.set_ticks([])
    plt.gca().xaxis.set_ticklabels([])
    ax.tick_params(labelsize=font_size)
    ax.plot(x,y_err,linewidth=line_width)
    ax.axhline(np.mean(y_err),ls="dashed",linewidth=line_width)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir,"priv_corr_6.pdf"),format='pdf')
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5))
    plt.gca().xaxis.set_ticks([])
    plt.gca().xaxis.set_ticklabels([])
    ax.tick_params(labelsize=font_size)
    ax.plot(x,y_err_tr,linewidth=line_width)
    ax.axhline(np.mean(y_err_tr),ls="dashed",linewidth=line_width)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir,"priv_corr_7.pdf"),format='pdf')
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5))
    plt.gca().xaxis.set_ticks([])
    plt.gca().xaxis.set_ticklabels([])
    ax.tick_params(labelsize=font_size)
    ax.plot(x,[y_err_t_s]*len(x),linewidth=line_width)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir,"priv_corr_8.pdf"),format='pdf')
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5))
    plt.gca().xaxis.set_ticks([])
    plt.gca().xaxis.set_ticklabels([])
    ax.tick_params(labelsize=font_size)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                      box.width, box.height * 0.9])
    # Put a legend below current axis
    ax.plot(x,y_err,color='b',label="Standard",linewidth=line_width)
    ax.axhline(np.mean(y_err),ls="dashed",color='b',linewidth=line_width)
    ax.plot(x,y_err_tr,color='g',label="Translation invariant",linewidth=line_width)
    ax.set_ylim([-0.2,1.6])
    ax.axhline(np.mean(y_err_tr),ls="dashed",color='g',linewidth=line_width)
    ax.axhline(np.mean(y_err_t_s),color='r',label="Translation & scale invariant",linewidth=line_width+1)
    legend=ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0),
                      fancybox=True, shadow=True, ncol=3,fontsize=font_size-2)
    #fig.tight_layout()
    fig.savefig(os.path.join(plot_dir,"priv_corr_comparison.pdf"),format='pdf')
    plt.close(fig)



# parser = argparse.ArgumentParser(description='reads in parameters')

# # Add the arguments for the parser to be passed on the cmd-line
# # Defaults could be added default=
# parser.add_argument('--data_dir', metavar='data_dir', nargs=1,help='the data directory')
# parser.add_argument('--plot_dir', metavar='plot_dir', nargs=1,help='the plot directory')
# parser.add_argument('--percentage', metavar='percentage', nargs=1,help='the percentage of users to plot')

# args = parser.parse_args()

#main (args.data_dir[0],args.plot_dir[0],args.percentage[0])


# Notes:
# Let us denote with xg the x coordinate at which the global errors intersect and xl the x coordinate at which the local errors intersect
# The plots show that, given an accuracy requirement y, one should choose a starting compression and a recompression values such that y is below the value of global error at xg, as the gain is maximum below that point.
# The difference plots show that, if there is no accuracy constranit, the best idea is to recompress to 3 clusters (because we cap the error to 1 at recompressions 1 and 2??), and that the best accuracy difference is obtained close to xl.

### Plot an example of user data and its summarization
# from tools import *
# from matplotlib import rcParams
# rcParams.update({'figure.autolayout': True})
# font_size=16
# cmap = plt.get_cmap('cubehelix_r')
# ks=[5,3]
# cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=range(len(ks)+3)[-1])
# scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)

# input_file="./datasets/ecbt/run_48/output/daily/users/cluster48/user_3182.csv"
# df=pd.read_csv(input_file)
# df["TIME"]=separate_daily_obs(df['TIME'])
# df=df[df["TIME"]==340]
# df["TIME"]=range(len(df))        # reindex
# fig,ax=plt.subplots()
# #fig.suptitle("Example of smart meter data",fontsize=font_size)
# # global error
# ax.set_ylabel("Power demand (kW)",fontsize=font_size)
# ax.set_xlabel("Hours",fontsize=font_size)
# ax.xaxis.set_ticks(np.arange(0,49,4))
# ax.xaxis.set_ticklabels(np.arange(0,25,2))
# ax.tick_params(labelsize=font_size)
# ax.plot(df["TIME"],df["RAW"],color=scalarMap.to_rgba(1),label="Raw data",linewidth=2)
# for k,i in zip(ks,range(len(ks))):
#     classifier=KMeans(n_clusters=k,max_iter=300,tol=1e-4) # create classifier
#     raw=np.asarray(df['RAW'])[np.newaxis].T
#     df["SUML"+str(k)]=classifier.fit_predict(raw) # perform classification
#     df["SUMC"+str(k)]=np.take(classifier.cluster_centers_,classifier.labels_)
#     ax.plot(df["TIME"],df["SUMC"+str(k)],label=str(k)+" clusters",color=scalarMap.to_rgba(i+2),linewidth=2)
# ax.legend(fontsize=font_size)
# fig.savefig(os.path.join(".","data_example.pdf"),format='pdf')
# plt.close(fig)
