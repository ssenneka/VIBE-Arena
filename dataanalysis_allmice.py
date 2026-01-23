# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 17:01:11 2025

@author: mcmadmx
"""

import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.transforms as transforms
from scipy.stats import ranksums
from scipy.stats import wilcoxon
import scipy
import starbars
import statannot


def flatten_list(xss):
    return [x for xs in xss for x in xs]
def get_unique(data):
     used = set()
     try:
         unique_data = [x for x in data if x not in used and (used.add(x) or True)]
     except Exception:
         print(Exception)
         flat_data = flatten_list(data)
         unique_data = [int(x) for x in flat_data if x not in used and (used.add(x) or True)]
     return unique_data
def find(data,value):
    locs = np.argwhere(np.array(data)==value)
    if len(locs) > 0:
        locs = np.concatenate(locs)
    else:
        locs = []
    return locs

stats = pd.DataFrame(index = range(0,5),columns=["Average","STDEV","Stim Paradigm"])
stim_types = ["Combined","ICMS Only","Dim Visual Only","Bright Visual Only","Sham"]


mice = ["M1","M2","M3","M4","M5","M6"]

allmice_data = {i:[] for i in mice}
for k in mice:
    mouse = k
    all_data = pd.read_csv("C://Users//mcmadmx//Personal//Downloads//Mouse training data//Mouse training data//C1//"+mouse+"//"+mouse+" all data.csv")
    all_data["Average Speed"] = all_data["Average Speed"]
    allmice_data[k] = all_data


last5 = {}
last5s = {}
last5f = {}
first5  = {}
first5s = {}
first5f = {}

first_icms = {}
spd_counts = pd.DataFrame(columns=["Success","Total","Mouse","Stim Paradigm"],index = range(0,20))
counter = 0
for i in allmice_data:
    temp = allmice_data[i].copy()
    seshe = 6
    if i == "M6":
        seshl = 19
    elif i == "M1":
        seshl = 70
    else:
        seshl = 23
    # seshl = max(temp["Session"])
    last5[i] = temp[temp.columns].iloc[np.concatenate(np.argwhere(np.array(temp["Session"]) <= seshl))]
    last5[i] = last5[i][last5[i].columns].iloc[np.concatenate(np.argwhere(np.array(last5[i]['Session']) > seshl-5))]
    last5s[i] = last5[i][last5[i].columns].iloc[np.concatenate(np.argwhere(np.array(last5[i]["Success"]) == 1))]
    last5f[i] = last5[i][last5[i].columns].iloc[np.concatenate(np.argwhere(np.array(last5[i]["Success"]) == 0))]
    first5[i] = temp[temp.columns].iloc[np.concatenate(np.argwhere(np.array(temp["Session"]) < seshe))]
    first5[i] = first5[i][first5[i].columns].iloc[np.concatenate(np.argwhere(np.array(first5[i]["Session"]) >= seshe-5))]
    first5s[i] = first5[i][first5[i].columns].iloc[np.concatenate(np.argwhere(np.array(first5[i]["Success"]) == 1))]
    first5f[i] = first5[i][first5[i].columns].iloc[np.concatenate(np.argwhere(np.array(first5[i]["Success"]) == 0))]
    firstprobe = temp["Session"].iloc[min(np.concatenate(np.argwhere(np.array(temp["Stim Paradigm"]) == "ICMS Only")))]
    if i == "C71L":
        firstprobe = 15
    elif i == "SC11LM":
        firstprobe = 15
    first_icms[i] = temp[temp.columns].iloc[np.concatenate(np.argwhere(np.array(temp["Session"]) == firstprobe))]

averagel5 = {"Time-to-Target":[],"Average Speed":[],"Path Efficiency":[],"AD":[],"Success":[]}
averagel5f = {"Time-to-Target":[],"Average Speed":[],"Path Efficiency":[],"AD":[],"Success":[]}

for i in averagel5:
    averagel5[i] = pd.DataFrame(index = range(0,200),columns=["Average","STDEV","Stim Paradigm","Mouse","Session"])
    averagel5f[i] = pd.DataFrame(index = range(0,200),columns=["Average","STDEV","Stim Paradigm","Mouse","Session"])
    counter = 0
    for j in last5s:
        # temp = last5[j].copy()
        if i == "Success":
            temp = last5[j].copy()
        else:
            temp = last5s[j].copy()
        tempf = last5f[j].copy()
        sesh1 = round(min(temp["Session"]))
        seshend = round(max(temp["Session"])) + 1
        for k in stim_types:
            if len(np.argwhere(np.array(temp["Stim Paradigm"])==k)) > 1:
                temp2 = temp[temp.columns].iloc[np.concatenate(np.argwhere(np.array(temp["Stim Paradigm"]) == k))].copy()
            for l in range(sesh1,seshend):
                locs = np.argwhere(np.array(temp2["Session"]) == l)
                if len(locs) > 0:
                    temp3 = temp2[temp2.columns].iloc[np.concatenate(locs)].copy()
                    
                    averagel5[i]["Average"].iloc[counter] = np.average([float(temp3[i].iloc[k]) for k in range(0,len(temp3[i]))])
                    averagel5[i]["STDEV"].iloc[counter] = np.std([float(temp3[i].iloc[k]) for k in range(0,len(temp3[i]))])
                    averagel5[i]["Stim Paradigm"].iloc[counter] = k
                    averagel5[i]["Mouse"].iloc[counter] = j
                    averagel5[i]["Session"].iloc[counter] = l
                else:
                    averagel5[i]["Average"].iloc[counter] = np.nan
                    averagel5[i]["STDEV"].iloc[counter] = np.nan
                    averagel5[i]["Stim Paradigm"].iloc[counter] = k
                    averagel5[i]["Mouse"].iloc[counter] = j

                
                counter += 1

palettes = {"Combined":"tab:purple","ICMS Only":"tab:red","Dim Visual Only":"tab:blue","Bright Visual Only":"tab:cyan","Sham":"tab:pink"}
palette2 = {"C71L":"dimgray","C82R":"black","C91L":"grey","C91R":"silver","C91L1R":"lightgray","SC11LM":"gainsboro"}



Drive = "C://Users//mcmadmx//Personal//Downloads//Mouse training data//Mouse training data"
stim_paradigms = ["Combined","ICMS Only","Bright Visual Only","Dim Visual Only","Sham"]
mice = ["M1","M2","M3","M4","M5", "M6"]
palette = {"Combined":"tab:purple","ICMS Only":"tab:red","Dim Visual Only":"tab:blue","Bright Visual Only":"tab:cyan","Sham":"tab:pink"}
plt.figure(figsize=(11,4))
for l in stim_paradigms:
    all_micedata = {mice[i]:[] for i in range(0,len(mice))}
    for k in mice:
        mouse = k
        all_data = pd.read_csv(Drive+"//C1//"+mouse+"//"+mouse+" all data.csv")
        all_data = all_data[all_data.columns].iloc[np.concatenate(np.argwhere(np.array(all_data["Stim Paradigm"])==l))]
        if mouse == "1LF":
            switch_sesh = 19
        elif mouse == "1LM":
            switch_sesh = 20
        elif mouse != "1LF" or mouse != "1LM":
            switch_sesh = 26
        
        if mouse != "M1":
            temp = all_data[all_data.columns].iloc[np.concatenate(np.argwhere(np.array(all_data["Session"])<switch_sesh))]
            temp = temp[temp.columns].iloc[np.concatenate(np.argwhere(np.array(temp["Stim Paradigm"])==l))]
            learning_data = {"Session":[],"Success":[],"mouse":[]}
            for i in get_unique(temp["Session"]):
                temp2 = temp[temp.columns].iloc[np.concatenate(np.argwhere(np.array(temp["Session"])==i))]
                learning_data["Session"].append(int(i))
                learning_data["Success"].append(np.average(temp2["Success"]))
                learning_data["mouse"].append(k)
            learning_data = pd.DataFrame(learning_data)
            if l != "Combined":
                if mouse != "1LF" and mouse != "1LM":
                    learning_data = learning_data[learning_data["Session"]>=16]
                else:
                    pass
            all_micedata[k] = learning_data
        else:
            if (l != "Sham" and l != "Bright Visual Only"):
                temp = all_data[all_data.columns].iloc[np.concatenate(np.argwhere(np.array(all_data["Session"])<switch_sesh))]
                temp = temp[temp.columns].iloc[np.concatenate(np.argwhere(np.array(temp["Stim Paradigm"])==l))]
                learning_data = {"Session":[],"Success":[],"mouse":[]}
                for i in get_unique(temp["Session"]):
                    temp2 = temp[temp.columns].iloc[np.concatenate(np.argwhere(np.array(temp["Session"])==i))]
                    learning_data["Session"].append(int(i))
                    learning_data["Success"].append(np.average(temp2["Success"]))
                    learning_data["mouse"].append(k)
                learning_data = pd.DataFrame(learning_data)
                if l != "Combined":
                    learning_data = learning_data[learning_data["Session"]>=16]
                all_micedata[k] = learning_data
            else: 
                learning_data = {"Session":[],"Success":[],"mouse":[]}
                learning_data = pd.DataFrame(learning_data)
                if l != "Combined":
                    learning_data = learning_data[learning_data["Session"]>=16]
                all_micedata[k] = learning_data

    all_micedata = pd.concat([all_micedata[i] for i in list(all_micedata.keys())],ignore_index=True)   
    overall_avg = {"Session":[],"Success":[],"STD":[]}
    two_mice = {"Session":[],"Success":[],"STD":[]}
    for i in get_unique(all_micedata["Session"]):
        overall_avg["Session"].append(int(i))
        overall_avg["Success"].append(np.average(all_micedata["Success"].iloc[np.concatenate(np.argwhere(np.array(all_micedata["Session"])==i))]))
        overall_avg["STD"].append(np.std(all_micedata["Success"].iloc[np.concatenate(np.argwhere(np.array(all_micedata["Session"])==i))]))
                                                                            
    overall_avg = pd.DataFrame(overall_avg)
    overall_avg = overall_avg.sort_values(by="Session")
    plt.errorbar(overall_avg["Session"],overall_avg["Success"],overall_avg["STD"],color=palette[l],label=l,linewidth=2)
    plt.xlabel("Session")
    plt.ylabel("Success Rate")
    plt.plot([1,25],[0.75,0.75],color='k',alpha=0.5)

    sns.despine(top=True,right=True)
    plt.ylim([0,1])
    plt.xlim([0,25.1])
    plt.xticks([1,5,10,15,20,25])
    plt.legend(bbox_to_anchor=[1.05,.75])
plt.savefig("All (new) Mice Learning SR.pdf",bbox_inches="tight")
plt.show()
    



for i in averagel5:
    averagel5[i]["Stim Paradigm"].iloc[np.concatenate(np.argwhere(np.array(averagel5[i]["Stim Paradigm"])=="Combined"))] = "Multimodal"
    averagel5[i]["Stim Paradigm"].iloc[np.concatenate(np.argwhere(np.array(averagel5[i]["Stim Paradigm"])=="ICMS Only"))] = "ICMS"
    averagel5[i]["Stim Paradigm"].iloc[np.concatenate(np.argwhere(np.array(averagel5[i]["Stim Paradigm"])=="Dim Visual Only"))] = "Dim\nVisual"
    averagel5[i]["Stim Paradigm"].iloc[np.concatenate(np.argwhere(np.array(averagel5[i]["Stim Paradigm"])=="Bright Visual Only"))] = "Bright\nVisual"
    



palettes = {"Multimodal":"tab:purple","ICMS":"tab:red","Dim\nVisual":"tab:blue","Bright\nVisual":"tab:cyan","Sham":"tab:pink"}
palette2 = {"M1":"dimgray","M2":"black","M3":"grey","M4":"silver","M5":"lightgray","M6":"gainsboro"}
stim_types = ["Multimodal", "ICMS", "Dim\nVisual","Bright\nVisual","Sham"]
statistics = {}
statistics2 = {}
aov = {}
post_hocs = {}
plt.rcParams.update({'font.size':16})
fig, axes = plt.subplots(3,2,figsize=(30,20),constrained_layout=True)
sp_coords = [[0,0],[0,1],[1,0],[1,1],[2,0],[2,1],[3,0],[3,1]]
itr = 0
flat_axes = axes.flatten()
for i in flat_axes:
    i.set_box_aspect(1)

stim_save = stim_types.copy()
p_vals_all = {i:[] for i in list(averagel5.keys())}

for i in averagel5:
    data = averagel5[i].copy()
    data["Average"] = data["Average"].astype(float)
    data_avg = data[['Average','Stim Paradigm','Mouse']].groupby(['Mouse','Stim Paradigm'])['Average'].mean().reset_index().copy()
    stim_types = stim_save.copy()
    if i == "Success":
        stim_types = ["Multimodal", "ICMS", "Dim\nVisual","Bright\nVisual","Sham"]
    else:
        stim_types = ["Multimodal", "ICMS", "Dim\nVisual","Bright\nVisual","Sham"]
   

        
    statistics[i] = []
    box_pair = []
    p_vals = []
    stim_types2 = stim_types.copy()
    for k in stim_types:
        data2 = data.dropna()
        for j in stim_types2:
            if j != k:
                try:
                    stat, wrs_xcom = wilcoxon(data2["Average"].iloc[np.concatenate(np.argwhere(np.array(data2["Stim Paradigm"])==k))], data2["Average"].iloc[np.concatenate(np.argwhere(np.array(data2["Stim Paradigm"])==j))])
                except Exception:
                    print("Different Sample Counts")
                    stat, wrs_xcom = ranksums(data2["Average"].iloc[np.concatenate(np.argwhere(np.array(data2["Stim Paradigm"])==k))], data2["Average"].iloc[np.concatenate(np.argwhere(np.array(data2["Stim Paradigm"])==j))])

                box_pair.append((k,j))
                p_vals.append([wrs_xcom,(k,j)])                
                statistics[i].append((str(k),str(j),float(wrs_xcom)))
        stim_types2.remove(k)
    p_vals = pd.DataFrame(p_vals,columns=["P-value","Pair"])
    p_vals = p_vals.sort_values(by='P-value').reset_index(drop=True)
    p_vals["P-val-corr"]= [p_vals["P-value"][k]*(k+1) for k in range(0,len(p_vals))]
    statistics[i] = p_vals.copy()
    if len(np.argwhere(np.array(p_vals["P-val-corr"])>0.05)) > 0:
        p_vals = p_vals.drop(np.concatenate(np.argwhere(np.array(p_vals["P-val-corr"])>0.05)))
    p_vals_all[i] = p_vals
        
        
        
    box_pair2 = box_pair.copy()
    statistics2[i] = pd.DataFrame(index=stim_types,columns=stim_types)
    stim_types2 = stim_types.copy()
    try:
        data2 = data.dropna()
    except Exception:
        data2 = data.copy()
    counter = 0
    stats2 = pd.DataFrame(index = range(0,20),columns=["Average","STDEV","Stim Paradigm"])
    for k in stim_types:
        stats2["Average"].iloc[counter] = float(np.average(data2["Average"].iloc[np.concatenate(np.argwhere(np.array(data2["Stim Paradigm"]) == k))]))
        stats2["STDEV"].iloc[counter] = np.std(data2["Average"].iloc[np.concatenate(np.argwhere(np.array(data2["Stim Paradigm"]) == k))])
        stats2["Stim Paradigm"].iloc[counter] = k
        counter += 1
    offset = lambda p: transforms.ScaledTranslation(p/72.,0,plt.gcf().dpi_scale_trans)
    
    mpl.rcParams['pdf.fonttype'] = 42

    plt.rcParams['font.family'] = 'Arial'
    trans = flat_axes[itr].transData
    for k in stim_types:
        
        if i != "Success":
            new_handle = flat_axes[itr].errorbar(stats2["Stim Paradigm"].iloc[np.concatenate(np.argwhere(np.array(stats2["Stim Paradigm"])==k))],
                                  stats2["Average"].iloc[np.concatenate(np.argwhere(np.array(stats2["Stim Paradigm"])==k))],
                                  stats2["STDEV"].iloc[np.concatenate(np.argwhere(np.array(stats2["Stim Paradigm"])==k))],
                                  marker='o',ms=15,mfc=palettes[k],
                                  mec=palettes[k],zorder=2,ecolor=palettes[k],
                                  linestyle='',linewidth=2,alpha=1,
                                  transform=trans+offset(30))
        elif i == "Success":
            new_handle = flat_axes[itr].errorbar(stats2["Stim Paradigm"].iloc[np.concatenate(np.argwhere(np.array(stats2["Stim Paradigm"])==k))],
                                    stats2["Average"].iloc[np.concatenate(np.argwhere(np.array(stats2["Stim Paradigm"])==k))],
                                    stats2["STDEV"].iloc[np.concatenate(np.argwhere(np.array(stats2["Stim Paradigm"])==k))],
                                    marker='o',ms=18,mfc=palettes[k],
                                    mec=palettes[k],zorder=2,ecolor=palettes[k],
                                    linestyle='',linewidth=2,alpha=1,
                                    transform=trans+offset(30))
    
    marker = ['s','d','o','x','+']
    offsets = [-18.75,-11.25,-3.75,3.75,11.25,18.75]
    counter = 0
    if itr == 0:
        leg = True
    else:
        leg = False
    for j in mice:
        
        data3 = data[data.columns].iloc[np.concatenate(np.argwhere(np.array(data["Mouse"]) == j))]
        
        axs=sns.stripplot(x="Stim Paradigm",y="Average",hue="Mouse",data=data3,size=5,zorder=1,palette = palette2,dodge=True,jitter=0,transform=trans+offset(offsets[counter]),ax = flat_axes[itr])
        counter+=1
    
    
    
    y_height = np.average(data2["Average"])+3*np.std(data["Average"])
    
    sns.despine(top=True,right=True,bottom=True)
    handles, labels = plt.gca().get_legend_handles_labels()
    if i == "Path Efficiency" or i == "AD":
        flat_axes[itr].set_yticks(np.linspace(0,1,num=5))
    
    elif i == "Time-to-Target":
        flat_axes[itr].set_yticks(np.linspace(0,5,num=5))
    elif i == "Success":
        flat_axes[itr].set_yticks(np.linspace(0,1,num=5))
   
    if len(p_vals) > 0:
        statannot.add_stat_annotation(flat_axes[itr],x="Stim Paradigm",y="Average",data=data2,
                                  box_pairs=p_vals["Pair"], perform_stat_test=False,
                                  pvalues=p_vals["P-val-corr"],
                                  loc="outside",
                                  text_format='full',
                                  line_offset_to_box = .1)
    flat_axes[itr].margins(0.1)
    plt.margins(0.2)
    
    if itr == 0:
        flat_axes[itr].legend(loc='lower left')
    else:
        flat_axes[itr].legend(handlelength=0,frameon=False)
    if i == "Success":
        flat_axes[itr].set_ylabel("Success Rate")
    elif i == "Time-to-Target":
        flat_axes[itr].set_ylabel("Time-to-Target (seconds)")
        flat_axes[itr].set_ylim([0.5,5])
    elif i == "Average Speed":
        flat_axes[itr].set_ylabel("Average Speed (cm/s)")
        flat_axes[itr].set_ylim([8,18])
    elif i == "AD":
        flat_axes[itr].set_ylabel("Angular Dispersion")
        flat_axes[itr].set_ylim([0,1])
    else:
        flat_axes[itr].set_ylabel(i)
        flat_axes[itr].set_ylim([0,1])
    flat_axes[itr].get_legend().remove()
    flat_axes[itr].set_box_aspect(.45)
    
    itr += 1

fig.savefig('All mice behavior metrics.pdf')
plt.show()


#####################################################################################
spd_counts = pd.DataFrame(columns=["Success","Fail","Total","Mouse","Stim Paradigm"],index = range(0,125))
counter = 0
stim_types = ["Combined","ICMS Only","Dim Visual Only","Bright Visual Only","Sham"]
for j in stim_types:
    for i in last5:
        temp = last5[i][last5[i].columns]#.iloc[np.concatenate(np.argwhere(np.array(last5[i]["Session"])==l))]
        success = sum(temp["Success"].iloc[np.concatenate(np.argwhere(np.array(temp["Stim Paradigm"])==j))])
        total = len(np.argwhere(np.array(temp["Stim Paradigm"]) == j))
        spd_counts["Success"].iloc[counter] = success
        spd_counts["Total"].iloc[counter] = total
        spd_counts["Fail"].iloc[counter] = total-success
        spd_counts["Mouse"].iloc[counter] = i
        spd_counts["Stim Paradigm"].iloc[counter] = j
        counter += 1
        
    spd_counts["Success"].iloc[counter] = sum(spd_counts["Success"].iloc[np.concatenate(np.argwhere(np.array(spd_counts["Stim Paradigm"])==j))])
    spd_counts["Total"].iloc[counter] = sum(spd_counts["Total"].iloc[np.concatenate(np.argwhere(np.array(spd_counts["Stim Paradigm"])==j))])
    spd_counts["Fail"].iloc[counter] = spd_counts["Total"].iloc[counter] - spd_counts["Success"].iloc[counter]
    spd_counts["Mouse"].iloc[counter] = "Total"
    spd_counts["Stim Paradigm"].iloc[counter] = j    
    counter += 1

t_totals = spd_counts[spd_counts.columns].iloc[np.concatenate(np.argwhere(np.array(spd_counts["Mouse"])=="Total"))].copy().reset_index(drop=True)
temp = {"Success": [sum(t_totals["Success"])],"Total": [sum(t_totals["Total"])],"Mouse":["All"],"Stim Paradigm":["Total"]}
temp = pd.DataFrame(temp)
t_totals = pd.concat([t_totals,temp])

plotting = "Total"
prev = 0
palettes = {"Combined":"tab:purple","ICMS Only":"tab:red","Dim Visual Only":"tab:blue","Bright Visual Only":"tab:cyan","Sham":"tab:pink","ICMS-dfix":"sandybrown","ICMS-pfix":"coral","ICMS-bfix":"moccasin"}

fig, ax = plt.subplots()
total = int(t_totals[plotting].iloc[np.concatenate(np.argwhere(np.array(t_totals["Stim Paradigm"]) == "Total"))])
for i in stim_types:
    sp_total = int(t_totals[plotting].iloc[np.concatenate(np.argwhere(np.array(t_totals["Stim Paradigm"]) == i))]) 
    sp_s = int(t_totals["Success"].iloc[np.concatenate(np.argwhere(np.array(t_totals["Stim Paradigm"]) == i))])
    sps_tot = int(t_totals["Total"].iloc[np.concatenate(np.argwhere(np.array(t_totals["Stim Paradigm"]) == i))])
    temp_frac = sp_total / total
    temp_sfrac = sp_s / total
    if prev == 0:
        ax.add_patch(plt.Rectangle([0,4.5],temp_frac,1,facecolor=palettes[i],alpha =0.25))
        ax.add_patch(plt.Rectangle([0,4.5],temp_sfrac,1,facecolor=palettes[i]))
        ax.add_patch(plt.Rectangle([0,4.5],temp_frac,1,facecolor='none',edgecolor="k",linewidth=1))
        prev = temp_frac      
    else:
        ax.add_patch(plt.Rectangle([prev,4.5],temp_frac,1,alpha=0.25,color=palettes[i]))
        ax.add_patch(plt.Rectangle([prev,4.5],temp_sfrac,1,facecolor=palettes[i]))
        ax.add_patch(plt.Rectangle([prev,4.5],temp_frac,1,facecolor='none',edgecolor="k",linewidth=1))


        prev += temp_frac
    plt.text(prev-temp_frac/2,3.5,str(sp_s)+"\n"+str(sp_total-sp_s)+"\n "+str(round(temp_frac*100,1))+"%",size = "xx-small",horizontalalignment='right',verticalalignment = 'center')
    plt.ylim([0,10])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


    plt.yticks([])
    plt.xticks([])
plt.savefig("Trial Split and SR summary.pdf")

