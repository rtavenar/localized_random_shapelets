import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_shapeletinfo_from_name(name_shap):
    """
    Method to get a dictionnary that describes the shapelet

    Parameters
    ----------
    name_shap : name of the shapelet

    Returns
    -------
    dictionary
    """
    name_shap_split = name_shap.split('#')
    shapelet_info= {}
    shapelet_info['ts'] = int(name_shap_split[1])
    shapelet_info['pos'] = int(name_shap_split[2].split('-')[0])
    shapelet_info['length'] = int(name_shap_split[2].split('-')[1])
    return shapelet_info


def get_shapelet(df_ts, shapelet_info):
    """
    Method to get the shapelet

    Parameters
    ----------
    df_ts : dataframe that contains the time series in which the shapelet has been drawn
    shapelet_info : dictionary that describes the shapelet

    Returns
    -------
    numpy array, shape (1, length of shapelet)
    """
    ts = shapelet_info['ts']
    pos = shapelet_info['pos']
    L = shapelet_info['length']
    return df_ts.loc[ts][pos:pos + L]


def get_ts(df_ts, shapelet_info):
    """
    Method to get the timeseries in which the shapelet has been drawn

    Parameters
    ----------
    df_ts : dataframe that contains the time series in which the shapelet has been drawn
    shapelet_info : dictionary that describes the shapelet

    Returns
    -------
    numpy array, shape (1, length of the time series)
    """
    ts = shapelet_info['ts']
    return df_ts.loc[ts]

def plot_UCRdataset_shapelet(name_shap, df_ts):
    """
    Method to plot the shapelet in the time series where it has been drawn

    Parameters
    ----------
    df_ts : dataframe that contains the time series in which the shapelet has been drawn
    name_shap : name of the shapelet

    Returns
    -------
    numpy array, shape (1, length of shapelet)
    """
    shapelet_info = get_shapeletinfo_from_name(name_shap)
    ts = get_ts(df_ts, shapelet_info)
    plt.figure(figsize=(20, 3))
    ts.plot()
    shap= get_shapelet(df_ts, shapelet_info)
    shap.plot(marker='.', c='r')

def multihist(x, **kwargs):
    """
    Method to plot an histogram with specific parameters

    Parameters
    ----------
    x : data
    kwargs : supplementary parameters

    Returns
    -------
    histogram
    """
    return plt.hist(x,histtype='step',linewidth=3, **kwargs)

def plot_distribution_loc_dist_TwoPatterns(name_shap,twpatt_df_shaprep,twpatt_y_train):
    """
    Method to plot the distribution of localization and of distance for the selected shapelet in the TwoPatterns dataset

    Parameters
    ----------
    name_shap : name of the shapelet to plot
    twpatt_df_shaprep : dataframe of the time series represented via shapelet features
    twpatt_y_train : class of the time series, lenght must be equal to the length of twpatt_df_shaprep

    Returns
    -------
    save and plot the distribution of localization and of distance for the selected shapelet
    """
    twpatt_df_shaprep['class'] = ['1 (A-A)' if y == 1 else '2 (B-A)' if y == 2 else '3 (A-B)' if y == 3 else '4 (B-B)' for y in twpatt_y_train]
    plt.rc("legend",title_fontsize = 'x-large')
    shapelet_df = twpatt_df_shaprep[['{}loc'.format(name_shap),'{}dist'.format(name_shap),'class']]
    shapelet_df = pd.melt(shapelet_df,id_vars=['class'],value_vars=['{}loc'.format(name_shap),'{}dist'.format(name_shap)])
    shapelet_df.replace({'{}loc'.format(name_shap):'loc','{}dist'.format(name_shap):'dist'},inplace=True)
    shapelet_df.columns =['class','category','value']
    
    g0 = sns.FacetGrid(shapelet_df[shapelet_df['category']=='dist'], hue='class', size=5, aspect=2, sharex='none')
    _ = g0.map(multihist, 'value', alpha=0.6)
    g0.axes[0,0].set_xlabel('shapelet distance',fontsize=35)
    g0.axes[0,0].set_ylabel('number of time series',fontsize=35)
    g0.axes[0,0].legend(title='class',title_fontsize=20,loc=1,bbox_to_anchor=(1.25, 0.75))
    handles,labels = g0.axes[0,0].get_legend_handles_labels()
    handles = [handles[3],handles[0],handles[1],handles[2]]
    labels = [labels[3],labels[0],labels[1],labels[2]]
    g0.axes[0,0].legend(handles,labels,title='class',title_fontsize=30,loc=1,bbox_to_anchor=(1.25, 0.75))
    g0.set(xlim = (0,40))
    g0.savefig("./TwoPatterns_{}_dist_distribution.pdf".format(name_shap),bbox_inches='tight')
    
    g1 = sns.FacetGrid(shapelet_df[shapelet_df['category']=='loc'], hue='class', size=5, aspect=2, sharex='none')
    _ = g1.map(multihist, 'value', alpha=0.6)
    g1.axes[0,0].set_xlabel('localization',fontsize=35)
    g1.axes[0,0].set_ylabel('number of time series',fontsize=35)
    g1.axes[0,0].legend(title='class',title_fontsize=20,loc=1,bbox_to_anchor=(1.25, 0.75))
    handles,labels = g1.axes[0,0].get_legend_handles_labels()
    handles = [handles[3],handles[0],handles[1],handles[2]]
    labels = [labels[3],labels[0],labels[1],labels[2]]
    g1.set(xlim = (0,120))
    g1.savefig("./TwoPatterns_{}_loc_distribution.pdf".format(name_shap),bbox_inches='tight')
    plt.show()

