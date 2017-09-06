'''
Created on 24 de mai de 2016

@author: Fellipe

'''
import numpy as np
import matplotlib as mpl
from pprint import pprint
from scipy.stats.morestats import wilcoxon, shapiro
# mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import os.path
import csv
import glob
import collections
import re
import pandas as pd
from math import floor
from plagiarism_detection.extermal_plagiarism.heuristic_retrieval import hdf_to_sparse_matrix

def load_temp(files_root_path,_results_files_list):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for file_pathi in _results_files_list:
        file_path = os.path.join(files_root_path, file_pathi)
        print(file_path)
        if os.path.exists(file_path):

            a = pd.read_csv(file_path, delimiter='\t')
            
            a_keys = [
                'lsht__n_permutations',
                'lsht__selection_function',
                'lshnns__n_neighbors',
#                 'MAP',
#                 'MAP_std',
                'recall_mean',
                'recall_std',
                'lsht_features',
                'cv_documents_mean_time',
                'cv_queries_mean_time',
                'lsht_documents_mean_time',
                'lsht_queries_mean_time',
                'nns_documents_mean_time',
                'nns_queries_mean_time'
            ]
            print(a.loc[:,a_keys])
            
            colors = ['r','g','b','c','m','y','k']
            i = 0 
            
            for tok_id, group_values in a.groupby(['lsht__selection_function','lsht__n_permutations']).groups.items():
                print(tok_id,'--',group_values)
                if 'minmax_' in tok_id :
                    tok_id = 'minmax'
                    
#                 for gi in group_values:
#                     print(a.loc[gi,['lshnns__n_neighbors','lsht__n_permutations','lsht_features','lsht_queries_mean_time','nns_queries_mean_time']])
#                 elif 'MinMaxAFP(RAP)' in tok_id :
#                     pass
#                 else:
#                     continue
                a_i = a.loc[group_values,:]
                color_i = colors[i]
                i = (i+1) % len(colors)
                ax.scatter(a_i.loc[:,'lshnns__n_neighbors'],a_i.loc[:,'recall_mean'], a_i.loc[:,'nns_queries_mean_time'], c=color_i)
#                 
                for li in group_values:
                    zdir, x, y, z = None, a.loc[li,'lshnns__n_neighbors'], a.loc[li,'recall_mean'], a.loc[li,'nns_queries_mean_time']                   
                    label = '(%s, %d)' % (tok_id, a.loc[li,'lsht_features'])
                    ax.text(x, y, z, label, zdir)
                    
            ax.set_xlabel('neighbors')
            ax.set_ylabel('recall mean')
            ax.set_zlabel('search time')
            plt.show()
            print('----')
#     plt.show()

    exit()    

if __name__ == "__main__":
    
    files_root_path = "."

#     files_root_path = "."
    results_files_list = [
                            '2017-09-05_21-14-07_pan10-%d-samples_lsh_results.csv',
                            '2017-09-05_21-14-07_pan10-%d-samples_results.csv',
                        ]

    print(len(results_files_list))
    if len(results_files_list) > 0:
        results_files_list = [os.path.join(files_root_path, filei) for filei in results_files_list]
    else:
        results_files_list = glob.glob(os.path.join(files_root_path,'*csv'))
            
    tokenizer_parameters_to_group = [
        "cv__analyzer",
        "cv__ngram_range",
        "cv__tokenizer",
        "cv__lowercase",
        "cv__min_df",
        "cv__binary",
        "cv__stop_words",
        "cv__use_idf",
        "cv__norm",
#         "lsht__selection_function" 
    ]            

    recalls_dict = collections.defaultdict(list)
    
    lsh_features_list = set()
    
    load_temp(files_root_path,results_files_list)
    
    for file_pathi in results_files_list:
        file_path = os.path.join(files_root_path, file_pathi)
        if os.path.exists(file_path):

            contenti = pd.read_csv(file_path, delimiter='\t')
            '''
                groups by tokenizers configs.
            '''
            for tok_id, tok_values in contenti.groupby(tokenizer_parameters_to_group).groups.items():
                contenti_by_tokenizer = contenti.loc[tok_values,:]
                
                try:
                    contenti_by_sel_func = contenti_by_tokenizer.groupby(['lsht__selection_function']) 
                    for sel_func,sel_func_values in contenti_by_sel_func.groups.items():
                        print(sel_func,'=',len(sel_func_values))
                
                        contenti_by_neig = contenti.loc[sel_func_values,:].groupby(['lshnns__n_neighbors',])
                         
                        for neig,neig_values in contenti_by_neig.groups.items():
                            print('recall@',neig,' with ',len(neig_values),' values')
                            
                            estimators_results = contenti.loc[neig_values,['precision_recall_path','lsht__n_permutations','lsht_features','lshnns__n_neighbors','recall_mean','recall_std','indexing_mean_time','querying_mean_time','documents_count','nns_queries_mean_time']]
                             
                            x = np.asarray(estimators_results.loc[:,'lsht_features'])
                    
                            y = np.asarray(estimators_results.loc[:,'recall_mean'])
                            yerror = np.asarray(estimators_results.loc[:,'recall_std'])
                             
                            lsh_features_list = lsh_features_list.union(set(x))
                            indexing_time = np.asarray(estimators_results.loc[:,'indexing_mean_time'])
                            querying_time = np.asarray(estimators_results.loc[:,'querying_mean_time'])
                            query_extraction_mean_time = np.asarray(estimators_results.loc[:,'querying_mean_time']) - np.asarray(estimators_results.loc[:,'nns_queries_mean_time'])
                            recall_per_query_time = np.asarray(estimators_results.loc[:,'recall_mean'].divide(estimators_results.loc[:,'querying_mean_time']))
                             
                            print(lsh_features_list)
                            print('\tpermutations: ',np.asarray(estimators_results.loc[:,'lsht__n_permutations']))
                            print('\thashes (x)  : ',x)
                            print('\tmean(recall): ',y)
                            print('\tstd(recall): ',yerror)
                            print('\tindexing time: ',indexing_time)
                            print('\tquerying time: ',querying_time)
                            print('\tprecision_recall_pathes: ',estimators_results.loc[:,'precision_recall_path'])
                            
                            recalls_dict[neig].append({
                                                       'label': re.sub(r" at .*", "", sel_func).replace("<function ",''),
                                                       'x':x,
                                                       'recalls_y':y,
                                                       'recalls_yerror':yerror,
                                                       'indexing_y':indexing_time,
                                                       'querying_y':querying_time,
                                                       'documents_count':np.asarray(estimators_results.loc[:,'documents_count']),
                                                       'querying_extraction_time':query_extraction_mean_time,
                                                       'recall_per_query_time':recall_per_query_time,
                                                       'precision_recall_pathes':estimators_results.loc[:,'precision_recall_path'],
                                                       })
                except:
                    print('NNS:')
                    contenti_by_neig = contenti.groupby(['nns__n_neighbors',])
                         
                    for neig,neig_values in contenti_by_neig.groups.items():
                        print('recall@',neig,' with ',len(neig_values),' values')
            
                        estimators_results = contenti.loc[neig_values,['nns__n_neighbors','recall_mean','recall_std','indexing_mean_time','querying_mean_time']]

                        x = np.array([0])##np.asarray(estimators_results.loc[:,'lsht_features'])
                
                        y = np.asarray(estimators_results.loc[:,'recall_mean'])
                        yerror = np.asarray(estimators_results.loc[:,'recall_std'])
                         
                        lsh_features_list = lsh_features_list.union(set(x))
                        indexing_time = np.asarray(estimators_results.loc[:,'indexing_mean_time'])
                        querying_time = np.asarray(estimators_results.loc[:,'querying_mean_time'])
                        query_extraction_mean_time = np.asarray(estimators_results.loc[:,'querying_mean_time']) - np.asarray(estimators_results.loc[:,'nns_queries_mean_time'])
                        recall_per_query_time = np.asarray(estimators_results.loc[:,'recall_mean'].divide(estimators_results.loc[:,'querying_mean_time']))
                         
                        print(lsh_features_list)
                        print('\tpermutations: ',np.asarray(estimators_results.loc[:,'lsht__n_permutations']))
                        print('\thashes (x)  : ',x)
                        print('\tmean(recall): ',y)
                        print('\tstd(recall): ',yerror)
                        print('\tindexing time: ',indexing_time)
                        print('\tquerying time: ',querying_time)
                        
                        recalls_dict[neig].append({
#                                                    'label':sel_func,
                                                   'x':x,
                                                   'recalls_y':y,
                                                   'recalls_yerror':yerror,
                                                   'indexing_y':indexing_time,
                                                   'querying_y':querying_time,
                                                   'documents_count':np.asarray(estimators_results.loc[:,'documents_count']),
                                                   'querying_extraction_time':query_extraction_mean_time,
                                                   'recall_per_query_time':recall_per_query_time,
                                                   })


    '''
        to evaluate Mean Error and speedup
    '''
    baseline_name = "min_hashing"
    
    '''
        plotting the results !
    '''
    
    approaches_info = {
                            "min_hashing":('-',"d",2),
                            "minmax_hashing":('-','*',  2),
                            "justCSALowerBound_hashing":('-','+', 2),
                            "minmaxCSALowerBound_hashing":('-','o', 2),
                            "minmaxCSAFullBound_hashing":('dashed','^',2)
                        }
    approaches_legends ={
                            "min_hashing":"$Min$",
                            "minmax_hashing":"$Minmax$", 
                            "justCSALowerBound_hashing":"$JCSA_L$",
                            "minmaxCSALowerBound_hashing":"$CSA_L$",
                            "minmaxCSAFullBound_hashing":"$CSA$"
                        }

    colors = ['b', 'g', 'r', 'c', 'm', 'y']


    fontsize = 10.5
    '''
        plotting recall@k / permutation
    '''
    '''
    configuracao sem o min
    fig, ax1 = plt.subplots(figsize=(8,4))
    plt.subplots_adjust(left=0.07, right=0.71, top=0.98, bottom=0.15)
    '''    
    fig, ax1 = plt.subplots(figsize=(9,4.5))
    plt.subplots_adjust(left=0.07, right=0.71, top=0.91, bottom=0.1)

    fig.canvas.set_window_title("Recalls x signatures")
    ax1.grid(True)
    ax1.set_ylabel("Recalls")
    ax1.set_xlabel('Signatures')
    
    
    
#     lsh_features_list = lsh_features_list.union(set([0, max(lsh_features_list) + 10]))
    lsh_features_list = list(lsh_features_list)
    plt.xticks(lsh_features_list)

    ax1.set_ylim(0.0,1.1)# np.max(y)+ 0.1)
    ax1.set_xlim(0, max(lsh_features_list) + 20)
    
    # Two subplots, the axes array is 1-d
    sharedx_fig, sharedx_ax = plt.subplots(len(recalls_dict), figsize=(8,10),sharex=True,sharey=True)
    sharedx_fig.canvas.set_window_title("Elapsed time x signatures")
    plt.subplots_adjust(left=0.06, right=0.74, top=0.98, bottom=0.05)
    plt.xticks(lsh_features_list)
    sharedx_ax[-1].set_xlabel('signatures')    
    
    sorted_keys = sorted(list(recalls_dict.keys()))
    
    i = 0
    for key in sorted_keys:
        values = recalls_dict[key]
        recall_label = "recall@%d"%(key)
        recall_color = colors[i]
        i += 1
        print(recall_label, recall_color)


        '''
            plotting time / permutation
        '''
        fig1, ax2 = plt.subplots(figsize=(8,4))
        fig1.canvas.set_window_title("Elapsed time x signatures")
        plt.subplots_adjust(left=0.06, right=0.74, top=0.94, bottom=0.11)
        ax2.grid(True)
        plt.xticks(lsh_features_list)
        ax2.set_xlim(0, max(lsh_features_list) + 20)

        
        ax2.set_title("Mean Elapsed Time (seconds) to retrieve %d documents"%(key), fontweight='bold')
        ax2.set_xlabel('signatures')    
        
        
        sharedx_ax[i-1].grid(True)
        sharedx_ax[i-1].set_xlim(0, max(lsh_features_list) + 20)
        sharedx_ax[i-1].set_title("Mean Elapsed Time (seconds) to retrieve %d documents"%(key), fontweight='bold')

        for approachi in values:
            approachi_label = "%s %s"%(approaches_legends[approachi['label']], recall_label)
            print('\t',approachi_label)
            
            approachi_linedecorator, approachi_marker, approachi_linewidth = approaches_info[approachi['label']] 


            x = approachi['x']
            y = approachi['recalls_y']
            yerror = approachi['recalls_yerror']
            
            ax1.errorbar(x,y,
#                             yerror, 
#                             errorevery=100,                            
                            marker=approachi_marker,
                            linestyle='--',
#                                         lw=approach_linewidth,
                            color = recall_color,
                            alpha=0.7,
                            label=approachi_label,
                         )
        
#             ax1.set_xlim(floor(x[0]/2), x[-1]+x[0])        

            ax2.errorbar(x,approachi['indexing_y'],
                            marker=approachi_marker,
                            linestyle='--',
#                                         lw=approach_linewidth,
                            color = 'blue',
                            alpha=0.7,
                            label="%s Index $\Delta_t$"%approaches_legends[approachi['label']],
                         )

            ax2.errorbar(x,approachi['querying_y'],
                            marker=approachi_marker,
                            linestyle='--',
#                                         lw=approach_linewidth,
                            color = 'red',
                            alpha=0.7,
                            label="%s Queries $\Delta_t$"%approaches_legends[approachi['label']],
                         )
            
            sharedx_ax[i-1].errorbar(x,approachi['indexing_y'],
                            marker=approachi_marker,
                            linestyle='--',
#                                         lw=approach_linewidth,
                            color = 'blue',
                            alpha=0.7,
                            label="%s Index $\Delta_t$"%approaches_legends[approachi['label']],
                         )

            sharedx_ax[i-1].errorbar(x,approachi['querying_y'],
                            marker=approachi_marker,
                            linestyle='--',
#                                         lw=approach_linewidth,
                            color = 'red',
                            alpha=0.7,
                            label="%s Queries $\Delta_t$"%approaches_legends[approachi['label']],
                         )            
            
        #     ax2.set_ylim(np.min(time_per_permutation)- 10, np.max(time_per_permutation)+ 10)
    #     ax2.set_xlim(floor(x[0]/2), x[-1]+x[0])    
        ax2.legend(loc='center left', fontsize=fontsize
            #                                        ,handlelength=1
            #                                        ,borderpad=1
                                                    , bbox_to_anchor=(1, 0.5)
                                                   )
    
        for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] + ax2.get_xticklabels() + ax2.get_yticklabels()):
            item.set_fontsize(fontsize)  
          
        sharedx_ax[i-1].legend(loc='center left', fontsize=fontsize
            #                                        ,handlelength=1
            #                                        ,borderpad=1
                                                    , bbox_to_anchor=(1, 0.5)
                                                   )
    
        for item in ([sharedx_ax[i-1].title, sharedx_ax[i-1].xaxis.label, sharedx_ax[i-1].yaxis.label] + sharedx_ax[i-1].get_xticklabels() + sharedx_ax[i-1].get_yticklabels()):
            item.set_fontsize(fontsize)           
        fig1.savefig(os.path.join(files_root_path,'elapsed_time_x_signatures@%d.pdf'%key),format='pdf')         
    
    ax1.legend(loc='center left', fontsize=fontsize
            #                                        ,handlelength=1
            #                                        ,borderpad=1
                                                    , bbox_to_anchor=(1, 0.5)
                                                   )

    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] + ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(fontsize)     
        
    fig.savefig(os.path.join(files_root_path,'recalls_x_signatures.pdf'),format='pdf') 
    sharedx_fig.savefig(os.path.join(files_root_path,'elapsed_time_x_signatures_all_in_one.pdf'),format='pdf')  
          
    
    '''
        evaluating index size, time to create index, Mean Error (recall) and speedup from baseline
    '''
    
    fig, ax1 = plt.subplots(figsize=(9,4.5))
    plt.subplots_adjust(left=0.07, right=0.86, top=0.98, bottom=0.10)

    fig.canvas.set_window_title("Throughput x signatures")
    ax1.grid(True)
    ax1.set_ylabel("Documents per Hour")
    ax1.set_xlabel('Signatures')
    
#     lsh_features_list = lsh_features_list.union(set([0, max(lsh_features_list) + 10]))
    lsh_features_list = list(lsh_features_list)
    plt.xticks(lsh_features_list)

#     ax1.set_ylim(0.0,1.1)# np.max(y)+ 0.1)
    ax1.set_xlim(0, max(lsh_features_list) + 20)
    
    approaches_pr_files = {}
    for key in sorted_keys:
        values = recalls_dict[key]

        results_dataframe = None
        
        for approachi in values:
    #         print('='*80)
    #         print(approachi_label)
            approachi_label = approaches_legends[approachi['label']].replace('$','')
            resultsi_dict = {}
            
            approaches_pr_files[approachi_label] = approachi['precision_recall_pathes']
            resultsi_dict["signatures"] = approachi['x']
            resultsi_dict["%sIndexSize"%(approachi_label)] = np.multiply(approachi['documents_count'],approachi['x'])
            resultsi_dict["%sIndexTotalTime(sec)"%(approachi_label)] = approachi['indexing_y']*approachi['documents_count'][0]
            resultsi_dict["%sIndexTotalTime(hours)"%(approachi_label)] = approachi['indexing_y']*approachi['documents_count'][0]/3600 
            resultsi_dict["%sIndexTotalTime(days)"%(approachi_label)] = approachi['indexing_y']*approachi['documents_count'][0]/(3600*24)

            resultsi_dict["%sRecMean"%(approachi_label)] = approachi['recalls_y']
            resultsi_dict["%sRecStd"%(approachi_label)] = approachi['recalls_yerror']
            resultsi_dict["%sQueryMeanTime"%(approachi_label)] = approachi['querying_y']
            
            resultsi_dict["%sQueryExtractionMeanTime"%(approachi_label)] = approachi['querying_extraction_time']
            resultsi_dict["%sIndexThroughput(sec)"%(approachi_label)] = approachi['documents_count'][0] / resultsi_dict["%sIndexTotalTime(sec)"%(approachi_label)]
            resultsi_dict["%sIndexThroughput(hours)"%(approachi_label)] = approachi['documents_count'][0] / resultsi_dict["%sIndexTotalTime(hours)"%(approachi_label)]
            
            resultsi_dict["%sRecMeanPerQueryTime"%(approachi_label)] = approachi['recall_per_query_time']
            
            dt = pd.DataFrame.from_dict(resultsi_dict)
            dt = dt.set_index("signatures")
            
            if key == sorted_keys[0]:
                approachi_linedecorator, approachi_marker, approachi_linewidth = approaches_info[approachi['label']]                 
                ax1.errorbar(approachi['x'],resultsi_dict["%sIndexThroughput(hours)"%(approachi_label)],
    #                             yerror, 
    #                             errorevery=100,                            
                    marker=approachi_marker,
                    linestyle='--',
    #                                         lw=approach_linewidth,
                    color = 'blue',
                    alpha=0.7,
                    label=approaches_legends[approachi['label']],
                 )

            
            if results_dataframe is None:
                results_dataframe = dt
            else:
                results_dataframe = pd.concat([results_dataframe, dt], axis=1, join='outer')#on='signatures', how='outer', indicator=True)
    #         print(results_dataframe)
    #         print('-'*80)
    #         input()
        baseline_label = approaches_legends[baseline_name].replace('$','')
#         results_dataframe.loc[:,"%sRecMean"%(baseline_label)]
        
        for approachi in values:
    #         print('='*80)
    #         print(approachi_label)
            approachi_label = approaches_legends[approachi['label']].replace('$','')
            if approachi_label == "Min":
                continue
            results_dataframe.loc[:,"%sMe"%(approachi_label)] = results_dataframe.loc[:,"%sRecMean"%(baseline_label)] - results_dataframe.loc[:,"%sRecMean"%(approachi_label)]
            results_dataframe.loc[:,"%sSpeedup"%(approachi_label)] = results_dataframe.loc[:,"%sQueryMeanTime"%(baseline_label)].divide(results_dataframe.loc[:,"%sQueryMeanTime"%(approachi_label)])
            results_dataframe.loc[:,"%sQueryExtractionSpeedup"%(approachi_label)] = results_dataframe.loc[:,"%sQueryExtractionMeanTime"%(baseline_label)].divide(results_dataframe.loc[:,"%sQueryExtractionMeanTime"%(approachi_label)])

            approachi_rpd,me_improvement = [],[]
            approachi_wilcoxon,approachi_wilcoxon_pratt = [],[]
            approachi_shapiro = []
            
            for baseline_path,app_path in zip(approaches_pr_files[baseline_label].tolist(), approaches_pr_files[approachi_label].tolist()):
                b_rec = hdf_to_sparse_matrix('recalls', baseline_path)[:,-1]
                appro_rec = hdf_to_sparse_matrix('recalls', app_path)[:,-1]
                
                elementwise_me = 2*(b_rec-appro_rec)
                elementwise_sum = b_rec+appro_rec
                nz_ids = elementwise_sum.nonzero()
                elementwise_rpd = np.zeros(b_rec.shape)
                elementwise_rpd[nz_ids] = np.divide(elementwise_me[nz_ids],elementwise_sum[nz_ids])
                
                approachi_rpd.append(elementwise_rpd)
                approachi_shapiro.append(shapiro(appro_rec.T.toarray()[0])[1])
                
                me_improvement.append(len(np.nonzero(elementwise_rpd <= 0)[0])/elementwise_rpd.shape[0])
                if approachi_label != baseline_label:
                    h_test = b_rec - appro_rec
                    approachi_wilcoxon.append(wilcoxon(h_test.T.toarray()[0])[1])
                    approachi_wilcoxon_pratt.append(wilcoxon(h_test.T.toarray()[0],zero_method="pratt")[1])
                    del h_test
                else:
                    approachi_wilcoxon.append(1)
                    approachi_wilcoxon_pratt.append(1)

                del b_rec, appro_rec
                    
            approachi_rpd = np.hstack(approachi_rpd)
            results_dataframe.loc[:,"%sMe_std"%(approachi_label)] = (approachi_rpd/2).std(axis=0)
            results_dataframe.loc[:,"%sRPD_mean"%(approachi_label)] = approachi_rpd.mean(axis=0)
            results_dataframe.loc[:,"%sRPD_std"%(approachi_label)] = approachi_rpd.std(axis=0)
            results_dataframe.loc[:,"%sRPD_min"%(approachi_label)] = approachi_rpd.min(axis=0)
            results_dataframe.loc[:,"%sRPD_max"%(approachi_label)] = approachi_rpd.max(axis=0)
            results_dataframe.loc[:,"%s_error_improvement"%(approachi_label)] = me_improvement
            results_dataframe.loc[:,"%s_wilcoxon"%(approachi_label)] = approachi_wilcoxon
            results_dataframe.loc[:,"%s_wilcoxon_pratt"%(approachi_label)] = approachi_wilcoxon_pratt
            results_dataframe.loc[:,"%s_shapiro"%(approachi_label)] = approachi_shapiro
            
            del me_improvement,approachi_wilcoxon,approachi_wilcoxon_pratt
            
            print(results_dataframe.loc[:,["%sMe"%(approachi_label),"%sMe_std"%(approachi_label),"%sRPD_mean"%(approachi_label),
                                           "%sRPD_std"%(approachi_label),"%sRPD_min"%(approachi_label),"%sRPD_max"%(approachi_label),"%s_error_improvement"%(approachi_label),
                                           "%s_wilcoxon"%(approachi_label), "%s_wilcoxon_pratt"%(approachi_label)
                                           ]])
            
            approachi_rpd = pd.DataFrame(approachi_rpd, columns=results_dataframe.index.values).round(5)
            print(approachi_rpd.loc[0:2,:])
            print(approachi_rpd.T.index.values)
            print(approachi_rpd.T.loc[:,0:2])
            
            heatmap_dict = {}
            xtics = [0]
            for k in approachi_rpd.columns.values:
                approachi_rpd.loc[:,"%d_keys"%k] = int(k)
                a = approachi_rpd.loc[:,k].value_counts()
                if not "RPD" in heatmap_dict:
                    heatmap_dict["RPD"] = np.array([])
                    heatmap_dict["RPD_count"] = np.array([])
                    heatmap_dict["k"] = np.array([])
                
                heatmap_dict["RPD"] = np.concatenate((heatmap_dict["RPD"], a.index.values))
                heatmap_dict["RPD_count"] = np.concatenate((heatmap_dict["RPD_count"], a.tolist()))
                heatmap_dict["k"] = np.concatenate((heatmap_dict["k"], k*np.ones((len(a.tolist())))))
                xtics.append(k)

            del approachi_rpd
            
            xtics.append(xtics[-1]+100)
            xtics = np.array(xtics)

            heatmap_df = pd.DataFrame.from_dict(heatmap_dict)
            fig, ax1 = plt.subplots(figsize=(5,3))
            plt.subplots_adjust(left=0.12, right=0.99, top=0.92, bottom=0.1)
            plt.title("$%s$"%approachi_label)

            heatmap_df.plot.scatter(x='k', y="RPD", s = heatmap_df.loc[:,'RPD_count']/12, marker='o',alpha=0.8 , ax=ax1)
            plt.xlabel("assinaturas(k)")
            plt.ylabel("Diferença Percentual Relativa")
            ax1.set_ylim(-3,3)
            plt.xticks(xtics, fontsize=12)

            plt.savefig("RDP_%d_%s"%(key,approachi_label))
#             print(results_dataframe.loc[:,["%s_me"%(approachi_label), "%s_speedup"%(approachi_label)]])
        print("%d resultant dataframe:"%key)
#         print(results_dataframe.to_csv(sep = '\t'))
        results_dataframe.to_csv(os.path.join(files_root_path,'results_summary_%d.csv'%(key)),sep = '\t')    
    
        
    ax1.legend(loc='center left', fontsize=fontsize
            #                                        ,handlelength=1
            #                                        ,borderpad=1
                                                    , bbox_to_anchor=(1, 0.5)
                                                   )

    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] + ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(fontsize)     
        
fig.savefig(os.path.join(files_root_path,'throughput_x_signatures.pdf'),format='pdf') 
plt.show()
