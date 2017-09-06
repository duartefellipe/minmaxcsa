'''
Created on 24 de mai de 2016

@author: Fellipe

'''
import numpy as np
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os.path
from pprint import pprint
import operator

if __name__ == "__main__":
    
    plt.rcParams['font.size'] = 16    
    
#     results_file_name = "psa[100, 200, 400, 800]x100"
#     results_file_name = "psa[100, 200, 400, 800]x1"
#     results_file_name = "meter[100]x100"
#     results_file_name = "meter[200]x100"
#     results_file_name = "meter[400]x100"
#     results_file_name = "meter[800]x100"
#     results_file_name = "pan11[100, 200, 400, 800]x100"
#     results_file_name = "psa[100]x100"
#     results_file_name = "psa[200]x100"
#     results_file_name = "psa[400]x100"
#     results_file_name = "psa[800]x100"

#     results_file_name = "pan11[100]x100"
#     results_file_name = "pan11[200]x100"
#     results_file_name = "pan11[400]x100"
    results_file_name = "psa[10]x1"


    path = "./"
    permutations_to_display = {
#                                 100 : [], 
#                                 200 : [], 
#                                 400 : [], 
                                10 : [], 
                               }

    with open(os.path.join(path,"pjs_approaches_files_%s.pkl"%(results_file_name)),'rb') as f:
        results_lists = pickle.load(f)
        results = {}
        for resulti in results_lists:
            
            permutation_key, approach_name, result_type, result_filename = resulti
#             print(permutation_key, approach_name, result_type, result_filename) 

            if permutation_key in permutations_to_display.keys():
                try:
                    results[permutation_key]
                except:
                    results[permutation_key] = {}
    
                try:
                    results[permutation_key][approach_name]
                except:
                    results[permutation_key][approach_name] = {}
    
                with open(os.path.join(path,result_filename,), 'rb') as f1:
                    results[permutation_key][approach_name][result_type] = results_lists = pickle.load(f1)
    #                 try:
    #                     print(results[permutation_key][approach_name][result_type].shape)
    #                 except:
    #                     print(result_type,' = ')
    #                     print(results[permutation_key][approach_name][result_type])
        del results_lists
#     print(results)
#     exit()
    
    results_ae  = []
#     results_se  = []
#     results_rse = []
    results_names = []
    approaches_legends ={
                            "minmax            ":"$Minmax$", 
#                             "sqrt(max^2-min^2) ":"$T_{prop}$",
                            "minmaxCSA_L         ":"$CSA_L$",
                            "minmaxCSA       ":"$CSA$",
                            }
    approaches_to_display = approaches_legends.keys()
    
    boxplot_ylim = 0
    
    for permutation_count in sorted(results.keys()):
        print("%d permutations:"%(permutation_count))
        for approach_name, ap_dict in sorted(results[permutation_count].items(), key=operator.itemgetter(0)):
            if approach_name in approaches_to_display:
                permutations_to_display[permutation_count].append(len(results_ae))
                squared_errors = np.array(ap_dict['errors'])**2
                absolute_errors = np.sqrt(squared_errors)
                print("[%s] MSE:%2.2e[+/- %2.2e] MAE:%2.2e[+/- %2.2e](in %4.2fs) "%(
                    approach_name, 
                    np.mean(squared_errors), np.std(squared_errors),
                    np.mean(absolute_errors), np.std(absolute_errors),
                    np.array(ap_dict['approach_time']).mean()))
                results_ae.append(np.abs(ap_dict['errors']))
#                 results_se.append(np.array(ap_dict['errors'])**2)
#                 results_rse.append(np.sqrt(np.array(ap_dict['errors'])**2))
                
                boxplot_ylim = max(boxplot_ylim, np.mean(absolute_errors))
                print('boxplot_ylim:',boxplot_ylim)
    
                results_names.append(approaches_legends[approach_name])
#                 results_names.append("%d %s perm. (in %4.2fs[+/- %2.2f])"%(permutation_count,approaches_legends[approach_name],np.array(ap_dict['approach_time']).mean(),np.array(ap_dict['approach_time']).std()))
            del ap_dict['errors']
    del results
    for window_title,ylabel, file_name,results_to_plot in [
                                                            ('Mean Absolute Errors','MAE','absolute_errors',results_ae),
#                                                             ('Mean Squared Errors','Mean Squared Error','squared_errors',results_se),
#                                                             ('Root Mean Squared Errors','Root Mean Squared Error','root_squared_errors',results_rse),
                                                           ]:         

        for permutationi_label,permutationi_values in permutations_to_display.items():
            print("Evaluating %s (%d permutations)!"%(window_title,permutationi_label))
            
            fig, ax1 = plt.subplots(figsize=(4,3))
            fig.canvas.set_window_title(window_title)
            plt.subplots_adjust(left=0.18, right=0.98, top=0.9, bottom=0.1)
              
            # Hide these grid behind plot objects
            ax1.set_axisbelow(True)
            ax1.set_title('%s (k = %d)'%(ylabel,permutationi_label))
        #     ax1.set_xlabel('Distribution')
#             ax1.set_ylabel(ylabel)
#             ax1.set_ylim(-0.1, np.array(results_to_plot).max()+0.1)
            ax1.set_ylim(-0.01, boxplot_ylim+0.1)
          
            # multiple box plots on one figure
            plt.boxplot([results_ae[i] for i in permutationi_values], showmeans=True, showfliers=False)
            xtickNames = plt.setp(ax1, xticklabels=results_names)
          
#             plt.setp(xtickNames, rotation=-90, fontsize=15)    
            plt.setp(xtickNames, fontsize=15)    
          
#             results_file_name = "psa"
#             plt.savefig('%d-%s_%s.pdf'%(permutationi_label,file_name,results_file_name), format='pdf')
            plt.savefig('%d-%s_%s'%(permutationi_label,file_name,results_file_name))

#     plt.show()
