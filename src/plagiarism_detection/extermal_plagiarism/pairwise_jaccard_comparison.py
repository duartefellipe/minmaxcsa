'''
Created on 24 de mai de 2016

@author: Fellipe

'''
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from time import time
from math import floor

from locality_sensitive_hashing import pairwise_jaccard_similarity,minmax_hashing,minmaxCSALowerBound_hashing, minmaxCSAFullBound_hashing, justCSALowerBound_hashing, justCSAFullBound_hashing, min_hashing
import pickle
from datasets.extractors import meter_extractor, short_plagiarised_answers_extractor, pan_plagiarism_corpus_2011_extractor, pan_plagiarism_corpus_2010_extractor


def prepare_jaccard_sim_results(results_dict, perm_repetition, perm_count, perm_true_jaccard, approach_name, approach_time,approach_jaccard ):
    try:
        perm_dict = results_dict[perm_count]
    except KeyError:
        results_dict[perm_count] = {}
        perm_dict = results_dict[perm_count]

    try:
        approach_dict = perm_dict[approach_name]
    except KeyError:
        perm_dict[approach_name] = {'approach_time':[],'errors':[]}
        approach_dict = perm_dict[approach_name]
    
    approach_dict['perm_repetition'] = perm_repetition
    approach_dict['approach_time'].append(approach_time)
    approach_dict['errors'].append(approach_jaccard - perm_true_jaccard)
    


if __name__ == "__main__":
    
    '''
        dataset extraction
    '''
#     corpus_name ="meter"
#     corpus_name ="psa"
    corpus_name ="pan11"
#     corpus_name ="pan10"
    if corpus_name == "meter" or corpus_name == "psa":
        if corpus_name == "meter":
            dataset_documents,dataset_target,_,_,dataset_encoding = meter_extractor.load_as_pairs()
            nonzero_indexes = np.argwhere(dataset_target)
        elif corpus_name == "psa":
            corpus_name, (dataset_documents,dataset_target,_,_,dataset_encoding) = "psa", short_plagiarised_answers_extractor.load_as_pairs()
            nonzero_indexes = range(dataset_documents.shape[0])

#         print(dataset_documents.shape)
        
        documents,queries = [],[] 
        for i in nonzero_indexes:
            queries.append(dataset_documents[i,0])
            documents.append(dataset_documents[i,1])
            if corpus_name == "meter":
                queries[-1] = queries[-1].flatten()[0,0] 
                documents[-1] = documents[-1].flatten()[0,0] 
            
        del dataset_documents,dataset_target,dataset_encoding    
    else:
        if corpus_name == "pan11" or corpus_name == "pan10":
            if corpus_name == "pan11":
                queries_, doc_index_, dataset_target, dataset_encoding = pan_plagiarism_corpus_2011_extractor.load_as_ir_task()
            else:
                queries_, doc_index_, dataset_target, dataset_encoding = pan_plagiarism_corpus_2010_extractor.load_as_ir_task()
            nonzero_indexes = np.argwhere(dataset_target)
#             print(dataset_target)
#             print(nonzero_indexes)
#             print(dataset_target.shape,len(nonzero_indexes))

            documents,queries = [],[] 
            for nzi in nonzero_indexes[:1000]:
#                 print(nzi)
                queries.append(queries_.loc[nzi[0],'content'])
                documents.append(doc_index_.loc[nzi[1],'content'])
            
            del queries_, doc_index_,dataset_target,dataset_encoding
    
#     print(queries[-1])
#     print(len(queries),len(documents))
#     print(documents[0])
#     exit()
    
    
    '''
        using scikit-learn : tokenization
    '''    
    vectorizer = CountVectorizer(binary=True,min_df=1,ngram_range=(1,3))
    all_fingerprints = vectorizer.fit_transform(queries+documents, None).T
#     vocabulary_indexes = [di for di in vectorizer.vocabulary_.values()]
    print("%s all_fingerprints: "%(corpus_name),all_fingerprints.shape)
    
    true_jaccard_sim = pairwise_jaccard_similarity(set_per_row = np.vstack([i*all_fingerprints[i,:].toarray() for i in range(all_fingerprints.shape[0])]).T)
    true_jaccard_sim_mean, true_jaccard_sim_std = true_jaccard_sim.mean(), true_jaccard_sim.std()
    print('true_jaccard_sim (mean,std) =(',true_jaccard_sim_mean,',',true_jaccard_sim_std,')')

    results = {}
    '''
        using scikit-learn : permutation
            each permutation has one term-document matrix
    '''    
#     permutation_repetition = 1
    permutation_repetition = 100 
    permutation_count_list = [100*(2**i) for i in range(0,4)]
#     permutation_count_list = [10]
    results_file_name = "%s%sx%d"%(corpus_name,str(permutation_count_list),permutation_repetition)

    for permutation_count in permutation_count_list:
        for permutation_repetitioni in range(permutation_repetition):
            indexes_permutations = [shuffle([i+1 for i in range(all_fingerprints.shape[0])]) for j in range(permutation_count)]
            
            # approach: name, selection size, selection function, permutation count
            approaches = [
                        ("min               ", 1, min_hashing, permutation_count),
                        ("minmax            ", 2, minmax_hashing, floor(permutation_count/2)),
#                         ("minmaxxor         ", 1, minmaxxor_hashing, floor(permutation_count/2)),
                        ("minmaxCSA_L         ", 3,minmaxCSALowerBound_hashing, floor(permutation_count/3)),
                        ("minmaxCSA       ", 4, minmaxCSAFullBound_hashing, floor(permutation_count/4)),
#                         ("justCSA_L         ", 1, justCSALowerBound_hashing, floor(permutation_count/1)),
#                         ("justCSA       ", 2, justCSAFullBound_hashing, floor(permutation_count/2)),

                        ]
            
            for approachi in approaches:
                approach_name, selection_size, selection_function, approach_permutation_count = approachi
                approach_finger = np.empty((all_fingerprints.shape[1],selection_size*approach_permutation_count),np.int)
                approach_time = np.zeros((all_fingerprints.shape[1],))
    
                for j in range(all_fingerprints.shape[1]):
                    t0 = time() 
                    current_document = all_fingerprints[:,j]
                    
                    for i in range(approach_permutation_count):
                        pi_results = selection_function(current_document, indexes_permutations[i])
                        approach_finger[j,selection_size*i:selection_size*(i+1)] = pi_results
#                         print(approach_name, selection_size,'%d/%d'%(i,approach_permutation_count), pi_results.shape)
        
                    approach_time[j] = time() - t0
                    del current_document
                
                approach_jaccard_sim = pairwise_jaccard_similarity(set_per_row = approach_finger)
                prepare_jaccard_sim_results(approach_name=approach_name, 
                                            approach_time=approach_time.sum(), 
                                            approach_jaccard=approach_jaccard_sim, 
                                            results_dict=results, 
                                            
                                            perm_repetition=permutation_repetition, 
                                            perm_count=permutation_count, 
                                            perm_true_jaccard=true_jaccard_sim
                                            )

                del approach_finger

            print("%d permutations[%d/%d]:"%(permutation_count,permutation_repetitioni,permutation_repetition))
            for approach_name, ap_dict in  results[permutation_count].items():
                squared_errors = ap_dict['errors'][permutation_repetitioni]**2
                absolute_errors = np.sqrt(squared_errors)
                print("[%s] MSE:%2.2e[+/- %2.2e] MAE:%2.2e[+/- %2.2e](in %4.2fs) "%(
                        approach_name, 
                        np.mean(squared_errors), np.std(squared_errors),
                        np.mean(absolute_errors), np.std(absolute_errors),
                        ap_dict['approach_time'][permutation_repetitioni]))


            
    results_names = []
    results_json = ["approach, k, MSE, MSE(std), MAE, MAE(std), time, time(std)"]
    
    approaches_files_json = []
             
    for permutation_count in results.keys():
        print("%d permutations:"%(permutation_count))
        for approach_name, ap_dict in  results[permutation_count].items():
            squared_errors = np.array(ap_dict['errors'])**2
            absolute_errors = np.sqrt(squared_errors)
            print("[%s] MSE:%2.2e[+/- %2.2e] MAE:%2.2e[+/- %2.2e](in %4.2fs) "%(
                approach_name, 
                np.mean(squared_errors), np.std(squared_errors),
                np.mean(absolute_errors), np.std(absolute_errors),
                np.array(ap_dict['approach_time']).mean()))
 
            results_names.append("%d %s perm. (in %4.2fs[+/- %2.2f])"%(permutation_count,approach_name.replace(' ',''),np.array(ap_dict['approach_time']).mean(),np.array(ap_dict['approach_time']).std()))
            results_json.append("%s,%d,%2.4e,%2.4e,%2.4e,%2.4e,%2.4fs,%2.4fs"%(approach_name,permutation_count,np.mean(squared_errors),np.std(squared_errors),np.mean(absolute_errors),np.std(absolute_errors),np.array(ap_dict['approach_time']).mean(),np.array(ap_dict['approach_time']).std()))

            for keyi, valuei in  ap_dict.items():
                filename_id = len(approaches_files_json)
                temp_filename = "pjs_%s_%d.pkl"%(results_file_name,filename_id)            
                approaches_files_json.append((permutation_count, approach_name, keyi,temp_filename))

                with open(temp_filename,'wb') as f:
                    pickle.dump(valuei,f)

                del valuei
            
            del ap_dict
            
#         del results[permutation_count]
    
    del results
    
    with open("pjs_%s.csv"%(results_file_name),'w') as f:
        f.write('\n'.join(results_json)) 
        del results_json       
 
    with open("pjs_approaches_files_%s.pkl"%(results_file_name),'wb') as f:
        pickle.dump(approaches_files_json,f)
