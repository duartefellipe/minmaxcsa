'''
Created on 24 de jan de 2017

@author: Fellipe Duarte
'''
import numpy as np
from time import time
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.grid_search import ParameterGrid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals.joblib.parallel import delayed, Parallel

from datasets.extractors import short_plagiarised_answers_extractor, pan_plagiarism_corpus_2010_extractor, pan_plagiarism_corpus_2011_extractor
from locality_sensitive_hashing import LSHTransformer, minmax_hashing, LSHIINearestNeighbors, InvertedIndexNearestNeighbors, PBINearestNeighbors, min_hashing, minmaxCSALowerBound_hashing,minmaxCSAFullBound_hashing, justCSALowerBound_hashing, justCSAFullBound_hashing
from scipy.sparse.lil import lil_matrix
from scipy.sparse.csr import csr_matrix
from scipy.sparse.base import issparse
import pickle
import os.path
from scipy import vstack
from datetime import datetime
 
def encode_dataframe_content(dataframe_contenti, encoding):
    return dataframe_contenti.encode(encoding)    

def parameters_gridlist_dataframe(parameters_dict):
    '''
        parameters_dict is a (sklearn) Pipeline parameters dict 
        
        returns a (pandas) Dataframe containing a list of parameters' grid from parameters_dict. 
    '''
    grid_list = list(ParameterGrid(parameters_dict))

    df = pd.DataFrame(grid_list)

#     print(grid_list)    
#     print(df.to_dict("records"))
#     print(df.describe)
#     exit()
    
    return df

def h5_results_filename(dataset_name,result_type,dataframe_pos):
    h5_file_path = "%s_%s_%d_results.h5"%(dataset_name,result_type,dataframe_pos)
    return h5_file_path

def sparse_matrix_to_hdf(sparse_matrix,name_to_store,hdf_file_path):
    nonzero_indices = np.nonzero(sparse_matrix>0)
    if len(nonzero_indices[0]) == 0:
            raise Exception("can't store empty sparse matrix!")
    
    if issparse(sparse_matrix):
        if sparse_matrix.__class__ is lil_matrix:
            nonzero_values = sparse_matrix.tocsr()[nonzero_indices].A1
        else:
            nonzero_values = lil_matrix(sparse_matrix).tocsr()[nonzero_indices].A1
    else:
        nonzero_values = np.array(sparse_matrix[nonzero_indices])

#     print(sparse_matrix.__class__,'=',name_to_store,sparse_matrix.shape,len(nonzero_values))
        
    matrix_dataframe = pd.DataFrame({
                               "row_indexes":nonzero_indices[0],
                               "col_indexes":nonzero_indices[1],
                               "data":nonzero_values})
    matrix_shape_series = pd.Series(sparse_matrix.shape)
    
    matrix_dataframe.to_hdf(hdf_file_path, name_to_store)
    matrix_shape_series.to_hdf(hdf_file_path, "%s_shape"%name_to_store)
    
    del nonzero_indices,nonzero_values,matrix_dataframe,matrix_shape_series

def hdf_to_sparse_matrix(name_to_load,hdf_file_path):
    matrix_dataframe = pd.read_hdf(hdf_file_path, name_to_load)
    matrix_shape_series = pd.read_hdf(hdf_file_path, "%s_shape"%name_to_load)

    col = matrix_dataframe.loc[:,'col_indexes'].values.tolist()
    row = matrix_dataframe.loc[:,'row_indexes'].values.tolist()
    data = matrix_dataframe.loc[:,'data'].values.tolist()

#     print(np.array(col).shape)
#     print(np.array(row).shape)
#     print(np.array(data).shape)
#     print(matrix_shape_series.values.tolist())

    sparse_matrix = csr_matrix((data, (row, col)), shape=matrix_shape_series.values.tolist())

    del col,row,data, matrix_dataframe, matrix_shape_series
    
    return sparse_matrix

def tokenize_by_parameters(documents,queries,target,dataset_name, cv_parameters_dataframe_line,cv_parameters_dataframe_line_index,encoding):
    '''
        tokenize and store results 
    '''
    start_time = time()
    file_path = h5_results_filename(dataset_name, 'cv', cv_parameters_dataframe_line_index) #"%s_cv_%d_results.h5"%(dataset_name,cv_parameters_dataframe_line_index)
    
    if os.path.exists(file_path):
        print(file_path,' already exists!')
    else:
        print('start:',file_path)
        pipe_to_exec = Pipeline([("cv",TfidfVectorizer(binary=True, encoding=dataset_encoding))])
        parameters = cv_parameters_dataframe_line.to_dict()
        parameters['cv__encoding'] = encoding
    
        pipe_to_exec.set_params(**cv_parameters_dataframe_line)
    
        t0 = time()
        td_documents = pipe_to_exec.fit_transform(documents, None)
        documents_elapsed_time = (time()-t0) / td_documents.shape[0]
        sparse_matrix_to_hdf(td_documents,'documents',file_path)
        print('documents:',td_documents.shape)
        del td_documents
    
        t0 = time()
        td_queries = pipe_to_exec.transform(queries)
        queries_elapsed_time = (time()-t0) / td_queries.shape[0]
        sparse_matrix_to_hdf(td_queries,'queries',file_path)
        print('queries:',td_queries.shape)
        del td_queries
    
        sparse_matrix_to_hdf(target,'targets',file_path)
        
        time_dataframe = pd.DataFrame({
                   'documents_mean_time' : [documents_elapsed_time],
                   'queries_mean_time' : [queries_elapsed_time],
                   })
        time_dataframe.to_hdf(file_path.replace('results.h5', 'time.h5'), 'time_dataframe')
    
        del time_dataframe    
        with open(file_path.replace('results.h5', 'vocabulary.pkl'),'wb') as f:
            pickle.dump(pipe_to_exec.steps[0][1].vocabulary_,f)
        
        del pipe_to_exec
         
        print('end:',file_path, "in %4.2f s"%(time()-start_time)) 

        d = hdf_to_sparse_matrix('documents', file_path)
        
        for i in range(d.shape[0]):
            a = d[i,:].sum()
            if a ==0 :
                print(i,'/',d.shape[0])
                input('vazio!')
                print(documents[i])

        d = hdf_to_sparse_matrix('queries', file_path)
        
        for i in range(d.shape[0]):
            a = d[i,:].sum()
            if a ==0 :
                print(i,'/',d.shape[0])
                input('vazio!')
                print(queries[i])
        

def lsh_transform(dataset_name, lsht_parameters_dataframe_line, lsth_parameters_dataframe_line_index, encoding):
    indexi  = lsht_parameters_dataframe_line['input__filename_index']
    source_file_path = h5_results_filename(dataset_name, 'cv', indexi)
    file_path = h5_results_filename(dataset_name, 'lsht', lsth_parameters_dataframe_line_index)
    
    if os.path.exists(file_path) :
        print(file_path,' already exists!')
    else:    
        pipe_to_exec = Pipeline([('lsht',LSHTransformer(n_permutations=1,n_jobs=0))])
        pipe_to_exec.set_params(**lsht_parameters_dataframe_line.drop('input__filename_index'))
        
        print(lsht_parameters_dataframe_line.drop('input__filename_index'))
        
        
        d = hdf_to_sparse_matrix('documents', source_file_path)

        d_line,d_time = pipe_to_exec.fit_transform(d, None)
        sparse_matrix_to_hdf(d_line,'documents',file_path) 
        sparse_matrix_to_hdf(d_time,'documents_time',file_path.replace('results.h5', 'time.h5'))
        print(d_line.shape, "in %f[+/-%4.2f] s"%(d_time.mean(),d_time.std()))
        
        del d,d_line,d_time
        
        q = hdf_to_sparse_matrix('queries', source_file_path)

        q_line,q_time = pipe_to_exec.transform(q)
        sparse_matrix_to_hdf(q_line,'queries',file_path)
        sparse_matrix_to_hdf(q_time,'queries_time',file_path.replace('results.h5', 'time.h5'))
        print(q_line.shape, "in %f[+/-%4.2f] s"%(q_time.mean(),q_time.std()))
        
        del q,q_line,q_time
        
        t = hdf_to_sparse_matrix('targets', source_file_path)
        sparse_matrix_to_hdf(t,'targets',file_path)
        del t

def __nearest_neighbors_search(pipe_to_exec,source_file_path,file_path):
    '''
        runs "pipe_to_exec" nearest neighbors search estimator
            
        parameters: 
        
            * source_file_path : hdf file in which input documents, queries and targets are stored
            * file_path: hdf filename where nns results will be stored
    '''
        
#     print(linei.describe)
        
    d = hdf_to_sparse_matrix('documents', source_file_path)
    pipe_to_exec.fit(d, None)
    d_mean_time = pipe_to_exec.steps[0][1].fit_time
         
    print("fitted in %f s"%(d_mean_time))
        
    del d
        
    q = hdf_to_sparse_matrix('queries', source_file_path)
    d_indices,qd_distances,q_mean_time = pipe_to_exec.transform(q)
        
#     print("mean retrieval time %f s"%(q_mean_time))
        
    time_dataframe = pd.DataFrame({
               'documents_mean_time' : [d_mean_time],
               'queries_mean_time' : [q_mean_time],
            })
        
    '''
        storing nearest neighbors search results
    '''
    time_dataframe.to_hdf(file_path.replace('results.h5', 'time.h5'), 'time_dataframe')
    sparse_matrix_to_hdf(d_indices,'retrieved_docs',file_path)
    sparse_matrix_to_hdf(lil_matrix(qd_distances),'qd_distances',file_path)
        
    del q, d_mean_time, q_mean_time, qd_distances, time_dataframe
        
    '''
        Evaluating results in terms of Precision, Recalls and MAP.
    '''

    t = hdf_to_sparse_matrix('targets', source_file_path)
        
    retrieved_relevants = []
    for q_index in range(d_indices.shape[0]):
        q_retrieved_relevants = np.cumsum(t[q_index,d_indices[q_index,:]].A,axis=1)
        retrieved_relevants.append(q_retrieved_relevants)
        
    retrieved_relevants = vstack(retrieved_relevants)
        
    '''
        broadcasting
    '''        
    approachi_recalls = np.divide(retrieved_relevants,np.matrix(t.sum(axis=1)))
    ranking_sum = np.multiply(np.ones(retrieved_relevants.shape),np.matrix(range(1,retrieved_relevants.shape[1]+1)))
    approachi_precisions = np.divide(retrieved_relevants,ranking_sum)
        
    average_precision = np.zeros((d_indices.shape[0],1))
    for q_index in range(d_indices.shape[0]):
        relevants_precision = np.multiply(approachi_precisions[q_index,:],t[q_index,d_indices[q_index,:]].A)
        average_precision[q_index,0] = relevants_precision.mean(axis=1)
#         print(q_index,'.MAP =',average_precision[q_index,0])    

#     print(t.sum(axis=1))
#     print(retrieved_relevants)
    del d_indices, retrieved_relevants

#     print("MAP=",average_precision.mean(),average_precision.std(),'precision.sum=',average_precision.sum())
#     print("recalls.sum = ",approachi_recalls.sum(),'| mean = ',approachi_recalls.sum()/(approachi_recalls.shape[0]*approachi_recalls.shape[1]))
        
    for to_store,to_store_name in [(approachi_precisions,'precisions'),(approachi_recalls,'recalls'),(average_precision,'average_precisions')]:
        if not issparse(to_store):
            to_store = csr_matrix(to_store)
        sparse_matrix_to_hdf(to_store,to_store_name,file_path.replace('results','results_evaluation'))
        
        del to_store

def lsh_nearest_neighbors_search(dataset_name, lshnns_parameters_dataframe_line, lshnns_parameters_dataframe_line_index, encoding):
    indexi = lshnns_parameters_dataframe_line['input__filename_index']
    source_file_path = h5_results_filename(dataset_name, 'lsht', indexi)
    file_path = h5_results_filename(dataset_name, 'lshnns', lshnns_parameters_dataframe_line_index)
    
    print(lshnns_parameters_dataframe_line)

    if os.path.exists(file_path):
#     if os.path.exists(file_path):
        print(file_path,' already exists!')
    else:    
        pipe_to_exec = Pipeline([('lshnns',LSHIINearestNeighbors(n_neighbors=2))])
        pipe_to_exec.set_params(**lshnns_parameters_dataframe_line.drop('input__filename_index'))
        
        __nearest_neighbors_search(pipe_to_exec, source_file_path, file_path)

def pbinearest_neighbors_search(dataset_name, nns_parameters_dataframe_line, nns_parameters_dataframe_line_index, encoding):
    indexi = nns_parameters_dataframe_line['input__filename_index']
    source_file_path = h5_results_filename(dataset_name, 'cv', indexi)
    file_path = h5_results_filename(dataset_name, 'pbinns', nns_parameters_dataframe_line_index)

    if os.path.exists(file_path):
        print(file_path,' already exists!')
    else:    
        pipe_to_exec = Pipeline([('pbinns',PBINearestNeighbors(bucket_count=2, reference_set_size=2, prunning_size=1, ref_sel_threshold=0.5, n_neighbors=2, sort_neighbors=False))])
        pipe_to_exec.set_params(**nns_parameters_dataframe_line.drop('input__filename_index'))
        
        __nearest_neighbors_search(pipe_to_exec, source_file_path, file_path)
        
def nearest_neighbors_search(dataset_name, nns_parameters_dataframe_line, nns_parameters_dataframe_line_index, encoding):
    indexi = nns_parameters_dataframe_line['input__filename_index']
    source_file_path = h5_results_filename(dataset_name, 'cv', indexi)
    file_path = h5_results_filename(dataset_name, 'nns', nns_parameters_dataframe_line_index)

#     print(nns_parameters_dataframe_line)
    
    if os.path.exists(file_path) :
        print(file_path,' already exists!')
    else:    
        pipe_to_exec = Pipeline([('nns',InvertedIndexNearestNeighbors(n_neighbors=2))])
        pipe_to_exec.set_params(**nns_parameters_dataframe_line.drop('input__filename_index'))

        __nearest_neighbors_search(pipe_to_exec, source_file_path, file_path)

def generate_or_load_parameters_grids(parameters_sequence,dataset_name):
    parameters_grids_list = []

    for i in range(len(parameters_sequence)):
        sufix,_parameters = parameters_sequence[i]
        dataframe_df_path = "%s_parameters_%s.h5"%(dataset_name,sufix)
        
        dataframe_df_paramaters = parameters_gridlist_dataframe(_parameters).drop_duplicates()

        '''
            search and add new parameters!
        '''
        if os.path.exists(dataframe_df_path):
            print(dataframe_df_path,'exists! merging with new values!')

            current_df = pd.read_hdf(dataframe_df_path, "parameters")
            
            '''
                getting what already exists!
            '''

            intersection = pd.merge(current_df,dataframe_df_paramaters)
            
            
            if intersection.shape[0] > 0:
                for _,rowi in dataframe_df_paramaters.iterrows():
                    inter_i = pd.merge(pd.DataFrame([rowi]),intersection)
                    
                    if inter_i.shape[0] == 0:
                        temp = pd.DataFrame([rowi])
                        temp.index =[current_df.shape[0]]
                        current_df = current_df.append(temp)
            else:
                temp = dataframe_df_paramaters
                temp.index = list(range(current_df.shape[0],current_df.shape[0] + dataframe_df_paramaters.shape[0]))
#                 print(temp.index)
                current_df = current_df.append(temp)
        else:
            current_df = dataframe_df_paramaters

        current_df.to_hdf(dataframe_df_path, "parameters")
        
        '''
            preserving original index! 
        '''
        current_df.loc[list(current_df.index),'%s_file_index'%(sufix)] = list(current_df.index)
        parameters_grids_list.append(pd.merge(current_df,dataframe_df_paramaters))
        parameters_grids_list[-1] = parameters_grids_list[-1].set_index('%s_file_index'%(sufix),drop=True)
                
        if i  < len(parameters_sequence) - 1:
            parameters_sequence[i+1][1]['input__filename_index'] = parameters_grids_list[-1].index
    
    return parameters_grids_list



if __name__ == '__main__':
    
    '''
        creating TfIdfVectorizer, LSHTransformer and LSHIINearestNeighbors parameters grids and
        storing it as pandas Dataframes on hdf
    '''
#     dataset_name = "psa"
#     dataset_name = "pan10"
    dataset_name = "pan11"

#     dataset_name,sample_size = "pan10-%d-samples",10 
#     dataset_name = dataset_name%(sample_size)
     
    cv_parameters = {
        "cv__analyzer" : ('word',),
        "cv__ngram_range" : (
                            (1,1),
#                             (3,3), 
#                             (5,5), 
#                             (1,3),
#                             (3,5),
                             ),
        "cv__tokenizer" : (
                            None,
                           ),
        "cv__lowercase" : (True,),
        "cv__min_df" : (
#                         1,
                        2, 
                        ),
        "cv__binary" : (False,),
        "cv__stop_words" : ('english',),
        "cv__use_idf" : (True,),
        "cv__norm" : (
                      'l1',
#                       None,'l2'
                      ),
        
    }
    
    lsht_parameters = {
        "lsht__n_permutations" : (
                                  1,
#                                 48, 96, 192, 384, 768, # min
#                                 24, 48,  96, 192, 384, #minmax
#                                 16, 32,  64, 128, 256, #csa_l
#                                 12, 24,  48,  96, 192, #csa
                                  ),
        "lsht__selection_function" : (
                                    min_hashing,
#                                     minmax_hashing,
#                                     minmaxCSALowerBound_hashing,
#                                     minmaxCSAFullBound_hashing,
#                                     justCSALowerBound_hashing,
#                                     justCSAFullBound_hashing,
                                      ),
        "lsht__n_jobs" : (
#                         1,
#                         2,
                        -1,
                          )                       
    }
    
    lshnns_parameters = {
        "lshnns__n_neighbors" : (
#        5,
                                5,10
#                                 22,10,
#                                160, 1597, 3991, 7983, 11975, 15966 # PAN11(EN, just queries with relevants) 1%, 10%, 25%, 50%, 75%, 100%
#                                 1597, 3991, 7983, 11975 # PAN11(EN, just queries with relevants) 10%, 25%, 50%, 75%
#                                 ,
#                                 15,30
#                                 15,22
                                 ),
        "lshnns__sort_neighbors" : (
                                     False,
#                                    True,
                                    ),
                         
    }
    
    nns_parameters = {
        "nns__n_neighbors" : lshnns_parameters['lshnns__n_neighbors'],
        "nns__sort_neighbors" : lshnns_parameters['lshnns__sort_neighbors'],
    }

    pbinns_parameters = {
        "pbinns__n_neighbors" : nns_parameters['nns__n_neighbors'],
        "pbinns__sort_neighbors" : nns_parameters['nns__sort_neighbors'],
        "pbinns__bucket_count" : (2,),
        "pbinns__reference_set_size" : (40,),
        "pbinns__prunning_size" : (10,),
        "pbinns__ref_sel_threshold" : (0.1, #0.2, 0.4, 0.5 
                                       ),
    }


    '''
        storing parameters dataframes 
    '''

    parameters_sequence = [('cv',cv_parameters),('lsht',lsht_parameters),('lshnns',lshnns_parameters)]
    parameters_grids_list = generate_or_load_parameters_grids(parameters_sequence,dataset_name)

    cv_df_paramaters, lsht_df_paramaters, lshnns_df_paramaters = parameters_grids_list

#     for i in parameters_grids_list:
#         print(i)             
#         print('xxxxxxxxxxxxxxxxxxxx')
#     exit()
    
    '''
        nearest neighbor search without LSH
    '''    
    parameters_sequence = [('cv',cv_parameters),('nns',nns_parameters)]
    parameters_grids_list = generate_or_load_parameters_grids(parameters_sequence,dataset_name)

    nns_df_paramaters = parameters_grids_list[1]

#     for i in parameters_grids_list:
#         print(i)             
#         print('xxxxxxxxxxxxxxxxxxxx')
#     exit()

    '''
        permutation-Based Index(PBI) nearest neighbor search
    '''    

    parameters_sequence = [('cv',cv_parameters),('pbinns',pbinns_parameters)]
    parameters_grids_list = generate_or_load_parameters_grids(parameters_sequence,dataset_name)

    pbinns_df_paramaters = parameters_grids_list[1]
    
#     for i in parameters_grids_list:
#         print(i)             
#         print('xxxxxxxxxxxxxxxxxxxx')
#     exit()
    

    '''
        dataset extraction
    '''
    if dataset_name == "psa":
        corpus_name, (suspicious_info, source_info,target, dataset_encoding) = dataset_name, short_plagiarised_answers_extractor.load_as_ir_task()
    elif dataset_name == "pan10":
        corpus_name, (suspicious_info, source_info,target, dataset_encoding) = dataset_name, pan_plagiarism_corpus_2010_extractor.load_as_ir_task(allow_queries_without_relevants=False)
    elif dataset_name == "pan11":
        corpus_name, (suspicious_info, source_info,target, dataset_encoding) = dataset_name, pan_plagiarism_corpus_2011_extractor.load_as_ir_task(allow_queries_without_relevants=False, language_filter="EN")
    elif "pan10" in dataset_name and "-samples" in dataset_name:
        corpus_name, (suspicious_info, source_info,target, dataset_encoding) = dataset_name, pan_plagiarism_corpus_2010_extractor.load_sample_as_ir_task(sample_size, language_filter="EN")

    print('queries:',suspicious_info.shape,' Documents:',source_info.shape)


    documents = Parallel(n_jobs=-1,backend="threading",verbose=1)(delayed(encode_dataframe_content)(si, dataset_encoding) for si in source_info['content'].values)
    queries = Parallel(n_jobs=-1,backend="threading",verbose=1)(delayed(encode_dataframe_content)(si, dataset_encoding) for si in suspicious_info['content'].values)
    
    del suspicious_info, source_info
    
    print(nns_df_paramaters)
    print(lsht_df_paramaters) 
        
#     exit()
    
    '''
        using scikit-learn : tokenization
    '''

    for i,linei in cv_df_paramaters.iterrows():
        tokenize_by_parameters(documents,queries,target,dataset_name,linei,i,dataset_encoding)

    queries_count,documents_count = target.shape
    del documents, queries, target
    
    '''
        transforming bow to lsh representations (min-hasing, minmax-hashing, ...)
    '''

    for i,linei in lsht_df_paramaters.iterrows():
        print(linei)
        print('xxxxxx')
        lsh_transform(dataset_name,linei,i,dataset_encoding)

    '''
        nearest neighbor search (ranking)
    '''

    for i,linei in lshnns_df_paramaters.iterrows():
        print("#"*10+" LSH N.N.S. "+"#"*10)
        print(linei)
        lsh_nearest_neighbors_search(dataset_name,linei,i,dataset_encoding)
        print("-"*20)

    for i,linei in nns_df_paramaters.iterrows():
        print("#"*10+" N.N.S. "+"#"*10)
        print(linei)
        nearest_neighbors_search(dataset_name,linei,i,dataset_encoding)
        print("-"*20)
 
    for i,linei in pbinns_df_paramaters.iterrows():
        print("#"*10+" PBI N.N.S. "+"#"*10)
        print(linei)
        pbinearest_neighbors_search(dataset_name,linei,i,dataset_encoding)
        print("-"*20)
    
    today = datetime.now()
    today = today.strftime('%Y-%m-%d_%H-%M-%S_')
    
    
    '''
        logging LSH nearest neighbors results on csv
    '''
    a = pd.merge(cv_df_paramaters, lsht_df_paramaters, how='inner', left_index=True, right_on=['input__filename_index',],)
    b = pd.merge(a, lshnns_df_paramaters, how='inner', left_index=True, right_on=['input__filename_index',],suffixes=('_lsht','_lshtnns'))
    del a

    for rowi in b.iterrows():
        cv_index = rowi[1]['input__filename_index_lsht']
        lsht_index = rowi[1]['input__filename_index_lshtnns']
        nns_index = rowi[0]
        
#         print(cv_index,'-',lsht_index,'-',nns_index)
        cv_file_path = h5_results_filename(dataset_name, 'cv', cv_index).replace('results','time')
        lsht_file_path = h5_results_filename(dataset_name, 'lsht', lsht_index).replace('results','time')
        lshnns_file_path = h5_results_filename(dataset_name, 'lshnns', nns_index).replace('results','results_evaluation')
        lshnns_time_file_path = h5_results_filename(dataset_name, 'lshnns', nns_index).replace('results','time')

#         print('\t',cv_file_path)
#         print('\t',lsht_file_path)
#         print('\t',lshnns_file_path)
#         print('=========')
        approach_precisions = hdf_to_sparse_matrix('precisions', lshnns_file_path)
        approach_recalls = hdf_to_sparse_matrix('recalls', lshnns_file_path)
        average_precision = hdf_to_sparse_matrix('average_precisions', lshnns_file_path).todense()
        
        b.loc[nns_index,'MAP'] = average_precision.mean()
        b.loc[nns_index,'MAP_std'] = average_precision.std()
        b.loc[nns_index,'precision_recall_path'] = lshnns_file_path
        b.loc[nns_index,'recall_mean'] = approach_recalls[:,-1].todense().mean()
        b.loc[nns_index,'recall_std'] = approach_recalls[:,-1].todense().std()
        b.loc[nns_index,'precision_mean'] = approach_precisions[:,-1].todense().mean()
        b.loc[nns_index,'precision_std'] = approach_precisions[:,-1].todense().std()
        
        del approach_precisions, approach_recalls, average_precision

        b.loc[nns_index,'documents_count'] = documents_count
        b.loc[nns_index,'queries_count'] = queries_count
        
        with open(cv_file_path.replace('time.h5', 'vocabulary.pkl'),'rb') as f:
            b.loc[nns_index,'vocabulary_size'] = len(pickle.load(f))

        q = hdf_to_sparse_matrix('queries', lsht_file_path.replace('time','results'))
        b.loc[nns_index,'lsht_features'] = q.shape[1]
        del q
        
        b.loc[nns_index,'indexing_mean_time'] = 0
        b.loc[nns_index,'querying_mean_time'] = 0
        
        cv_time_dataframe = pd.read_hdf(cv_file_path, 'time_dataframe')
        b.loc[nns_index,'cv_documents_mean_time'] = cv_time_dataframe.loc[0,'documents_mean_time'] 
        b.loc[nns_index,'cv_queries_mean_time'] = cv_time_dataframe.loc[0,'queries_mean_time']

        b.loc[nns_index,'indexing_mean_time'] += b.loc[nns_index,'cv_documents_mean_time']
        b.loc[nns_index,'querying_mean_time'] += b.loc[nns_index,'cv_queries_mean_time']
        del cv_time_dataframe 

        d_time = hdf_to_sparse_matrix('documents_time',lsht_file_path)
        q_time = hdf_to_sparse_matrix('queries_time',lsht_file_path)
        
        b.loc[nns_index,'lsht_documents_mean_time'] = d_time.sum(axis=1).mean() 
        b.loc[nns_index,'lsht_queries_mean_time'] = q_time.sum(axis=1).mean()

        b.loc[nns_index,'indexing_mean_time'] += b.loc[nns_index,'lsht_documents_mean_time']
        b.loc[nns_index,'querying_mean_time'] += b.loc[nns_index,'lsht_queries_mean_time']
        del d_time, q_time 

        nns_time_dataframe = pd.read_hdf(lshnns_time_file_path, 'time_dataframe')
        b.loc[nns_index,'nns_documents_mean_time'] = nns_time_dataframe.loc[0,'documents_mean_time'] 
        b.loc[nns_index,'nns_queries_mean_time'] = nns_time_dataframe.loc[0,'queries_mean_time']

        b.loc[nns_index,'indexing_mean_time'] += b.loc[nns_index,'nns_documents_mean_time']
        b.loc[nns_index,'querying_mean_time'] += b.loc[nns_index,'nns_queries_mean_time']
        del nns_time_dataframe

        print(b.loc[nns_index,'lsht__selection_function'],' : ',int(b.loc[nns_index,'lsht_features']),' features x ',b.loc[nns_index,'lsht__n_permutations'],' permutation')
        print("MAP = %4.2f[+-%4.2f]"%(b.loc[nns_index,'MAP'],b.loc[nns_index,'MAP_std']))
        print("recall = %4.2f[+-%4.2f]"%(b.loc[nns_index,'recall_mean'],b.loc[nns_index,'recall_std']))
        print("index time = %4.4f"%(b.loc[nns_index,'cv_documents_mean_time']+b.loc[nns_index,'lsht_documents_mean_time']+b.loc[nns_index,'nns_documents_mean_time']))
        print("query time = %4.4f"%(b.loc[nns_index,'cv_queries_mean_time']+b.loc[nns_index,'lsht_queries_mean_time']+b.loc[nns_index,'nns_queries_mean_time']))
         
        print("---->",b.loc[nns_index,'indexing_mean_time'])
    b.to_csv('%s%s_lsh_results.csv'%(today,dataset_name),sep='\t')
    
    del b
    
    
    '''
        logging nearest neighbors (without LSH) results on csv
    '''
    b = pd.merge(cv_df_paramaters, nns_df_paramaters, how='inner', left_index=True, right_on=['input__filename_index',],)
 
    for rowi in b.iterrows():
        cv_index = rowi[1]['input__filename_index']
        nns_index = rowi[0]
         
        print(cv_index,'-',lsht_index,'-',nns_index)
        cv_file_path = h5_results_filename(dataset_name, 'cv', cv_index).replace('results','time')
        nns_file_path = h5_results_filename(dataset_name, 'nns', nns_index).replace('results','results_evaluation')
        nns_time_file_path = h5_results_filename(dataset_name, 'nns', nns_index).replace('results','time')
 
#         print('\t',cv_file_path)
#         print('\t',lsht_file_path)
#         print('\t',lshnns_file_path)
#         print('=========')
        approach_precisions = hdf_to_sparse_matrix('precisions', nns_file_path)
        approach_recalls = hdf_to_sparse_matrix('recalls', nns_file_path)
        average_precision = hdf_to_sparse_matrix('average_precisions', nns_file_path).todense()
         
        b.loc[nns_index,'MAP'] = average_precision.mean()
        b.loc[nns_index,'MAP_std'] = average_precision.std()
        b.loc[nns_index,'recall_mean'] = approach_recalls[:,-1].todense().mean()
        b.loc[nns_index,'recall_std'] = approach_recalls[:,-1].todense().std()
        b.loc[nns_index,'precision_mean'] = approach_precisions[:,-1].todense().mean()
        b.loc[nns_index,'precision_std'] = approach_precisions[:,-1].todense().std()
         
        del approach_precisions, approach_recalls, average_precision
 
        b.loc[nns_index,'documents_count'] = documents_count
        b.loc[nns_index,'queries_count'] = queries_count
         
        with open(cv_file_path.replace('time.h5', 'vocabulary.pkl'),'rb') as f:
            b.loc[nns_index,'vocabulary_size'] = len(pickle.load(f))
         
        b.loc[nns_index,'indexing_mean_time'] = 0
        b.loc[nns_index,'querying_mean_time'] = 0
 
        cv_time_dataframe = pd.read_hdf(cv_file_path, 'time_dataframe')
        b.loc[nns_index,'cv_documents_mean_time'] = cv_time_dataframe.loc[0,'documents_mean_time'] 
        b.loc[nns_index,'cv_queries_mean_time'] = cv_time_dataframe.loc[0,'queries_mean_time']
 
        b.loc[nns_index,'indexing_mean_time'] += b.loc[nns_index,'cv_documents_mean_time']
        b.loc[nns_index,'querying_mean_time'] += b.loc[nns_index,'cv_queries_mean_time']
        del cv_time_dataframe 
 
        nns_time_dataframe = pd.read_hdf(nns_time_file_path, 'time_dataframe')
        b.loc[nns_index,'nns_documents_mean_time'] = nns_time_dataframe.loc[0,'documents_mean_time'] 
        b.loc[nns_index,'nns_queries_mean_time'] = nns_time_dataframe.loc[0,'queries_mean_time']
 
        b.loc[nns_index,'indexing_mean_time'] += b.loc[nns_index,'nns_documents_mean_time']
        b.loc[nns_index,'querying_mean_time'] += b.loc[nns_index,'nns_queries_mean_time']
        del nns_time_dataframe 
 
        print('nns:')
        print("MAP = %4.2f[+-%4.2f]"%(b.loc[nns_index,'MAP'],b.loc[nns_index,'MAP_std']))
        print("recall = %4.2f[+-%4.2f]"%(b.loc[nns_index,'recall_mean'],b.loc[nns_index,'recall_std']))
        print("index time = %4.4f"%(b.loc[nns_index,'cv_documents_mean_time']+b.loc[nns_index,'nns_documents_mean_time']))
        print("query time = %4.4f"%(b.loc[nns_index,'cv_queries_mean_time']+b.loc[nns_index,'nns_queries_mean_time']))
 
    b.to_csv('%s%s_results.csv'%(today,dataset_name),sep='\t')    
    
    
    '''
        Permutation-Based Index (PBI) logging nearest neighbors results on csv
    '''
    b = pd.merge(cv_df_paramaters, pbinns_df_paramaters, how='inner', left_index=True, right_on=['input__filename_index',],)
  
    for rowi in b.iterrows():
        cv_index = rowi[1]['input__filename_index']
        pbinns_index = rowi[0]
          
#         print(cv_index,'-',lsht_index,'-',nns_index)
        cv_file_path = h5_results_filename(dataset_name, 'cv', cv_index).replace('results','time')
        pbinns_file_path = h5_results_filename(dataset_name, 'pbinns', pbinns_index).replace('results','results_evaluation')
        pbinns_time_file_path = h5_results_filename(dataset_name, 'pbinns', pbinns_index).replace('results','time')
  
#         print('\t',cv_file_path)
#         print('\t',lsht_file_path)
#         print('\t',lshnns_file_path)
#         print('=========')
        approach_precisions = hdf_to_sparse_matrix('precisions', pbinns_file_path)
        approach_recalls = hdf_to_sparse_matrix('recalls', pbinns_file_path)
        average_precision = hdf_to_sparse_matrix('average_precisions', pbinns_file_path).todense()
          
        b.loc[pbinns_index,'MAP'] = average_precision.mean()
        b.loc[pbinns_index,'MAP_std'] = average_precision.std()
        b.loc[pbinns_index,'recall_mean'] = approach_recalls[:,-1].todense().mean()
        b.loc[pbinns_index,'recall_std'] = approach_recalls[:,-1].todense().std()
        b.loc[pbinns_index,'precision_mean'] = approach_precisions[:,-1].todense().mean()
        b.loc[pbinns_index,'precision_std'] = approach_precisions[:,-1].todense().std()
          
        del approach_precisions, approach_recalls, average_precision
  
        b.loc[pbinns_index,'documents_count'] = documents_count
        b.loc[pbinns_index,'queries_count'] = queries_count
          
        with open(cv_file_path.replace('time.h5', 'vocabulary.pkl'),'rb') as f:
            b.loc[pbinns_index,'vocabulary_size'] = len(pickle.load(f))
         
        b.loc[nns_index,'indexing_mean_time'] = 0
        b.loc[nns_index,'querying_mean_time'] = 0
 
        cv_time_dataframe = pd.read_hdf(cv_file_path, 'time_dataframe')
        b.loc[pbinns_index,'cv_documents_mean_time'] = cv_time_dataframe.loc[0,'documents_mean_time'] 
        b.loc[pbinns_index,'cv_queries_mean_time'] = cv_time_dataframe.loc[0,'queries_mean_time']
          
        b.loc[nns_index,'indexing_mean_time'] += b.loc[pbinns_index,'cv_documents_mean_time'] 
        b.loc[nns_index,'querying_mean_time'] += b.loc[pbinns_index,'cv_queries_mean_time']
        del cv_time_dataframe 
  
        pbinns_time_dataframe = pd.read_hdf(pbinns_time_file_path, 'time_dataframe')
        b.loc[pbinns_index,'pbinns_documents_mean_time'] = pbinns_time_dataframe.loc[0,'documents_mean_time'] 
        b.loc[pbinns_index,'pbinns_queries_mean_time'] = pbinns_time_dataframe.loc[0,'queries_mean_time']
          
        b.loc[nns_index,'indexing_mean_time'] += b.loc[pbinns_index,'pbinns_documents_mean_time'] 
        b.loc[nns_index,'querying_mean_time'] += b.loc[pbinns_index,'pbinns_queries_mean_time']
        del pbinns_time_dataframe 
  
        print('pbinns:')
        print("MAP = %4.2f[+-%4.2f]"%(b.loc[pbinns_index,'MAP'],b.loc[pbinns_index,'MAP_std']))
        print("recall = %4.2f[+-%4.2f]"%(b.loc[pbinns_index,'recall_mean'],b.loc[pbinns_index,'recall_std']))
        print("index time = %4.4f"%(b.loc[pbinns_index,'cv_documents_mean_time']+b.loc[pbinns_index,'pbinns_documents_mean_time']))
        print("query time = %4.4f"%(b.loc[pbinns_index,'cv_queries_mean_time']+b.loc[pbinns_index,'pbinns_queries_mean_time']))
  
    b.to_csv('%s%s_pbi_results.csv'%(today,dataset_name),sep='\t')        