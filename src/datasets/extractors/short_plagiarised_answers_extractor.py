import os.path
import re
import pandas as pd
import numpy as np
from scipy.sparse.lil import lil_matrix
from datasets.extractors import datasets_extractors
from locality_sensitive_hashing import LSHTransformer
from sklearn.neighbors.unsupervised import NearestNeighbors
from numpy import nonzero
from sklearn.metrics.classification import accuracy_score, f1_score
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse.coo import coo_matrix

__DATASET_ENCODING = 'latin1'

def load_to_pandas():
    '''
        loading 100 files (5 sources + 95 suspicious)
        
        return dataset info as a pandas.Dataframe
    '''
    path = datasets_extractors['DATASETS_PATH']['short_plagiarised_answers_dataset']
    
    dataset_encoding = __DATASET_ENCODING
    
    files_path = os.path.join(path,"query_answer_json")
    with open(files_path,'r', encoding=dataset_encoding) as f:
        json = ''.join(f.readlines())
        dataset_dataframe = pd.read_json(json, orient='records')
        
    files_path = os.path.join(path,"source")
    dir_list = os.listdir(files_path)

    source_values = []
    for dir_filename in dir_list:
        source_values.append([
                              dir_filename,
                              'source',
                              dir_filename.replace('.txt',"").replace("orig_task","")
                              ])

    dataset_dataframe = dataset_dataframe.append(
                            pd.DataFrame(source_values,columns=list(["document","plag_type","task"])), 
                            ignore_index=True)

    '''
        reading each file content
    '''
    for i in range(dataset_dataframe.shape[0]):
        file_pathi = os.path.join(path,dataset_dataframe.loc[i,"plag_type"],dataset_dataframe.loc[i,"document"])
        with open(file_pathi,'r', encoding=dataset_encoding) as f:
            dataset_dataframe.loc[i,'content'] = f.read()

    return dataset_dataframe,dataset_encoding 

def load_as_pairs():
    '''
        5 documents X 95 suspicious = 475 comparison pairs 
        
        475 plagiarism pairs to eval.
    '''

    spa_original,dataset_encoding = load_to_pandas()
    
    dataset_documents = []
    dataset_target = []

    for i in range(95,100):
        for j in range(0,95):
            dataset_documents.append((
                    re.sub(r'\.((\\r)?\\n)+','. \r\n',spa_original.loc[i,"content"]).replace('\\r','\r').replace('\\n','\n'),
                    re.sub(r'\.((\\r)?\\n)+','. \r\n',spa_original.loc[j,"content"].replace('\\r','\r').replace('\\n','\n'))
                    ))
    
            dataset_target.append(spa_original.loc[i,"task"] == spa_original.loc[j,"task"])

    dataset_documents = np.matrix(dataset_documents,dtype=np.ndarray)
    dataset_target = np.array(dataset_target)
    return dataset_documents ,dataset_target ,0 ,0,dataset_encoding

def load_as_ir_task():
    '''
        5 source documents (index) X 57 suspicious documents (queries)
        
        1 relevant document (source) for each query!
    '''
    
    path = datasets_extractors['DATASETS_PATH']['short_plagiarised_answers_dataset']
    files_path = os.path.join(path,"ir_task_short_plagiarised_answers.h5")
    
    if os.path.exists(files_path):
        #load and return 
        queries = pd.read_hdf(files_path, 'queries')
        documents = pd.read_hdf(files_path, 'documents')
        dataset_target = pd.read_hdf(files_path, 'targets')
        data = dataset_target.loc[:,'data'].values
        row = dataset_target.loc[:,'index'].values
        col = dataset_target.loc[:,'col'].values
        dataset_target = coo_matrix((data, (row, col)), shape=(queries.shape[0], documents.shape[0]))
        dataset_target = dataset_target.tolil()
        dataset_encoding = __DATASET_ENCODING

    else:
        spa_original,dataset_encoding = load_to_pandas()
        queries_dataframe, documents_dataframe = spa_original[0:95],spa_original[95:100]
        dataset_target = lil_matrix((100,100))
        del spa_original
        
        queries = []
        queries_dataframe_indexes = []
        documents = documents_dataframe['content'].tolist()
        documents_dataframe_indexes = documents_dataframe.index.values.tolist()
    
        for rowi_index, rowi in queries_dataframe[queries_dataframe.plag_type != "non"].iterrows():
            i = len(queries)
            for j,source_rowj in documents_dataframe.iterrows():
                j -= 95
                dataset_target[i,j] = source_rowj["task"] == rowi['task']
            queries.append(rowi['content'])
            queries_dataframe_indexes.append(rowi_index)
    
        non_plagiarism = queries_dataframe[queries_dataframe.plag_type == "non"]
        documents = documents + non_plagiarism['content'].values.tolist()
        documents_dataframe_indexes = documents_dataframe_indexes + non_plagiarism.index.values.tolist()
        dataset_target = dataset_target[:len(queries),:len(documents)]
    
    #     print(len(queries),len(documents),dataset_target.shape)
        del queries_dataframe, documents_dataframe        
    
        
        queries = pd.DataFrame({'content':queries,'original_indexes':queries_dataframe_indexes})
        documents = pd.DataFrame({'content':documents,'original_indexes':documents_dataframe_indexes})

        queries.to_hdf(files_path,'queries',append=True)
        documents.to_hdf(files_path,'documents',append=True)
        '''
            storing scipy sparse matrix on dataframe to dump on hdf5
        '''
        coo = dataset_target.tocoo()
        pd.DataFrame({'index': coo.row, 'col': coo.col, 'data': coo.data}
                )[['index', 'col', 'data']].sort_values(['index', 'col']
                ).reset_index(drop=True).to_hdf(files_path,'targets',append=True)        

        
    return queries, documents, dataset_target, dataset_encoding

def encode_dataframe_content(dataframe_contenti, encoding):
    return dataframe_contenti.encode(encoding)    

if __name__ == '__main__':
    dataset_pairs,targets,train_size,test_size,dataset_encoding = load_as_pairs()
    print(dataset_pairs.shape)
    print(targets.shape, 'train_size = ',train_size, 'test_size = ',test_size)
    print(np.nonzero(targets)[0].shape, '+', np.nonzero(targets==False)[0].shape)

    for i in range(dataset_pairs.shape[0]):
        s1,s2 = dataset_pairs[i,0],dataset_pairs[i,1]
        print('Plagiarism?',targets[i],':',s1[:50],'[X]',s2[:50])
        print('Plagiarism?',targets[i],':',(s1.encode(dataset_encoding))[:50],'[X]',(s2.encode(dataset_encoding))[:50])
#         input('next')

    queries, corpus_index, dataset_target,dataset_encoding = load_as_ir_task()
    print(len(queries), len(corpus_index), dataset_target.shape)
    