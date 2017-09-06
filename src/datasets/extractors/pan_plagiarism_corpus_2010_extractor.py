import pandas as pd
import os.path
import glob
from datasets.extractors import datasets_extractors
import xml.etree.ElementTree as ET
import re
import numpy as np
from scipy.sparse.coo import coo_matrix
from sklearn.externals.joblib.parallel import delayed, Parallel

def load_to_pandas():
    '''
        loading 27073 documents and 68558 plagiarism cases.
        
        return dataset info as two pandas.Dataframe
    '''
    root = datasets_extractors['DATASETS_PATH']['pan_plagiarism_corpus_2010']
    filenames_list =[
                ( 'source-document',
                  'suspicious-document'),
    ]
    
    source_dataframe = None
    susp_dataframe = None
    overwrite = ''
    encoding_name = "utf-8"

    for filenamesi in filenames_list:
#         folder_path = os.path.join(root,filenamesi[0])
        folder_path = root
        
        for i in range(0,len(filenamesi)):
            path = os.path.join(folder_path,filenamesi[i])

            if not overwrite == 'a':
                overwrite= ''          
            csv_name = os.path.join(path,filenamesi[i]+'.csv')
            if os.path.exists(csv_name):
                
                while not ('y' in overwrite or 'n' in overwrite or 'a' in overwrite):
                    overwrite = input("%s already exists! \nOverwrite [y]es, [n]o or [a]ll?"%(csv_name))
                
                if 'n' in overwrite:
                    if filenamesi[i] == filenamesi[0]:
                        source_dataframe = pd.DataFrame.from_csv(csv_name, sep='\t',encoding = encoding_name)
                    else:
                        susp_dataframe = pd.DataFrame.from_csv(csv_name, sep='\t',encoding = encoding_name)
                    
                    continue
            
            dir_list = os.listdir(path)
            for dirName in dir_list:
                file_pattern = os.path.join(path, dirName, filenamesi[i]+'?????.xml')
                docs = glob.glob(file_pattern)
                for doc in docs:
                    tree = ET.parse(doc)
                    root = tree.getroot()

                    atribs_dict = root.attrib.copy()
                    atribs_dict['dirName'] = dirName
                    for child in root:
                        if child.attrib['name'] == 'md5Hash':
                            atribs_dict['md5Hash'] = child.attrib['value']
                        elif child.attrib['name'] == 'about': 
                            atribs_dict.update(child.attrib)
                            atribs_dict.pop('name')
    
                    if filenamesi[i] == filenamesi[0]:
                        df2 = pd.DataFrame([atribs_dict.values()],columns=list(atribs_dict.keys()))
                        if source_dataframe is None:
                            source_dataframe = df2
                        else:
                            source_dataframe = source_dataframe.append(df2, ignore_index=True)
                    else:
                        plagiarism_count = 0
                        for child in root:
                            if child.attrib['name'] == 'plagiarism' and 'source_reference' in child.attrib:
#                                 print("[%s]"%(child.attrib['source_reference']))
                            
                                if (not child.attrib['source_reference'] is None) :
                                    plagiarism_count += 1
                                    susp_at_dict = atribs_dict.copy()
                                    susp_at_dict.update(child.attrib)
    
                                    df2 = pd.DataFrame([susp_at_dict.values()],columns=list(susp_at_dict.keys()))
                                    if susp_dataframe is None:
                                        susp_dataframe = df2
                                    else:
                                        susp_dataframe = susp_dataframe.append(df2, ignore_index=True)

                        '''
                            Non plagiarised cases (name=NaN)
                        '''
                        if plagiarism_count == 0 :
                            df2 = pd.DataFrame([atribs_dict.values()],columns=list(atribs_dict.keys()))
                            df2.loc[:,'name'] = None
                            if susp_dataframe is None:
                                susp_dataframe = df2
                            else:
                                susp_dataframe = susp_dataframe.append(df2, ignore_index=True)
            del dir_list
            if filenamesi[i] == filenamesi[0]:
                source_dataframe.to_csv(csv_name, sep='\t',encoding = encoding_name)
            else:
                susp_dataframe.to_csv(csv_name, sep='\t',encoding = encoding_name)
    
    del root
    
    return source_dataframe, susp_dataframe, encoding_name


def load_file_content(pan10_dataframe,filename,data_encoding='UTF-8'):
    root = datasets_extractors['DATASETS_PATH']['pan_plagiarism_corpus_2010']
    
    df_line = next(pan10_dataframe[pan10_dataframe.reference == filename].iterrows())
    dirName, reference = pan10_dataframe.loc[df_line[0],['dirName','reference']]

    if "source" in reference:
        doc_type = 'source-document'
    else:
        doc_type = 'suspicious-document'        

    file_path = os.path.join(root,doc_type,dirName,reference)
    with open(file_path,'r',encoding=data_encoding) as f:
        content = f.read()
        

    del dirName, reference, df_line
    return content

def sentences_pos_process(content):
    dot_list = ''.join(['.' for i in range(5)]) 
    dot_list += "+"       
        
    pattern_list = [
                        r'%s( )+([0-9]+\r\n)'%(dot_list),
                        r'%s( )+([0-9]+\n)'%(dot_list),
                        r'[a-zA-Z] / ',
                        r'%s.+;((\n)|( )|(--))'%(dot_list),
                        r'%s\|( )\.\.\r?\n'%(dot_list)
                    ]

    content = re.sub(r'\n[\n]+', '\n', content)
    content = re.sub(r'CHAPTER ', '. \nCHAPTER ', content)

    for patterni in pattern_list:
        for m in re.finditer(patterni, content):
            start_pos, end_pos =(m.start(0), m.end(0))
#                 if (end_pos - start_pos < 10):
#                     print(start_pos,end_pos,'=',content[start_pos:end_pos])
#                 else:
#                     print('end_pos - start_pos =',end_pos - start_pos )
            content = content[:end_pos-2] + '. ' + content[end_pos:]

#                 print(start_pos,end_pos,'=',content[start_pos:end_pos])
#                 input("!")
             
    return content


def _load_as_ir_task_without_content(allow_queries_without_relevants = True, language_filter = 'ALL'):
    '''
        ???? source documents (index) X ???? suspicious documents (queries) (allowing queries without relevants)
    '''
    
    path = datasets_extractors['DATASETS_PATH']['pan_plagiarism_corpus_2010']
    if allow_queries_without_relevants:
        files_path = os.path.join(path,"ir_task_PAN10(%s)_allows.h5"%(language_filter))
    else:
        files_path = os.path.join(path,"ir_task_PAN10(%s).h5"%(language_filter))
    
    if os.path.exists(files_path):
        source_dataframe, susp_dataframe,dataset_encoding = load_to_pandas()
        del source_dataframe, susp_dataframe

        #load and return 
        queries = pd.read_hdf(files_path, 'queries')
        documents = pd.read_hdf(files_path, 'documents')
        targets_pairs = pd.read_hdf(files_path, 'targets_pairs')

        source_dataframe = pd.read_hdf(files_path,'source_dataframe')
        susp_dataframe = pd.read_hdf(files_path,'susp_dataframe')
        
    else:
        source_dataframe, susp_dataframe,dataset_encoding = load_to_pandas()

        if not allow_queries_without_relevants:
            non_plag_susp = susp_dataframe[pd.isnull(susp_dataframe.source_reference)]
            new_susp_dataframe = susp_dataframe[pd.notnull(susp_dataframe.source_reference)].reset_index()
            new_source_dataframe = source_dataframe.append(non_plag_susp, ignore_index=True)
            del source_dataframe,susp_dataframe, non_plag_susp
        else:
            new_source_dataframe = source_dataframe
            new_susp_dataframe = susp_dataframe
            
        source_dataframe = new_source_dataframe
        susp_dataframe = new_susp_dataframe

        documents = []
        queries = []
        targets_pairs = []
        
        queries_names = {}
        
        for source_id, source_row in new_source_dataframe.iterrows():
            if language_filter != "ALL" and source_row['lang'] != language_filter.lower() and source_row['lang'] != None:
                continue
            documents.append(source_id)
            susp_documents = new_susp_dataframe[new_susp_dataframe.source_reference == source_row['reference']]
            
            '''
                groups by suspicious filename each source plagiarised slices of text (pandas indexes).
            '''
            grouped_susp_dataframe = susp_documents.groupby(['reference',])
             
            '''
                selecting one index to represent the query 
            '''
            for query_namei,valuesi in grouped_susp_dataframe.groups.items():
                 
                if query_namei in queries_names.keys():
                    targets_pairs.append([queries_names[query_namei],len(documents)-1])
                else:
                    queries_names[query_namei] = len(queries)
                    queries.append(valuesi[0])
                    targets_pairs.append([len(queries)-1,len(documents)-1])
            
        if allow_queries_without_relevants:                    
            empty_queries = susp_dataframe[pd.isnull(susp_dataframe.source_reference)].index.tolist()
            queries = queries + empty_queries
            del empty_queries
            
        del queries_names
            
        documents = pd.DataFrame(documents,columns=list(['dataframe_index']))
        queries = pd.DataFrame(queries,columns=list(['dataframe_index']))
        targets_pairs = pd.DataFrame(targets_pairs,columns=list(['query_row_index','document_col_index']))
        
         
        queries.to_hdf(files_path,'queries')
        documents.to_hdf(files_path,'documents')
        targets_pairs.to_hdf(files_path,'targets_pairs')
        source_dataframe.to_hdf(files_path,'source_dataframe')
        susp_dataframe.to_hdf(files_path,'susp_dataframe')

    return documents, queries, targets_pairs, source_dataframe, susp_dataframe, dataset_encoding


def _load_ir_task_content(documents, queries, targets_pairs, source_dataframe, susp_dataframe, dataset_encoding):
    '''
        loading content from files
    '''    

    col = targets_pairs.loc[:,'document_col_index'].values.tolist()
    row = targets_pairs.loc[:,'query_row_index'].values.tolist()
    data = np.ones(targets_pairs.shape[0])

    dataset_target = coo_matrix((data, (row, col)), shape=(queries.shape[0], documents.shape[0]))
    del col,row,data
    dataset_target = dataset_target.tolil()

    for id_row,source_row in documents.iterrows():
        contenti = load_file_content(source_dataframe, source_dataframe.loc[source_row['dataframe_index'],'reference'], dataset_encoding)
        print(id_row,'=>',len(contenti))
        if (len(contenti) < 2):
            print(source_row)
            print(source_dataframe.loc[source_row['dataframe_index'],'reference'])
#             print(__file_path(source_dataframe, source_dataframe.loc[source_row['dataframe_index'],'reference']))
            input('empty document!')
 
    for id_row,susp_row in queries.iterrows():
        contenti = load_file_content(susp_dataframe, susp_dataframe.loc[susp_row['dataframe_index'],'reference'], dataset_encoding)
        print(id_row,'=>',len(contenti))
        if (len(contenti) < 2):
            print(susp_row)
            print(susp_dataframe.loc[susp_row['dataframe_index'],'reference'])
#             print(__file_path(susp_dataframe, susp_dataframe.loc[susp_row['dataframe_index'],'reference']))
            input('empty query!')

    documents_content = Parallel(n_jobs=2,backend="threading",verbose=1)(delayed(load_file_content)(source_dataframe, source_dataframe.loc[source_row['dataframe_index'],'reference'], dataset_encoding) for _,source_row in documents.iterrows())        
    queries_content = Parallel(n_jobs=2,backend="threading",verbose=1)(delayed(load_file_content)(susp_dataframe, susp_dataframe.loc[susp_row['dataframe_index'],'reference'], dataset_encoding) for _,susp_row in queries.iterrows())

    del source_dataframe, susp_dataframe
    
    documents = pd.DataFrame({'dataframe_index':documents.loc[:,'dataframe_index'].values.tolist(),'content':documents_content})
    queries = pd.DataFrame({'dataframe_index':queries.loc[:,'dataframe_index'].values.tolist(),'content':queries_content})
    
    del documents_content, queries_content
    return queries, documents, dataset_target, dataset_encoding

def load_as_ir_task(allow_queries_without_relevants = True, language_filter="ALL"):
    '''
        ???? source documents (index) X ???? suspicious documents (queries) (allowing queries without relevants)
    '''
    documents, queries, targets_pairs, source_dataframe, susp_dataframe, dataset_encoding = _load_as_ir_task_without_content(allow_queries_without_relevants = True,language_filter=language_filter)
    queries, documents, dataset_target, dataset_encoding = _load_ir_task_content(documents, queries, targets_pairs, source_dataframe, susp_dataframe, dataset_encoding)
    del source_dataframe, susp_dataframe
    return queries, documents, dataset_target, dataset_encoding


def load_sample_as_ir_task(sample_threshold = 100,language_filter="ALL"):    
    documents, queries, targets_pairs, source_dataframe, susp_dataframe, dataset_encoding = _load_as_ir_task_without_content(allow_queries_without_relevants = True,language_filter=language_filter)
    sample_targets_pairs = targets_pairs[targets_pairs.query_row_index < sample_threshold]
    sample_document_indexes = list(set(sample_targets_pairs.loc[:,'document_col_index'].tolist()))
    sample_queries_indexes = list(set(sample_targets_pairs.loc[:,'query_row_index'].tolist()))

    col,row = [],[]
    for _,sample_target_row in sample_targets_pairs.iterrows():
        new_row_id = sample_queries_indexes.index(sample_target_row['query_row_index'])
        new_col_id = sample_document_indexes.index(sample_target_row['document_col_index'])
        col.append(new_col_id)
        row.append(new_row_id)
#         print(sample_target_row['document_col_index'],'=',new_col_id)
#         print(sample_target_row['query_row_index'],'=',new_row_id)

    col = list(col)
    row = list(row)
    data = np.ones(len(row))

    dataset_target = coo_matrix((data, (row, col)), shape=(len(sample_queries_indexes), len(sample_document_indexes)))
    del col,row,data
    dataset_target = dataset_target.tolil()
    
#     print(sample_targets_pairs)
#     print(sample_targets_pairs.shape,'=',dataset_target.shape,len(row),len(col),len(data))

    documents_content = Parallel(n_jobs=2,backend="threading",verbose=1)(delayed(load_file_content)(source_dataframe, source_dataframe.loc[documents.loc[source_col_id,'dataframe_index'],'reference'], dataset_encoding) for source_col_id in sample_document_indexes)        
#     print(len(documents_content))
    queries_content = Parallel(n_jobs=2,backend="threading",verbose=1)(delayed(load_file_content)(susp_dataframe, susp_dataframe.loc[queries.loc[susp_row_id,'dataframe_index'],'reference'], dataset_encoding) for susp_row_id in sample_queries_indexes)        
#     print(len(queries_content))
    
    documents = pd.DataFrame({'dataframe_index':documents.loc[sample_document_indexes,'dataframe_index'].values.tolist(),'content':documents_content})
    queries = pd.DataFrame({'dataframe_index':queries.loc[sample_queries_indexes,'dataframe_index'].values.tolist(),'content':queries_content})

    del sample_document_indexes, sample_queries_indexes, sample_targets_pairs
    del source_dataframe, susp_dataframe
    
    del documents_content, queries_content

#     exit()
    
    return queries, documents, dataset_target, dataset_encoding

if __name__ == '__main__':
#     queries, corpus_index, dataset_target,dataset_encoding = load_as_ir_task()
#     print(len(queries), len(corpus_index), dataset_target.shape)
#     del queries, corpus_index, dataset_target,dataset_encoding
#      
#     queries, corpus_index, dataset_target,dataset_encoding = load_as_ir_task(allow_queries_without_relevants=False)
#     print(len(queries), len(corpus_index), dataset_target.shape)
#     del queries, corpus_index, dataset_target,dataset_encoding
#  
#     queries, corpus_index, dataset_target,dataset_encoding = load_as_ir_task(allow_queries_without_relevants=False,language_filter="ALL")
#     print(len(queries), len(corpus_index), dataset_target.shape)
#     del queries, corpus_index, dataset_target,dataset_encoding

    queries, corpus_index, dataset_target,dataset_encoding = load_as_ir_task(allow_queries_without_relevants=True,language_filter="en")
    print(len(queries), len(corpus_index), dataset_target.shape)
    del queries, corpus_index, dataset_target,dataset_encoding

#     queries, corpus_index, dataset_target,dataset_encoding = load_as_ir_task(allow_queries_without_relevants=False,language_filter="es")
#     print(len(queries), len(corpus_index), dataset_target.shape)
#     del queries, corpus_index, dataset_target,dataset_encoding
# 
#     queries, corpus_index, dataset_target,dataset_encoding = load_as_ir_task(allow_queries_without_relevants=False,language_filter="de")
#     print(len(queries), len(corpus_index), dataset_target.shape)
#     del queries, corpus_index, dataset_target,dataset_encoding

#     for filesi in [queries,corpus_index]:
#         for i in range(filesi.shape[0]):
#             filesi[i]
#             print(i,'/',filesi.shape[0],':',filesi[i][:50],'[X]',filesi[i][:50].encode(dataset_encoding))
# #             input('next')    