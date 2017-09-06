import pandas as pd
import numpy as np
import os.path
import glob
from datasets.extractors import datasets_extractors
import xml.etree.ElementTree as ET
import re

def load_to_pandas():
    '''
        loading 771 PA documents and 917 newspapers documents.
        ( extracted from folders sgml's)
        grouping pages from the same source news results : 265 PA documents and 917 newspapers documents.
        
        return dataset info as two pandas.Dataframe
    '''
    
    root_path = datasets_extractors['DATASETS_PATH']['meter_corpus']

    filenames_list = [
                    "courts",
                    "showbiz",
                    ]
    encoding_name = "latin1"
    dataframe = None

    for doc_type in (
                    "PA", 
                    "newspapers",
                     ):   
        for filenamesi in filenames_list:
            folder_path = os.path.join(root_path,doc_type,"annotated",filenamesi)
            for date_dir in os.listdir(folder_path):
                if date_dir != "readme.txt":
                    for domain_dir in os.listdir(os.path.join(folder_path,date_dir)):
                        file_pattern = os.path.join(folder_path,date_dir,domain_dir,'*.sgml')
                        docs = glob.glob(file_pattern)
                        
                        if doc_type == "PA":
                            source_contenti = ""
                            
                        for doc in docs:
                            with open(doc,'r', encoding=encoding_name) as f:
                                linesi = "\n".join(f.readlines()).replace('&', " AND ")
                                linesi = linesi.replace('<\n','\n').replace('<\r','\n')
                                tree = ET.fromstring(linesi)
                                root = tree
                                atribs_dict = root.attrib.copy()
                                atribs_dict['content'] = ""
                                for child in root:
#                                     print("\t",child.tag)
                                    if child.tag=="body":
                                        for offspring in child:
#                                             print(offspring.text.replace('\n\n','\n').replace('\n\n','\n'),end=None)
                                            if doc_type == "PA":
                                                source_contenti += offspring.text
                                            else:
                                                atribs_dict['content'] += offspring.text                                        

                                if not doc_type == "PA":
                                    df2 = pd.DataFrame([atribs_dict.values()],columns=list(atribs_dict.keys()))
                                    if dataframe is None:
                                        dataframe = df2
                                    else:
                                        dataframe = dataframe.append(df2, ignore_index=True)
                        
                        '''
                            grouping pages from the same source news
                        '''
                        if doc_type == "PA":
                            atribs_dict['content'] = source_contenti
                            df2 = pd.DataFrame([atribs_dict.values()],columns=list(atribs_dict.keys()))
                            if dataframe is None:
                                dataframe = df2
                            else:
                                dataframe = dataframe.append(df2, ignore_index=True)
                                
    return dataframe,encoding_name

def load_as_derived_pairs():
    '''
        courts: 205 PA documents X 742 newspapers documents = 152110 pairs  
        shobiz:  60 PA documents X 175 newspapers documents =  10500 pairs 
        
        wholly-derived      53795 pairs 
        partially-derived   74025 pairs
        non-derived         34790 pairs
        
        162610 plagiarism pairs to eval.
    '''

    meter_original,dataset_encoding = load_to_pandas()
    
    dataset_documents = []
    dataset_target = []
    class_labels = ['wholly-derived','partially-derived','non-derived']
    
    p = re.compile('\n([ \r\n])*')
    p1 = re.compile('--*')
    
    for domaini in [
                    "courts",
                    "showbiz"
                    ]:
        meter_by_domain = meter_original[meter_original.domain == domaini]

        source_documents = meter_by_domain[pd.isnull(meter_by_domain.classification)]
        susp_documents = meter_by_domain[pd.notnull(meter_by_domain.classification)]
        
        print(domaini,':',source_documents.shape,susp_documents.shape)

        for _, source_row in source_documents.iterrows():
            for _, susp_row in susp_documents.iterrows():
        
                dataset_documents.append((
                                        p.sub('\n',p1.sub('',source_row["content"].replace('.','. '))),
                                        p.sub('\n',p1.sub('',susp_row["content"].replace('.','. ')))
#                                             source_row["content"],susp_row["content"]
                                          ))
                dataset_target.append(class_labels.index(susp_row['classification']))

#         print(len(dataset_documents))
        del meter_by_domain,source_documents,susp_documents
    
    del meter_original
    dataset_target = np.array(dataset_target)
    dataset_labels = class_labels
    dataset_documents = np.matrix(dataset_documents,dtype=np.ndarray)
    return dataset_documents ,dataset_target , dataset_labels, dataset_encoding

def load_as_pairs():
    '''
        courts: 205 PA documents X 742 newspapers documents = 152110 pairs  
        shobiz:  60 PA documents X 175 newspapers documents =  10500 pairs 
        
        162610 plagiarism pairs to eval.
    '''

    dataset_documents ,dataset_target ,dataset_labels ,dataset_encoding = load_as_derived_pairs()
    dataset_target = dataset_target != dataset_labels.index("non-derived")
    
    del dataset_labels

    return dataset_documents ,dataset_target ,0 ,0,dataset_encoding

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
    
    
    
    