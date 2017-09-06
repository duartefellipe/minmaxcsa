import os.path

datasets_root_path = None
path_list = ["C:/Users/Fellipe/Documents/Datasets/",
			   ]

while datasets_root_path == None and len(path_list) > 0:
    pathi = path_list.pop()
    if os.path.exists(pathi):
        datasets_root_path = pathi

if datasets_root_path == None:
    raise Exception("Datasets root path not available!")


datasets_extractors = {
    "DATASETS_PATH" : {
             'cpc-11':os.path.join(datasets_root_path,'corpus-webis-cpc-11'),
             'MSRParaphrase':os.path.join(datasets_root_path,'MSRParaphraseCorpus'),
            'pan_plagiarism_corpus_2010': os.path.join(datasets_root_path,'PAN/Plagiarism detection/pan-plagiarism-corpus-2010'),
            'pan_plagiarism_corpus_2011': os.path.join(datasets_root_path,'PAN/Plagiarism detection/pan-plagiarism-corpus-2011'),

             'P4P':os.path.join(datasets_root_path,'P4P_corpus'),
            
            'short_plagiarised_answers_dataset': os.path.join(datasets_root_path,'short plagiarised answers corpus/corpus-20090418'),
            'meter_corpus':os.path.join(datasets_root_path,'meter_corpus'),
             'co_derivative':os.path.join(datasets_root_path,'PAN/co-derivative/wik-coderiv-corpus-original'),
             'cf': os.path.join(datasets_root_path,'Common IR collections /Cystic Fibrosis/cfc-xml'),
             'cranfield':os.path.join(datasets_root_path,'Common IR collections /cranfield'),
             'NerACL2010':os.path.join(datasets_root_path,'NerACL2010_Experiments/Data/WordEmbedding'),
    }
}
