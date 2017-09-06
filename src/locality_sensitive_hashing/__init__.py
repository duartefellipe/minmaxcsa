import numpy as np
from sklearn.externals.joblib.parallel import Parallel, delayed
from sklearn.metrics.pairwise import pairwise_distances
from math import ceil, sqrt, floor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils import shuffle
from scipy.sparse.lil import lil_matrix

from time import time
from sklearn.preprocessing.data import binarize
from sklearn.neighbors import NearestNeighbors
import collections
from networkx.classes.function import neighbors
from sklearn.utils.fixes import argpartition
from pickle import FALSE
from random import sample
from scipy import argsort
from pprint import pprint
from scipy.sparse.csr import csr_matrix
import six

def min_hashing(element_set, permutation):
    rows,_ = np.nonzero(element_set)
    min_id = np.Inf
    
    for linei in rows:
        perm_id = permutation[linei]
        if perm_id < min_id:
            min_id = perm_id

    return np.array([min_id])

def max_hashing(element_set, permutation):
    rows,_ = np.nonzero(element_set)
    max_id = -np.Inf
    
    for linei in rows:
        
        perm_id = permutation[linei]
        if perm_id > max_id:
            max_id = perm_id

    return np.array([max_id])

def minmax_hashing(element_set, permutation):
    rows,_ = np.nonzero(element_set)
    min_id,max_id = np.Inf,-np.Inf
    
    for linei in rows:
        perm_id = permutation[linei]
        if perm_id < min_id:
            min_id = perm_id
        if perm_id > max_id:
            max_id = perm_id

    return np.array([min_id,max_id])

def minmaxxor_hashing(element_set, permutation):
    rows,_ = np.nonzero(element_set)
    min_id,max_id = np.Inf,-np.Inf
    for linei in rows:
        
        perm_id = permutation[linei]
        if perm_id < min_id:
            min_id = perm_id
        if perm_id > max_id:
            max_id = perm_id

    return np.array([(min_id^max_id)])

def minmaxCSALowerBound_hashing(element_set, permutation):
    rows,_ = np.nonzero(element_set)
    min_id,max_id = np.Inf,-np.Inf
    
    for linei in rows:
        
        perm_id = permutation[linei]
        if perm_id < min_id:
            min_id = perm_id
        if perm_id > max_id:
            max_id = perm_id
            
    return np.array([min_id,max_id,ceil(sqrt(max_id**2 - min_id**2))])

def minmaxCSAFullBound_hashing(element_set, permutation):
    rows,_ = np.nonzero(element_set)
    min_id,max_id = np.Inf,-np.Inf
    
    for linei in rows:
        
        perm_id = permutation[linei]
        if perm_id < min_id:
            min_id = perm_id
        if perm_id > max_id:
            max_id = perm_id

    set_len = sqrt(max_id**2 - min_id**2)
    return np.array([min_id,max_id,ceil(set_len), floor(set_len)])

def justCSALowerBound_hashing(element_set, permutation):
    rows,_ = np.nonzero(element_set)
    min_id,max_id = np.Inf,-np.Inf
    
    for linei in rows:
        
        perm_id = permutation[linei]
        if perm_id < min_id:
            min_id = perm_id
        if perm_id > max_id:
            max_id = perm_id
            
    return np.array([ceil(sqrt(max_id**2 - min_id**2)),])

def justCSAFullBound_hashing(element_set, permutation):
    rows,_ = np.nonzero(element_set)
    min_id,max_id = np.Inf,-np.Inf
    
    for linei in rows:
        
        perm_id = permutation[linei]
        if perm_id < min_id:
            min_id = perm_id
        if perm_id > max_id:
            max_id = perm_id

    set_len = sqrt(max_id**2 - min_id**2)
    return np.array([ceil(set_len), floor(set_len)])

def pairwise_jaccard_similarity(set_per_row):
    
    
    return pairwise_distances(set_per_row,metric='jaccard', n_jobs=1)
    
#     results = Parallel(
#         n_jobs=100,backend='threading',batch_size=10
#     )(
#             delayed(jaccard_similarity)(i,j,set_per_row)
#                 for i in range(set_per_row.shape[0])
#                 for j in range(set_per_row.shape[0])
#     )
#     
#     results = np.array(results)
#     
#     return results.reshape((set_per_row.shape[0],set_per_row.shape[0]))


class LSHTransformer(BaseEstimator, TransformerMixin):
    """Transform a count matrix to a binary Locality-sensitive Hashing representations
    
        
        selection_function returns integer values to represent each permutation
    
        The goal of using LSH-like instead of the raw frequencies of occurrence of a
        token in a given document is to ...
        
        Parameters
        ----------
    
        n_permutations : integer how many times attributes will be permuted. 
        
        selection_function : 'min', 'minmax','...' or default='min', 
            select features of each attribute permutation.
            
        n_jobs : int
            The number of jobs to use for the computation. This works by breaking
            down the pairwise matrix into n_jobs even slices and computing them in
            parallel.
    
            If -1 all CPUs are used. If 1 is given, no parallel computing code is
            used at all, which is useful for debugging. For n_jobs below -1,
            (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
            are used.

        References
        ----------
    
    """
    def __init__(self, n_permutations, selection_function = min_hashing, n_jobs=-1):
        self.n_permutations, self.selection_function, self.n_jobs = n_permutations, selection_function, n_jobs
        self.selection_size = None
        
    def measure_selection_function(self,current_document,items_to_permute,approach_permutation_count,selection_size,selection_function):
        '''
            Parameters
            ----------
            current_document : sparse matrix, [n_features, n_samples]
                a matrix of term/token counts
            """
        '''
        approach_finger = np.empty((1,selection_size*approach_permutation_count),np.int)
        approach_time = np.zeros((approach_permutation_count))
        
#         print(approach_finger.shape,'=',approach_permutation_count,' x ', len(selection_function(np.ones((1,1)), items_to_permute)))
        
        '''
            scores don't need to be binary
        '''
#         current_document = binarize(current_document)

    #     print("start")
        for i in range(approach_permutation_count):
            t0 = time() 
            pi_results = selection_function(current_document, shuffle(items_to_permute,random_state=i))
            approach_finger[0,selection_size*i:selection_size*(i+1)] = pi_results
            approach_time[i] = time() - t0
        del current_document
        
#         print("end in %4.2f"%(approach_time.sum()))
#         print(approach_finger)

        return (approach_finger, approach_time)
    
    def fit(self, X, y=None):
        """evaluate selection size

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """
        t0 = time()
        self.items_to_permute = [i+1 for i in range(X.shape[1])]
        
        self.selection_size = len(self.selection_function(np.ones((1,1)), self.items_to_permute))
         
        self.fit_time = time()-t0

        return self

    def transform(self, X):
        """Transform a count matrix to ..
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts

        Returns
        -------
        vectors : sparse matrix, [n_samples, n_features]
        sample's time-to-transform : sparse matrix, [n_samples] 
        """
        if self.selection_size == None:
            raise('must be fitted!')
        else:
            approach_all_finger = np.empty((X.shape[0],self.selection_size*self.n_permutations),np.int)
            approach_all_time = np.zeros((approach_all_finger.shape))
            r = np.array(Parallel(n_jobs=self.n_jobs,backend="multiprocessing",verbose=0)(delayed(self.measure_selection_function)(X[i,:].T, self.items_to_permute, self.n_permutations,self. selection_size, self.selection_function) for i in range(X.shape[0])))        
           
            approach_all_finger= np.vstack(r[:,0])
            approach_all_time = np.vstack(r[:,1])
            
            return (approach_all_finger,approach_all_time)  
        


'''
    Inverted index to improve retrieval performance
'''
       
class InvertedIndex(BaseEstimator):
    def __init__(self, index_function,scorer_function):
        self.index_function,self.scorer_function = index_function,scorer_function

    def index_collection(self,X):
        """ inverted index (ii) creation 

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
        """
        self.collection_size = X.shape[0]
        self.ii,self.index_time = self.index_function(X)
        
        return self.index_time
        
    def score_queries(self,X):
        """ scores(count) collection document collisions against queries features  

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
        """
        scores, time_to_score = self.scorer_function(self.ii,self.collection_size, X)

        return scores, time_to_score
    
def indexfunction_instance_atrib(X):
    t0 = time()
    ii = collections.defaultdict(list)
    for d_index, t_index in np.transpose(np.nonzero(X)):
        ii[t_index].append(d_index)
    t0 = time() - t0

#     print('time1:',t0)
    return ii, t0

def scorefunction_instance_atrib(ii, collection_size, X):
    scores = np.zeros((X.shape[0], collection_size))

    t0 = time()
    for q_index, t_index in np.transpose(np.nonzero(X)):
        colision_indexes = ii[t_index]
        scores[q_index,colision_indexes] = scores[q_index,colision_indexes] + 1 
    
    t0 = time() - t0
    
    return scores, t0

def indexfunction_lsh(X):
    t0 = time()
    ii = collections.defaultdict(lambda : collections.defaultdict(list))
    
    for d_index, t_index in np.transpose(np.nonzero(X)):
            ii[t_index][X[d_index, t_index]].append(d_index)
    #         print(t_index,X[d_index, t_index],'=',len(ii[t_index][X[d_index, t_index]]))
    t0 = time() - t0
    
#     print('time2:',t0)
    
#     for keyi,dicti in ii.items():
#         for keyj,listj in dicti.items():
#             print(keyi,keyj,'=',len(listj))
#             print(listj)

    return ii, t0

def scorefunction_lsh(ii, collection_size, X):
    scores = np.zeros((X.shape[0], collection_size))

    t0 = time()
    for q_index, t_index in np.transpose(np.nonzero(X)):
        colision_indexes = ii[t_index]
        term_permutation_index = X[q_index, t_index]
        scores[q_index,colision_indexes[term_permutation_index]] = scores[q_index,colision_indexes[term_permutation_index]] + 1 
    
    t0 = time() - t0
    
    return scores, t0

'''
    Nearest neighbors + Inverted Index
'''

class InvertedIndexNearestNeighborsBaseEstimator(BaseEstimator, TransformerMixin):
    """ Evaluate the Nearest Neighborhood for each predicted instance
    
        retrieve nearest trained (X's from fit) instances index for each predicted instance.
        can return evaluated instances.
        
        the metric is counting atributes colisions on the inverted index (faster than pairwise compare)   
            
            Parameters
            ----------
            
            ...
    
            References
            ----------
        
    """

    def __init__(self, invertedIndex, n_neighbors,sort_neighbors=False):
        self.ii, self.n_neighbors,self.sort_neighbors = invertedIndex, n_neighbors, sort_neighbors
    
    def fit(self, X, y=None):
        """evaluate nearest neighbors distances 

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """
        self.fit_time = self.ii.index_collection(X) 
        return self
    

    def transform(self, X):
        scores,time_to_score = self.ii.score_queries(X)
        dist = -1*scores
        
        t0 = time()
        n_samples, _ = X.shape
        sample_range = np.arange(n_samples)[:, None]

        neigh_ind = argpartition(dist, self.n_neighbors - 1, axis=1)
        neigh_ind = neigh_ind[:, :self.n_neighbors]
        
        # argpartition doesn't guarantee sorted order, so we sort again
        if self.sort_neighbors:
            neigh_ind = neigh_ind[sample_range, np.argsort(dist[sample_range, neigh_ind])]
        
        del dist
        
        sorted_scores = scores[sample_range, neigh_ind]

        time_to_score = time_to_score + t0 - time()

       
#         print(scores)
#         print('self.n_neighbors:',self.n_neighbors)
#         print(sorted_scores)
#         print(neigh_ind)
        
        return neigh_ind, sorted_scores, time_to_score
    
class InvertedIndexNearestNeighbors(InvertedIndexNearestNeighborsBaseEstimator):
    def __init__(self, n_neighbors, sort_neighbors=False):
        InvertedIndexNearestNeighborsBaseEstimator.__init__(self, InvertedIndex(indexfunction_instance_atrib,scorefunction_instance_atrib), n_neighbors, sort_neighbors)

class LSHIINearestNeighbors(InvertedIndexNearestNeighborsBaseEstimator):
    def __init__(self, n_neighbors, sort_neighbors=False):
        InvertedIndexNearestNeighborsBaseEstimator.__init__(self, InvertedIndex(indexfunction_lsh, scorefunction_lsh), n_neighbors, sort_neighbors)
#         print(self.__class__.__name__)


############################################################################################################################################
#
# Another approaches to do Nearest Neighbors using indexes schemes  
#
#
############################################################################################################################################

'''
    Permutation Based Index !
'''
class PermutationBasedIndex(InvertedIndex):
    def __init__(self, bucket_count, reference_set_size, prunning_size, ref_sel_threshold = 0.5):
        self.bucket_count,self.reference_set_size,self.prunning_size,self.ref_sel_threshold = bucket_count,reference_set_size,prunning_size,ref_sel_threshold

    def reference_set_selection(self,X):
        '''
            Distributed Selection: close reference points are neglected based on a threshold
             
        '''
        t0 = time()
        
        '''
            randomly selects the first reference point
        '''
#         set_id = sample(range(X.shape[0]),1)
        
        set_id = [ceil(X.shape[0]/2)]

        current_id = set_id[0]
        
        first_reference = np.nonzero(pairwise_distances(X[current_id,:],X, metric='cosine', n_jobs=1) > self.ref_sel_threshold)[1]

        i = 0        
        while len(set_id) < self.reference_set_size and i < len(first_reference):
            current_id = first_reference[i]
            i += 1
            current_reference = np.nonzero(pairwise_distances(X[current_id,:],X[set_id,:], metric='cosine', n_jobs=1) < self.ref_sel_threshold)[1]
            
            if len(current_reference) == 0:
                set_id.append(current_id)

            del current_reference
        
#         print(set_id)
#         print(len(set_id),' x ', self.reference_set_size)
#         set_id = sample(range(X.shape[0]),self.reference_set_size)
        return set_id, time()-t0
    
    def relative_ordered_list(self, X, d_index):
        
        t0 = time()
        
#         print(X.shape)
        distances = pairwise_distances(X[d_index,:],self.index_features[self.reference_set_id,:], metric='cosine', n_jobs=1)
        
        lr = argsort(distances,axis=1)
        return lr[:,:self.prunning_size], time() - t0
    
    def index_collection(self,X):
        """ inverted index (ii) creation 

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
        """
        self.collection_size = X.shape[0]
        self.ii = collections.defaultdict(lambda : collections.defaultdict(list))
        
        self.reference_set_id, self.index_time = self.reference_set_selection(X)
        self.index_features = X
         
        self.bij = np.empty((X.shape[0],len(self.reference_set_id)),np.int)
        for d_index in range(X.shape[0]):
            d_list_in_r, d_list_time = self.relative_ordered_list(X, d_index)

            t0 = time()
            for j in range(d_list_in_r.shape[1]):
                self.bij[d_index,j] = ceil(((self.bucket_count-1)*d_list_in_r[0,j])/len(self.reference_set_id))
                self.ii[j][self.bij[d_index,j]].append(d_index) 
            
            self.index_time += time() - t0
            self.index_time += d_list_time 
            
        return self.index_time
        
    def score_queries(self,X):
        """ scores(count) collection document collisions against queries features  

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
        """


        scores = np.zeros((X.shape[0], self.collection_size))
        
        time_to_score = 0
        
        for q_index in range(X.shape[0]):
            q_list_in_r, q_list_time = self.relative_ordered_list(X, q_index)

            t0 = time()
            for j in range(q_list_in_r.shape[1]):
                bqj = ceil(((self.bucket_count-1)*q_list_in_r[0,j])/len(self.reference_set_id))

                scores[q_index,self.ii[j][bqj-1]] = scores[q_index,self.ii[j][bqj-1]] + 1
                scores[q_index,self.ii[j][bqj]] = scores[q_index,self.ii[j][bqj]] + 2
                scores[q_index,self.ii[j][bqj+1]] = scores[q_index,self.ii[j][bqj+1]] + 1
                    
            
            time_to_score += time() - t0
            time_to_score += q_list_time 

        return scores, time_to_score
        
class PBINearestNeighbors(InvertedIndexNearestNeighborsBaseEstimator):
    def __init__(self, bucket_count, reference_set_size, prunning_size, ref_sel_threshold, n_neighbors, sort_neighbors):
        self.bucket_count, self.reference_set_size, self.prunning_size, self.ref_sel_threshold = bucket_count, reference_set_size, prunning_size, ref_sel_threshold
        InvertedIndexNearestNeighborsBaseEstimator.__init__(self, self.create_inverted_index(), n_neighbors, sort_neighbors)

    def create_inverted_index(self):
        return PermutationBasedIndex(self.bucket_count, self.reference_set_size, self.prunning_size, self.ref_sel_threshold)    
        
    def set_params(self, **params):
        InvertedIndexNearestNeighborsBaseEstimator.set_params(self,**params)
        self.ii = self.create_inverted_index()
