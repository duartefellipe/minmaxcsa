## Minmax Circular Sector Arcs (MinMaxCSA): A Locality-Sensitive Hashing Method. 
MinmaxCSA deals with:
1. Information Retrieval problems (an approximated search method);
2. the curse of dimensionality;
3. Big Data volume and velocity.

On [MinMaxCSA paper](https://doi.org/10.1016/j.knosys.2017.08.013) two MinMaxCSA methods, Minmax Circular Sector Arcs Lower Bound (CSAL) and Minmax Circular Sector Arcs Full Bound (CSA) were proposed. Moreover, this source code repository evaluates MinMaxCSA methods on a external plagiarism detection task (known as Heuristic Retrieval) as described on paper's abstract:

> *"Heuristic Retrieval (HR) task aims to retrieve a set of documents from which the external plagiarism detection identifies plagiarized pieces of text. In this context, we present Minmax Circular Sector Arcs (MinMaxCSA) algorithms that treats HR task as an approximate k-nearest neighbor search problem. Moreover, MinMaxCSA algorithms aim to retrieve the set of documents with greater amounts of plagiarized fragments, while reducing the amount of time to accomplish the HR task."*

two experiments were run: The Pairwise Jaccard Similarity (PJS) and the Heuristic Retrieval (HR).

PSJ code usage:
1. enter on folder (path_to_src/)src/plagiarism_detection/extermal_plagiarism
2. run:
```
$ python PYTHONPATH=(path_to_src/)src/plagiarism_detection/pairwise_jaccard_comparison.py
```

HR code usage:
1. enter on folder (path_to_src/)src/plagiarism_detection/extermal_plagiarism
2. run:
```
$ python PYTHONPATH=(path_to_src/)src/plagiarism_detection/heuristic_retrieval.py
```

# Citation Credit:
if you end up using this code in published work, please cite:

```Fellipe Duarte, Danielle Caled, Geraldo Xexéo, MinMax Circular Sector Arc for External Plagiarism’s Heuristic Retrieval Stage, Knowledge-Based Systems, 2017, ISSN 0950-7051, URL 'https://doi.org/10.1016/j.knosys.2017.08.013' ```

bibtex format:
```article{
title = "MinMax Circular Sector Arc for External Plagiarism’s Heuristic Retrieval Stage",
journal = "Knowledge-Based Systems",
volume = "137",
number = "Supplement C",
pages = "1 - 18",
year = "2017",
note = "",
issn = "0950-7051",
doi = "http://dx.doi.org/10.1016/j.knosys.2017.08.013",
url = "http://www.sciencedirect.com/science/article/pii/S0950705117303696",
author = "Fellipe Duarte and Danielle Caled and Geraldo Xexéo",
}
```
