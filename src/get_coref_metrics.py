#####################################################################################################
# implements coreference scorer metrics

# @InProceedings{pradhan-EtAl:2014:P14-2,
# author    = {Pradhan, Sameer  and  Luo, Xiaoqiang  and  Recasens, Marta  and  Hovy, Eduard  and  Ng, Vincent  and  Strube, Michael},
# title     = {Scoring Coreference Partitions of Predicted Mentions: A Reference Implementation},
# booktitle = {Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
# month     = {June},
# year      = {2014},
# address   = {Baltimore, Maryland},
# publisher = {Association for Computational Linguistics},
# pages     = {30--35},
# url       = {http://www.aclweb.org/anthology/P14-2006}
# }
#####################################################################################################

import numpy as np
#from sklearn.utils.linear_assignment_ import linear_assignment
from linear_assignment_ import linear_assignment
#https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/linear_assignment_.py

def get_f1(p, r):
    """
    :param p:
    :param r:
    :return: F1 score
    """
    if p + r:
        return 2 * p * r / (p + r)
    return 0.0

def get_prf(p_num, p_den, r_num, r_den):
    """
    :param p_num:
    :param p_den:
    :param r_num:
    :param r_den:
    :return: precision, recall, F1 score
    """
    p = p_num / p_den if p_den > 0 else 0.
    r = r_num / r_den if r_den > 0 else 0.
    return p, r, get_f1(p, r)

def get_inverse_coref_mapping(coref_graph):
    """
    :param coref_graph: {cluster_name: set([cluster_items])} dictionary
    :return: {cluster_item: cluster_name} dictionary
    """
    return {m: k for k, ms in coref_graph.items() for m in ms}

def _bcub(gold, response):
    """
    :param gold:
    :param response:
    :return:
    """
    gold_mapping = get_inverse_coref_mapping(gold)
    response_mapping = get_inverse_coref_mapping(response)
    num = 0.0
    for key, value in gold_mapping.items():
        gold_cluster = gold.get(value, set())
        num += (len(gold_cluster & response.get(response_mapping.get(key), set())))/float(len(gold_cluster))
    return num, float(len(gold_mapping))

def bcub(gold, response):
    """
    :param gold:
    :param response:
    :return:
    """
    p_num, p_den = _bcub(response, gold)
    r_num, r_den = _bcub(gold, response)
    return p_num, p_den, r_num, r_den

def _ceafe_sim(a,b):
    """
    :param a:
    :param b:
    :return:
    """
    if a and b:
        return float(len(a&b))/float(len(a)+len(b))
    return 0.0

def _ceafe_max_overlap(alignment_matrix):
    """
    :param alignment_matrix:
    :return:
    """
    indices = linear_assignment(-alignment_matrix)
    # since, linear assignment minimize the weight perfect matching.
    return alignment_matrix[indices[:, 0], indices[:, 1]].sum()

def ceafe(gold, response):
    """
    :param gold:
    :param response:
    :return:
    """
    alignment_matrix = np.empty((len(gold), len(response)))
    response_values = list(response.values())
    for R, alignment_matrix_row in zip(gold.values(), alignment_matrix):
        alignment_matrix_row[:] = [_ceafe_sim(R, S) for S in response_values]
    p_num = r_num = _ceafe_max_overlap(alignment_matrix)
    p_den = sum(_ceafe_sim(S,S) for S in response_values)
    r_den = sum(_ceafe_sim(R,R) for R in gold.values())
    return p_num, p_den, r_num, r_den

def _muc(gold, response_mapping):
    """
    :param gold:
    :param response_mapping:
    :return:
    """
    num = 0.0
    den = 0.0
    for cluster in gold.values():
        intersection = set()
        num_unaligned = 0
        for m in cluster:
            try: intersection.add(response_mapping[m])
            except KeyError: num_unaligned += 1
        num += float(len(cluster)-num_unaligned-len(intersection))
        den += float(len(cluster)-1)
    return num, den

def muc(gold, response):
    """
    :param gold:
    :param response:
    :return:
    """
    p_num, p_den = _muc(response, get_inverse_coref_mapping(gold))
    r_num, r_den = _muc(gold, get_inverse_coref_mapping(response))
    return p_num, p_den, r_num, r_den

def get_conll_scores(gold, response):
    pn,pd,rn,rd = bcub(gold, response)
    bcub_p, bcub_r, bcub_f1 = get_prf(pn,pd,rn,rd)
    pn,pd,rn,rd = muc(gold, response)
    muc_p, muc_r, muc_f1 = get_prf(pn,pd,rn,rd)
    pn,pd,rn,rd = ceafe(gold, response)
    ceafe_p, ceafe_r, ceafe_f1 = get_prf(pn,pd,rn,rd)
    return bcub_p, bcub_r, bcub_f1, muc_p, muc_r, muc_f1, ceafe_p, ceafe_r, ceafe_f1, (bcub_f1+muc_f1+ceafe_f1)/3.0

#####################################################################################################
#Test implementation
#####################################################################################################
TEST = 0
if TEST:
    g1 = {1:set(['a','b','c']), 2:set(['d','e','f','g'])}
    g2 = {71:set(['x','y','z']), 72:set(['d','e','f','g'])}
    r = {11:set(['a','b']), 12:set(['d','c']), 13:set(['f','g','h','i'])}
    print(g1)
    print(g1[1])
    print(get_conll_scores(g1, r))
    print(get_conll_scores(g2, r))
#####################################################################################################
