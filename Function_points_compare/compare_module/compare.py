import jieba


class Compare:
    """
        Compare methods set
    """
    def jieba_set_compare(query, candidate):
        set_query = set(jieba.lcut(query, cut_all=True))
        set_candidate = set(jieba.lcut(candidate, cut_all=True))
 
        return len(set_query&set_candidate) / (len(set_candidate)+1)