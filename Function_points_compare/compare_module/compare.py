import jieba
import difflib

class Compare:
    """
        Compare methods set
    """
    def jieba_set_compare(query, candidate):
        assert len(query)!=0
        assert len(candidate)!=0

        set_query = set(jieba.lcut(query, cut_all=True))
        set_candidate = set(jieba.lcut(candidate, cut_all=True))
 
        return len(set_query&set_candidate) / (len(set_candidate)+1)
    

    def SequenceMatcher(query, candidate):

        return difflib.SequenceMatcher(None, query, candidate).quick_ratio()