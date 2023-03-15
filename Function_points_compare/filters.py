class Filter:
    """
        use different methods to process query
    """
    def __init__(self, stop_words_path="./stop_words.txt"):
        self.sw_path = stop_words_path


    def load_stopWords(self, file_path):
        with open(file_path,mode='r') as sw:
            sw_list = [word.strip('\n') for word in sw.readlines()]

        return set(sw_list)


    def stopWords_filter(self, query):
        sw = self.load_stopWords(file_path=self.sw_path)

        return "".join(list(set(query).difference(sw)))


    def regular_expression_filter(function_query):
        """
            TODO:Use regular expression process function_query
        """
        
        pass