import re
class Preprocessor():
    def __init__(self, punctuation = True, url = True, numbers = True):
        self.punctuation = punctuation
        self.url = url
        self.numbers = numbers

    @staticmethod
    def remove_number(sentence:str) -> str:
        """Remove numbers in the text with re
        Args: 
            sentence: sentence with possible numbers
        Returns:
            sentence: sentence with numbers removed"""
        sentence = re.sub(r'\d+', ' ', sentence)
        return sentence
    
    @staticmethod
    def remove_punctuation(sentence:str) ->str:
        """Remove possible punctuations from the text with re
        Args:
            sentence: sentence with possible punctuations
        Returns:
            sentence: sentence with punctuations removed"""
        sentence = re.sub(r'(https|http)?://(\w|\.|/|\?|=|&|%)*\b', ' ', sentence)
        return sentence

    @staticmethod
    def remove_url(sentence:str) ->str:
        """Remove possible punctuations from the text with re
        Args:
            sentence: sentence with possible url
        Returns:
            sentence: sentence without punctuations url"""
        sentence = re.sub(r'[^\w\s]', ' ', sentence)
        return sentence
    
    @staticmethod
    def format( data:str):
        """Remove possible citations, brackets, digits, tables and figurs from the text with re
        Args:
            sentence: citations, brackets, digits, tables and figurs
        Returns:
            sentence: sentence without citations, brackets, digits, tables and figurs"""
        # replace citation with <cit> tag
        data = re.sub(r'[[][0-9]+[,0-9/-]*[]]',r' <cit> ',data)
        data = re.sub(r'[[][0-9]+[", ",0-9/-]*[]]',r' <cit> ',data)
        # remove text in brackets
        data = re.sub(r'\([^)]+\)','',data)
        data = re.sub(r'\[.*?\]','',data)
        # replace digits with <dig> tag
        data = re.sub(r'd <dig> ',' <dig> ', data)
        data = re.sub(r'(<dig> )+',' <dig> ', data)
        # remove table and figures
        data = re.sub(r'\ntable \d+.*?\n',r'',data)
        data = re.sub(r'.*\t.*?\n',r'',data)
        data = re.sub(r'\nfigure \d+.*?\n',r'',data)
        # data = re.sub(r'[(]figure \g+.*?[)]',r'',data)
        data = re.sub(r'[(]fig. \d+.*?[)]',r'',data)
        data = re.sub(r'[(]fig . \d+.*?[)]',r'',data)
        data = re.sub(r'[(]table \d+.*?[)]',r'',data)
        return data

    def apply(self, sentence:str) -> str:
        """ Apply the preprocessing rules to the sentence
        Args:
            sentence: raw sentence
        Returns:
            sentence: clean sentence
        """
        sentence = sentence.lower()
        sentence = sentence.replace('<unk>', '')
        sentence = sentence.replace('\n', ' ')
        if self.numbers:
            sentence = Preprocessor.remove_number(sentence=sentence)
        if self.url:
            sentence = Preprocessor.remove_number(sentence=sentence)
        if self.punctuation:
            sentence = Preprocessor.remove_punctuation(sentence=sentence)
        Preprocessor.format(data = sentence)
        return sentence