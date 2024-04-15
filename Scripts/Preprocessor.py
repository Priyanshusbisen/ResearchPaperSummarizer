import re
class Preprocessor():
    def __init__(self, punctuation = True, url = True, numbers = True):
        self.punctuation = punctuation
        self.url = url
        self.numbers = numbers

    @staticmethod
    def remove_number(self, sentence:str) -> str:
        """Remove numbers in the text with re
        Args: 
            sentence: sentence with possible numbers
        Returns:
            sentence: sentence with numbers removed"""
        sentence = re.sub(r'\d+', ' ', sentence)
        return sentence
    
    @staticmethod
    def remove_punctuation(self, sentence:str) ->str:
        """Remove possible punctuations from the text with re
        Args:
            sentence: sentence with possible punctuations
        Returns:
            sentence: sentence with punctuations removed"""
        sentence = re.sub(r'(https|http)?://(\w|\.|/|\?|=|&|%)*\b', ' ', sentence)
        return sentence

    @staticmethod
    def remove_url(self, sentence:str) ->str:
        """Remove possible punctuations from the text with re
        Args:
            sentence: sentence with possible url
        Returns:
            sentence: sentence with punctuations url"""
        sentence = re.sub(r'[^\w\s]', ' ', sentence)
        return sentence

    def apply(self, sentence:str) -> str:
        """ Apply the preprocessing rules to the sentence
        Args:
            sentence: raw sentence
        Returns:
            sentence: clean sentence
        """
        sentence = sentence.lower()
        sentence = sentence.replace('<unk>', '')
        if self.numbers:
            sentence = Preprocessor.remove_number(sentence=sentence)
        if self.url:
            sentence = Preprocessor.remove_number(sentence=sentence)
        if self.punctuation:
            sentence = Preprocessor.remove_punctuation(sentence=sentence)
        return sentence