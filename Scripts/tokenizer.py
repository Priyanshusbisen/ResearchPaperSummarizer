from preprocessor import Preprocessor
class Tokenizer:
    def __init__(self, sos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<unk>', mask_token='<mask>'):
        # Special tokens.
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token
        
        self.vocab = { sos_token: 0, eos_token: 1, pad_token: 2, unk_token: 3, mask_token: 4 }  # token -> id
        
        self.preprocessor = Preprocessor()

    @property
    def sos_token_id(self):
        """ Create a property method.
            You can use self.sos_token_id or tokenizer.sos_token_id to get the id of the sos_token.
        """
        return self.vocab[self.sos_token]

    @property
    def eos_token_id(self):
        return self.vocab[self.eos_token]

    @property
    def pad_token_id(self):
        return self.vocab[self.pad_token]

    @property
    def unk_token_id(self):
        return self.vocab[self.unk_token]

    @property
    def mask_token_id(self):
        return self.vocab[self.mask_token]
        
        
    def encode(self, sentence: str):
        """ Fit the tokenizer using all sentences.
        1. Tokenize the sentence by splitting with spaces.
        2. Record the occurrence of all tokens
        3. Construct the token to index (self.vocab) map and the inversed map (self.inverse_vocab) based on the occurrence. The token with a higher occurrence has the smaller index

        Args:
            sentences: All sentences in the dataset.
        """
        tokens = [self.sos_token_id] + self.preprocessor.apply(sentence.strip()).split() + [self.eos_token_id]
        
        return tokens
        
    # def encode(self, sentence: str) -> list[list[int]]:
    #     """ Encode the sentences into token ids
    #         Note: 1. if a token in a sentence does not exist in the fit encoder, we ignore it.
    #               2. If the number of tokens in a sentence is less than two, we ignore this sentence.
    #               3. Note that, for every sentence, we will add an sos_token, i.e., the id of <s> at the start of the sentence,
    #                  and add an eos_token, i.e., the id of </s> at the end of the sentence.
    #     Args:
    #         sentences: Raw sentences
    #     Returns:
    #         sent_token_ids: A list of id list
    #     """
    #     token_ids = []
    #     tokens = self.preprocessor.apply(sentence.strip()).split()
    #     for token in tokens:
    #         if token == '<unk>':
    #             continue
    #         if token in self.vocab:
    #             token_ids.append(self.vocab[token])
    #     if len(token_ids) <= 1:
    #         return
    #     token_ids = [self.sos_token_id] + token_ids + [self.eos_token_id]
    #     print(token_ids)
    #     return token_ids
    
    # def get_sent_token_ids(self):
    #     return self.sent_token_ids