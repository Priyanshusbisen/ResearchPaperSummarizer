import Preprocessor
class Tokenizer:
    def __init__(self, sos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<unk>', mask_token='<mask>'):
        # Special tokens.
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token
        
        self.vocab = { sos_token: 0, eos_token: 1, pad_token: 2, unk_token: 3, mask_token: 4 }  # token -> id
        self.inverse_vocab = { 0: sos_token, 1: eos_token, 2: pad_token, 3: unk_token, 4: mask_token }  # id -> token
        self.token_occurrence = { sos_token: 0, eos_token: 0, pad_token: 0, unk_token: 0, mask_token: 0 }  # token -> occurrence
        
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
        
    def __len__(self):
        """ A magic method that enable program to know the number of tokens by calling:
            ```python
            tokenizer = Tokenizer()
            num_tokens = len(tokenizer)
            ```
        """
        return len(self.vocab)
        
    def fit(self, sentences: list[str]):
        """ Fit the tokenizer using all sentences.
        1. Tokenize the sentence by splitting with spaces.
        2. Record the occurrence of all tokens
        3. Construct the token to index (self.vocab) map and the inversed map (self.inverse_vocab) based on the occurrence. The token with a higher occurrence has the smaller index

        Args:
            sentences: All sentences in the dataset.
        """
        n = len(sentences)
        for i, sentence in enumerate(sentences):
            if i % 100 == 0 or i == n - 1:
                print('Fitting Tokenizer:', (i + 1), '/', n)
            tokens = self.preprocessor.apply(sentence.strip()).split()
            if len(tokens) <= 1:
                continue
            for token in tokens:
                if token == '<unk>':
                    continue
                self.token_occurrence[token] = self.token_occurrence.get(token, 0) + 1
        print('\n')

        token_occurrence = sorted(self.token_occurrence.items(), key=lambda e: e[1], reverse=True)
        for token, occurrence in token_occurrence[:-5]:
            token_id = len(self.vocab)
            self.vocab[token] = token_id
            self.inverse_vocab[token_id] = token

        print('The number of distinct tokens:', len(self.vocab))
        
    def encode(self, sentences: list[str]) -> list[list[int]]:
        """ Encode the sentences into token ids
            Note: 1. if a token in a sentence does not exist in the fit encoder, we ignore it.
                  2. If the number of tokens in a sentence is less than two, we ignore this sentence.
                  3. Note that, for every sentence, we will add an sos_token, i.e., the id of <s> at the start of the sentence,
                     and add an eos_token, i.e., the id of </s> at the end of the sentence.
        Args:
            sentences: Raw sentences
        Returns:
            sent_token_ids: A list of id list
        """
        n = len(sentences)
        sent_token_ids = []
        for i, sentence in enumerate(sentences):
            if i % 100 == 0 or i == n - 1:
                print('Encoding with Tokenizer:', (i + 1), '/', n)
            token_ids = []
            tokens = self.preprocessor.apply(sentence.strip()).split()
            for token in tokens:
                if token == '<unk>':
                    continue
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
            if len(token_ids) <= 1:
                continue
            token_ids = [self.sos_token_id] + token_ids + [self.eos_token_id]
            sent_token_ids.append(token_ids)
        return sent_token_ids