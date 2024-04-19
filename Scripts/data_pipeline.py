import os
import torch
from tokenizer import Tokenizer
from vectorizedata import GloveEmbedding

class FileLoader:
    """
    A class to load and preprocess text files from a specified directory.
    
    Attributes:
        data_path (str): The path to the directory containing text files.
    
    Methods:
        load_files(): Generator function to read files sequentially.
    """
    def __init__(self, data_path):
        """
        Initializes the FileLoader with the specified data path.
        
        Args:
            data_path (str): The path to the directory containing text files.
        """
        self.data_path = data_path

    def load_files(self):
        """
        Yields the content of each text file in the directory, replacing new lines with paragraph markers.
        
        Yields:
            str: The content of a file with new lines replaced by ' <PAR> '.
        """
        for filename in os.listdir(self.data_path):
            file_path = os.path.join(self.data_path, filename)
            if file_path.endswith('.txt'):
                yield self._load_file(file_path)

    def _load_file(self, file_path):
        """
        Reads and preprocesses the content of a file.
        
        Args:
            file_path (str): Path to the file to be read.
        
        Returns:
            str: Preprocessed text of the file.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().replace('\n', ' <PAR> ')
        
class TextTokenizer:
    """
    A class to tokenize text using a predefined tokenizer.
    
    Attributes:
        tokenizer (Tokenizer): An instance of Tokenizer for encoding text.
    """
    def __init__(self):
        """
        Initializes TextTokenizer and sets up the tokenizer instance.
        """
        self.tokenizer = Tokenizer()

    def tokenize(self, text):
        """
        Tokenizes and encodes the given text.
        
        Args:
            text (str): The text to be tokenized.
        
        Returns:
            list: A list of tokens or token IDs.
        """
        self.tokenizer.fit(text)  
        return self.tokenizer.encode(text)
    
class DatasetBuilder:
    """
    A class to build a dataset from data tensors.
    
    Attributes:
        dataset (list): A list of tensors representing the dataset.
    """
    def __init__(self):
        """
        Initializes the DatasetBuilder with an empty list.
        """
        self.dataset = []

    def add_data(self, data_tensor):
        """
        Adds a tensor to the dataset.
        
        Args:
            data_tensor (Tensor): A PyTorch tensor to be added to the dataset.
        """
        self.dataset.append(data_tensor)

    def get_dataset(self):
        """
        Retrieves the entire dataset.
        
        Returns:
            list: A list of tensors representing the dataset.
        """
        return self.dataset
        
class TextProcessor:
    """
    A class to process text files into a dataset using a pipeline of loading, tokenizing, and embedding.
    
    Attributes:
        loader (FileLoader): An instance of FileLoader to handle file loading.
        tokenizer (Tokenizer): An instance of Tokenizer for tokenizing text.
        embedder (GloveEmbedding): An instance of GloveEmbedding to handle word embeddings.
        dataset_builder (DatasetBuilder): An instance of DatasetBuilder to build the dataset.
    """
    def __init__(self, data_path):
        """
        Initializes the TextProcessor with specified components.
        
        Args:
            data_path (str): The directory path containing text files to process.
        """
        self.loader = FileLoader(data_path)
        self.tokenizer = TextTokenizer()
        self.embedder = GloveEmbedding()
        self.dataset_builder = DatasetBuilder()

    def process(self):
        """
        Processes text files through the pipeline to create a dataset.
        
        Returns:
            list: A dataset compiled from processed text files.
        """
        for data in self.loader.load_files():
            tokens = self.tokenizer.tokenize(data)
            embeddings = self.embedder.fit_line(tokens)
            data_tensor = torch.tensor(embeddings)
            self.dataset_builder.add_data(data_tensor)
        
        return self.dataset_builder.get_dataset()

if __name__ == '__main__':
    data_path = '/Users/priyanshusingh/Desktop/AAI-627/AAI-627Project/ResearchPaperSummarizer/Data/sumpubmed/line_text'
    processor = TextProcessor(data_path)
    dataset = processor.process()
    print(f'Processed dataset with {len(dataset)} entries.')
