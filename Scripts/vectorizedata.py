import os
import zipfile
import numpy as np
import chakin
import torch

class GloveVector:
    """Class to manage download and extraction of pretrained GloVe word embeddings."""
    def __init__(self):
        self._chakin_index = 16  # Index for GloVe.840B.300d in chakin
        self._num_dimensions = 300
        self._subfolder_name = 'glove.840B.300d'
        self._data_folder = 'embeddings'
        self._zip_path = os.path.join(self._data_folder, f'{self._subfolder_name}.zip')
        self._unzip_folder = os.path.join(self._data_folder, self._subfolder_name)
        self.glove_file_name = os.path.join(self._unzip_folder, f'{self._subfolder_name}.txt')

    def download_embeddings(self):
        """Download the embeddings using chakin if they are not already downloaded."""
        if not os.path.exists(self._zip_path):
            print(f"Downloading embeddings to '{self._zip_path}'")
            chakin.download(number=self._chakin_index, save_dir=self._data_folder)
            print("Download complete.")

    def extract_embeddings(self):
        """Extracts the embeddings if not already extracted."""
        if not os.path.exists(self._unzip_folder):
            os.makedirs(self._unzip_folder, exist_ok=True)
            with zipfile.ZipFile(self._zip_path, 'r') as zip_ref:
                zip_ref.extractall(self._unzip_folder)
            print(f"Extracted embeddings to '{self._unzip_folder}'")

    def apply(self):
        """Ensure embeddings are downloaded and extracted."""
        self.download_embeddings()
        self.extract_embeddings()

    def get_glove_file(self):
        return self.glove_file_name

class GloveEmbedding:
    """Class to handle GloVe word embeddings loading and querying."""
    def __init__(self):
        glove_vector = GloveVector()
        glove_vector.apply()
        self.glove_file = glove_vector.get_glove_file()
        self.embedding_dict = {}
        self.load_glove_model()

    def load_glove_model(self):
        """Loads the GloVe model from a file while handling exceptions in vector parsing."""
        with open(self.glove_file, 'r') as file:
            for line in file:
                data = line.split()
                if len(data) <= 2:  # Checks if there is at least one word and one number
                    continue
                word = data[0]
                try:
                    vector = np.asarray(data[1:], dtype="float32")
                except ValueError:
                    continue  # Skip lines where conversion fails
                self.embedding_dict[word] = vector


    def get_word_embedding(self, word, dim=300):
        """Retrieve a word embedding, or return a zero vector if the word is not found."""
        return self.embedding_dict.get(word, np.zeros(dim))

    def fit_line(self, sentence):
        """Converts a list of words in a sentence to a tensor of embeddings."""
        embeddings = [self.get_word_embedding(word) for word in sentence]
        return torch.tensor(embeddings)
