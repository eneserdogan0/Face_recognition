import json
import numpy as np

class EmbeddingDatabase:
    def __init__(self, db_path):
        self.embeddings = self.load_embeddings(db_path)
        self.convert_embeddings_to_numpy()

    @staticmethod
    def load_embeddings(db_path):
        with open(db_path, "r") as f:
            return json.load(f)

    def convert_embeddings_to_numpy(self):
        for person, emb in self.embeddings.items():
            self.embeddings[person] = np.array(emb)
