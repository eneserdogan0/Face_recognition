import numpy as np

class SimilarityCalculator:
    @staticmethod
    def cosine_similarity(vector1, vector2):
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    @staticmethod
    def euclidean_similarity(vector1, vector2):
        return np.linalg.norm(vector1 - vector2)
