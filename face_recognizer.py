import numpy as np
from similarity_calculator import SimilarityCalculator

class FaceRecognizer:
    def __init__(self, db_embeddings):
        self.db_embeddings = db_embeddings

    def find_similar_faces(self, face_embedding, k=5, threshold=0.75, method="cosine"):
        similarities, people, above_threshold = [], [], []

        for person, emb in self.db_embeddings.items():
            if method == "cosine":
                similarity = SimilarityCalculator.cosine_similarity(face_embedding, emb)
                is_above_threshold = similarity >= threshold
            elif method == "euclidean":
                similarity = SimilarityCalculator.euclidean_similarity(face_embedding, emb)
                is_above_threshold = similarity <= threshold

            similarities.append(similarity)
            people.append(person)

            if is_above_threshold:
                above_threshold.append((person, similarity))

        if not similarities:
            return

        similarities = np.array(similarities)
        people = np.array(people)

        if method == "cosine":
            best_indices = np.argsort(-similarities)[:k]
        elif method == "euclidean":
            best_indices = np.argsort(similarities)[:k]

        best_people = people[best_indices]
        best_similarities = similarities[best_indices]

        return best_people, best_similarities, above_threshold
