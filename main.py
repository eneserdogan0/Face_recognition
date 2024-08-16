import os
from face_embedding import FaceEmbedding
from embedding_database import EmbeddingDatabase
from face_recognizer import FaceRecognizer
from utils import print_report, save_report_to_file, get_person_name

def process_image(image_path, face_embedding_model, recognizer, similarity_threshold, k, method, report_file=None):
    embedding = face_embedding_model.extract_face_embedding(image_path)
    if embedding is None:
        print(f"\n{image_path} için yüz bulunamadı.\n")
        if report_file:
            save_report_to_file(report_file, f"{image_path} için yüz bulunamadı.", [], [])
        return

    best_people, best_similarities, above_threshold = recognizer.find_similar_faces(
        embedding, k=k, threshold=similarity_threshold, method=method)

    if best_people.size == 0:
        print(f"\n{image_path} için benzer 5 kişi bulunamadı.\n")
        if report_file:
            save_report_to_file(report_file, f"{image_path} için benzer 5 kişi bulunamadı.", [], [])
        return

    best_names = [get_person_name(person) for person in best_people]
    best_data = list(zip(best_names, best_similarities))
    print_report(f"{image_path} - En Yakin 5 Kisi ({method} similarity)", ["Isim", "Benzerlik Orani"], best_data)
    if report_file:
        save_report_to_file(report_file, f"{image_path} - En Yakin 5 Kisi ({method} similarity)", ["Isim", "Benzerlik Orani"], best_data)

    if above_threshold:
        threshold_names = [get_person_name(person[0]) for person in above_threshold]
        threshold_similarities = [person[1] for person in above_threshold]
        threshold_data = list(zip(threshold_names, threshold_similarities))
        print_report(f"{image_path} - Threshold ({similarity_threshold*100}%) Ustu Kisiler ({method} similarity)", ["Isim", "Benzerlik Orani"], threshold_data)
        if report_file:
            save_report_to_file(report_file, f"{image_path} - Threshold ({similarity_threshold*100}%) Ustu Kisiler ({method} similarity)", ["Isim", "Benzerlik Orani"], threshold_data)
    else:
        print(f"\n{image_path} için threshold degeri ustu kisi yoktur {similarity_threshold * 100}%.\n")
        if report_file:
            save_report_to_file(report_file, f"{image_path} için threshold degeri ustu kisi yoktur {similarity_threshold * 100}%.", [], [])

def main(image_dir, db_path, similarity_threshold=0.75, k=5, method="cosine", report_file="report.txt"):
    face_embedding_model = FaceEmbedding()
    db = EmbeddingDatabase(db_path)
    recognizer = FaceRecognizer(db.embeddings)

    if os.path.exists(report_file):
        os.remove(report_file)

    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                process_image(image_path, face_embedding_model, recognizer, similarity_threshold, k, method, report_file)

if __name__ == "__main__":
    image_dir = "soccer_test_db"
    db_path = "vektorler2.json"
    main(image_dir, db_path, method="cosine")
