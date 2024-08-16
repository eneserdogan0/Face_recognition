import cv2
from insightface.app import FaceAnalysis

class FaceEmbedding:
    def __init__(self, det_size=(640, 640), ctx_id=0):
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def extract_face_embedding(self, image_path):
        img = cv2.imread(image_path)
        faces = self.app.get(img)
        if faces:
            return faces[0].embedding
        return None
