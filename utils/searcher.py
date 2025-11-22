# utils/searcher.py
import faiss
import numpy as np
import os
import pickle
from .embedder import get_image_embedding, get_text_embedding
from utils.detector import detect_and_draw

class VideoSearchEngine:
    def __init__(self):
        self.index = None
        self.metadata = []
        self.index_path = "embeddings/index.faiss"
        self.meta_path = "embeddings/metadata.pkl"
        os.makedirs("embeddings", exist_ok=True)

    def build(self, frame_paths, captions, timestamps):
        print("Building CLIP embeddings + FAISS index...")
        embeddings = []
        
        for i, (frame_path, caption, ts) in enumerate(zip(frame_paths, captions, timestamps)):
            emb = get_image_embedding(frame_path)
            embeddings.append(emb)
            
            # Save detected version for display
            detected_path = f"frames_detected/{os.path.basename(frame_path)}"
            detect_and_draw(frame_path, detected_path)
            
            self.metadata.append({
                "frame": frame_path,
                "detected": detected_path,
                "caption": caption,
                "timestamp": ts
            })
            
            if i % 5 == 0:
                print(f"  Embedded {i+1}/{len(frame_paths)} frames")

        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        
        self.index = faiss.IndexFlatIP(768)
        self.index.add(embeddings)
        
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print("Search engine ready!")

    def search(self, query: str, top_k: int = 9):
        if self.index is None:
            print("Loading index...")
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, 'rb') as f:
                self.metadata = pickle.load(f)
        
        query_emb = get_text_embedding(query)
        query_emb = query_emb.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_emb)
        
        scores, indices = self.index.search(query_emb, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            meta = self.metadata[idx]
            results.append({
                "score": round(float(score), 3),
                "timestamp": meta["timestamp"],
                "caption": meta["caption"],
                "image_path": meta["detected"]
            })
        return results