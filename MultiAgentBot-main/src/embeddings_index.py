from sentence_transformers import SentenceTransformer
import chromadb, os, json
from chromadb.config import Settings
from PyPDF2 import PdfReader

MODEL = r"C:\TextSummarization\TS\all-MiniLM-L6-v2"
DB_DIR = "storage/chroma"

def load_text(path):
    if path.lower().endswith(".pdf"):
        text = []
        for p in PdfReader(path).pages:
            text.append(p.extract_text() or "")
        return "\n".join(text)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def main():
    embed = SentenceTransformer(MODEL)
    client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(allow_reset=False))
    col   = client.get_or_create_collection("brochures")

    with open("data/products.json","r",encoding="utf-8") as f:
        products = json.load(f)["products"]

    docs, ids, metas = [], [], []
    for p in products:
        t = load_text(p["brochure_path"])
        docs.append(t[:50000])   # keep it simple
        ids.append(p["id"])
        metas.append({"product_id": p["id"], "name": p["name"], "company": p["company"]})

    # compute embeddings and upsert
    vecs = embed.encode(docs, normalize_embeddings=True).tolist()
    if ids:
        col.upsert(ids=ids, documents=docs, embeddings=vecs, metadatas=metas)

if __name__ == "__main__":
    main()
