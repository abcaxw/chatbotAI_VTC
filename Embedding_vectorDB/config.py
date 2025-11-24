import os


class Config:
    # ===== Milvus =====
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "103.252.0.129")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "document_embeddings")
    FAQ_COLLECTION: str = os.getenv("FAQ_COLLECTION", "faq_embeddings")

    # ===== Embedding =====
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "keepitreal/vietnamese-sbert")
    # THAY ĐỔI: 1024 -> 768 dimensions
    EMBEDDING_DIMENSION: int = int(
        os.getenv("EMBEDDING_DIM", os.getenv("EMBEDDING_DIMENSION", "768"))
    )

    # ===== Khác =====
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "100")) * 1024 * 1024  # 100MB
    TESSDATA_PREFIX: str = os.getenv(
        "TESSDATA_PREFIX", "/usr/share/tesseract-ocr/4.00/tessdata"
    )


config = Config()