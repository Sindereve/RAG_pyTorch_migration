from dataclasses import dataclass


@dataclass
class EmbeddingConfig():
    model_key: str = "mini"
    chunk_size: int = 512
    chunk_overlap: int = 50

@dataclass
class QDrantClientConfig():
    url: str ="https://711e45be-209f-4c40-bcf7-a28bda6fedb3.europe-west3-0.gcp.cloud.qdrant.io:6333"
    api_key: str ="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.9p-f5cVhz-wf49nwtKVuv6Mj7ZO-UvJ19dQE5CkrC4o"

cfg_emb = EmbeddingConfig()
cfg_db = QDrantClientConfig()