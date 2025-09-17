"""
Data ingestion service for Medical Quiz Assistant.
Implements idempotent data loading per ADR-001 using ChromaDB and LangChain.
"""

import json
import logging
import os
import time
from collections.abc import Iterable
from typing import Any, Optional

import chromadb
from datasets import load_dataset
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import Settings
from app.repositories.question_repository import QuestionRepository
from app.repositories.vector_repository import VectorRepository

logger = logging.getLogger(__name__)

# ----------------------------
# Config (env-driven)
# ----------------------------
DATASET = os.getenv("MEDMCQA_DATASET", "openlifescienceai/medmcqa")
EMBED_MODEL = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLL_Q = os.getenv("CHROMA_COLLECTION_QUESTIONS", "questions")
COLL_E = os.getenv("CHROMA_COLLECTION_EXPLANATIONS", "explanations")


def batched(iterable: Iterable, n: int) -> Iterable[list]:
    batch: list = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def to_question_record(row) -> tuple[str, str, dict]:
    """Build a (id, text, metadata) triple for the QUESTIONS collection."""
    qid = str(row.get("id") or row.get("qid") or row.get("question_id"))
    options = [row["opa"], row["opb"], row["opc"], row["opd"]]
    text = f"{row['question']}\nOptions: " + " | ".join(options)
    meta = {
        "id": qid,
        "split": row.get("split", "train"),
        "correct_idx": int(row["cop"]),
        "subject": (row.get("topic_name") or row.get("topic") or "Unknown"),
        "topic": (row.get("topic_name") or row.get("topic") or "Unknown"),
        "difficulty": (row.get("level") or "Unknown"),
    }
    return qid, text, meta


def to_expl_record(row) -> tuple[str, str, dict] | None:
    """Build a (id, text, metadata) triple for the EXPLANATIONS collection, if explanation exists."""
    exp = row.get("exp")
    if not exp:
        return None
    qid = str(row.get("id") or row.get("qid") or row.get("question_id"))
    meta = {
        "question_id": qid,
        "type": "correct",
        "subject": (row.get("topic_name") or row.get("topic") or "Unknown"),
        "topic": (row.get("topic_name") or row.get("topic") or "Unknown"),
        "difficulty": (row.get("level") or "Unknown"),
        "split": row.get("split", "train"),
    }
    return f"{qid}_correct", exp, meta


class IngestService:
    """Service for ingesting MedMCQA data into the system using ChromaDB and LangChain."""

    def __init__(
        self, question_repo: QuestionRepository, vector_repo: VectorRepository, settings: Settings
    ):
        self.question_repo = question_repo
        self.vector_repo = vector_repo
        self.settings = settings
        self._ingestion_jobs: dict[str, dict[str, Any]] = {}

        # ChromaDB configuration
        self.persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.client = None
        self.embedder = None
        self.vs_q = None
        self.vs_e = None

    async def ingest_data(
        self, split: str, limit: Optional[int] = None, batch_size: int = 1000
    ) -> dict[str, Any]:
        """
        Ingest data from MedMCQA dataset using ChromaDB and LangChain.

        Args:
            split: Dataset split to ingest
            limit: Maximum number of questions to process
            batch_size: Batch size for processing

        Returns:
            Dictionary with ingestion results
        """
        t0 = time.time()
        logger.info(f"Starting data ingestion for split {split}")

        # 1) Load dataset streamingly
        logger.info(f"Loading dataset: {DATASET} split={split}")
        ds = load_dataset(DATASET, split=split)
        if limit and limit > 0:
            ds = ds.select(range(min(limit, len(ds))))
        fingerprint = getattr(ds, "_fingerprint", None)

        # 2) Set up embeddings + Chroma (LangChain wrapper)
        self.embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

        # Use a single persistent client for idempotent upserts & counts
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # Ensure collections exist (public chromadb)
        col_q = self.client.get_or_create_collection(COLL_Q)
        col_e = self.client.get_or_create_collection(COLL_E)

        # Also create LC wrappers backed by that client (for parity with runtime)
        self.vs_q = Chroma(
            client=self.client,
            collection_name=COLL_Q,
            embedding_function=self.embedder,
            persist_directory=self.persist_dir,
        )
        self.vs_e = Chroma(
            client=self.client,
            collection_name=COLL_E,
            embedding_function=self.embedder,
            persist_directory=self.persist_dir,
        )

        total_q = total_e = 0
        # Collect simple catalogs to help the API expose available filters
        topics_set: set[str] = set()
        subjects_set: set[str] = set()
        difficulties_set: set[str] = set()
        by_topic: dict[str, int] = {}
        by_subject: dict[str, int] = {}
        by_difficulty: dict[str, int] = {}

        # 3) Stream in batches and **upsert** (idempotent) using chromadb collection
        for batch_rows in batched(ds, batch_size):
            ids_q, texts_q, metas_q = [], [], []
            ids_e, texts_e, metas_e = [], [], []

            for row in batch_rows:
                qid, qtext, qmeta = to_question_record(row)
                ids_q.append(qid)
                texts_q.append(qtext)
                metas_q.append(qmeta)

                # update catalogs
                t = (qmeta.get("topic") or "").strip()
                s = (qmeta.get("subject") or "").strip()
                d = (qmeta.get("difficulty") or "Unknown").strip()
                if t:
                    topics_set.add(t)
                    by_topic[t] = by_topic.get(t, 0) + 1
                if s:
                    subjects_set.add(s)
                    by_subject[s] = by_subject.get(s, 0) + 1
                if d:
                    difficulties_set.add(d)
                    by_difficulty[d] = by_difficulty.get(d, 0) + 1

                expl = to_expl_record(row)
                if expl:
                    eid, etext, emeta = expl
                    ids_e.append(eid)
                    texts_e.append(etext)
                    metas_e.append(emeta)

            if ids_q:
                # IMPORTANT: upsert for idempotency (LangChain add_texts = add only)
                # Upsert does not embed; we need embeddings vectors:
                # Easiest: use vs_q to embed, then pass vectors into chroma upsert
                q_embeddings = self.vs_q._embedding_function.embed_documents(
                    texts_q
                )  # public enough in LC
                col_q.upsert(
                    ids=ids_q, embeddings=q_embeddings, metadatas=metas_q, documents=texts_q
                )
                total_q += len(ids_q)

            if ids_e:
                e_embeddings = self.vs_e._embedding_function.embed_documents(texts_e)
                col_e.upsert(
                    ids=ids_e, embeddings=e_embeddings, metadatas=metas_e, documents=texts_e
                )
                total_e += len(ids_e)

            logger.info(
                f"Upserted batch: Q+{len(ids_q)} / E+{len(ids_e)}; totals Q={total_q} E={total_e}"
            )

        # 4) Persistence is automatic in Chroma 0.4.x when a persist_directory is set

        # 5) Simple validation via public chromadb
        count_q = col_q.count()
        count_e = col_e.count()
        logger.info(
            f"Final counts: questions={count_q}, explanations={count_e}, fingerprint={fingerprint}"
        )

        # 6) Persist a lightweight catalog for API discovery (topics/subjects/difficulties)
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            catalog_path = os.path.join(self.persist_dir, "topics.json")
            catalog = {
                "topics": sorted(list(topics_set)),
                "subjects": sorted(list(subjects_set)),
                "difficulties": sorted(list(difficulties_set)),
                "counts": {
                    "by_topic": by_topic,
                    "by_subject": by_subject,
                    "by_difficulty": by_difficulty,
                },
                "version": fingerprint or "unknown",
            }
            with open(catalog_path, "w", encoding="utf-8") as f:
                json.dump(catalog, f, ensure_ascii=False, indent=2)
            logger.info(f"Wrote topic catalog to {catalog_path}")
        except Exception as e:
            logger.warning(f"Failed to write topics catalog: {e}")

        # Optional: quick retrieval smoke test using LC retriever (MMR)
        retriever = self.vs_q.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 12})
        docs = retriever.get_relevant_documents("heart failure management")
        first_id = docs[0].metadata.get("id") if docs else None
        logger.info(f"Retriever smoke: got {len(docs)} docs; first id={first_id}")

        processing_time = int((time.time() - t0) * 1000)

        result = {
            "processed": total_q,
            "skipped": 0,  # Upsert handles idempotency
            "version": fingerprint or f"split_{split}_{int(time.time())}",
            "processing_time_ms": processing_time,
            "explanations_processed": total_e,
        }

        logger.info(
            f"Data ingestion completed in {processing_time}ms: {total_q} questions, {total_e} explanations"
        )
        return result

    async def get_ingestion_stats(self) -> dict[str, Any]:
        """Get overall ingestion statistics from ChromaDB."""
        if not self.client:
            # Initialize client if not already done
            self.client = chromadb.PersistentClient(path=self.persist_dir)

        try:
            col_q = self.client.get_collection(COLL_Q)
            col_e = self.client.get_collection(COLL_E)

            count_q = col_q.count()
            count_e = col_e.count()

            return {
                "total_questions": count_q,
                "total_explanations": count_e,
                "collections": {"questions": count_q, "explanations": count_e},
            }
        except Exception as e:
            logger.error(f"Error getting ingestion stats: {e}")
            return {
                "total_questions": 0,
                "total_explanations": 0,
                "collections": {"questions": 0, "explanations": 0},
            }
