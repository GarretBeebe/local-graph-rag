"""Global retrieval: numpy cosine similarity over community summary embeddings."""

import logging
from dataclasses import dataclass, field

import numpy as np

from api.embed import embed
from graph.store import GraphStore
from settings import COMMUNITY_RETRIEVAL_N

logger = logging.getLogger(__name__)


@dataclass
class GlobalContext:
    community_summaries: list[str] = field(default_factory=list)
    community_ids: list[int] = field(default_factory=list)


def global_retrieve(
    question: str,
    store: GraphStore,
    *,
    n: int = COMMUNITY_RETRIEVAL_N,
    communities: list[dict] | None = None,
) -> GlobalContext:
    """Cosine similarity over community embeddings to find the top-N relevant summaries.

    Pass pre-fetched communities to avoid a redundant DB round-trip when the caller
    already holds them (e.g. for the communities_available check in the router).
    """
    if communities is None:
        communities = store.get_communities()

    relevant = [c for c in communities if c["embedding"] is not None and c["summary"]]
    if not relevant:
        logger.warning("global_retrieve: no community summaries available")
        return GlobalContext()

    q_vec = np.array(embed(question), dtype=np.float32)

    # A blob's byte length must match the query vector's to form a matrix and matmul
    # cleanly. A mismatch means the embedding model/VECTOR_SIZE changed since this
    # community was summarized — skip it rather than letting numpy raise deep inside
    # the matrix construction or the dot product below.
    usable = [c for c in relevant if len(c["embedding"]) == q_vec.nbytes]
    if len(usable) != len(relevant):
        logger.warning(
            "global_retrieve: skipping %d/%d communities with mismatched embedding size "
            "(expected %d bytes for a %d-dim vector — embeddings may predate a "
            "model/VECTOR_SIZE change)",
            len(relevant) - len(usable), len(relevant), q_vec.nbytes, q_vec.size,
        )
    if not usable:
        return GlobalContext()

    matrix = np.array([np.frombuffer(c["embedding"], dtype=np.float32) for c in usable])

    matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix_norms[matrix_norms == 0] = 1.0
    matrix_unit = matrix / matrix_norms

    q_norm = float(np.linalg.norm(q_vec))
    q_unit = q_vec / q_norm if q_norm > 0 else q_vec

    scores = matrix_unit @ q_unit
    top_indices = np.argsort(scores)[::-1][:n]

    top = [usable[i] for i in top_indices]
    logger.debug("global_retrieve: %d communities → top %d selected", len(usable), len(top))
    return GlobalContext(
        community_summaries=[c["summary"] for c in top],
        community_ids=[c["id"] for c in top],
    )
