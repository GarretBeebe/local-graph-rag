"""Unit tests for graph/store.py and graph/extractor.py — no Ollama or Qdrant required."""

from pathlib import Path

import pytest

from local_graph_rag.graph.extractor import (
    ExtractionResult,
    _parse_extraction_response,
    extract_entities_for_file,
)
from local_graph_rag.graph.store import GraphStore, slugify

# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------


def test_slugify_lowercases():
    assert slugify("MyEntity") == "myentity"


def test_slugify_replaces_spaces():
    assert slugify("fingerprint store") == "fingerprint_store"


def test_slugify_replaces_special_chars():
    assert slugify("rag.embed") == "rag_embed"


def test_slugify_strips_leading_trailing_underscores():
    assert slugify("  _hello_  ") == "hello"


def test_slugify_collapses_repeated_separators():
    assert slugify("foo--bar  baz") == "foo_bar_baz"


# ---------------------------------------------------------------------------
# GraphStore — entities
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> GraphStore:
    return GraphStore(db_path=tmp_path / "test.db")


def test_upsert_entity_creates(store: GraphStore):
    eid = store.upsert_entity("Fingerprint Store", type="CLASS", description="Tracks file hashes")
    assert eid == "fingerprint_store"


def test_upsert_entity_returns_slug(store: GraphStore):
    eid = store.upsert_entity("My Module")
    assert eid == "my_module"


def test_upsert_entity_merges_longer_description(store: GraphStore):
    store.upsert_entity("Watcher", description="short")
    store.upsert_entity("Watcher", description="a much longer and more informative description")
    # Re-fetch via neighborhood to check stored value
    store.upsert_entity("Watcher", type="CLASS")  # trigger another upsert, shouldn't regress
    eid = store.upsert_entity("Watcher")
    assert eid == "watcher"


def test_upsert_entity_keeps_shorter_description(store: GraphStore):
    store.upsert_entity("Watcher", description="detailed first description here")
    store.upsert_entity("Watcher", description="short")
    # If second description is shorter, original should be kept — verified via neighborhood
    neighborhood = store.get_entity_neighborhood("watcher", hops=0)
    desc = neighborhood["entities"][0]["description"]
    assert desc == "detailed first description here"


def test_upsert_entity_sets_type_when_stored_null(store: GraphStore):
    store.upsert_entity("Parser")  # no type
    store.upsert_entity("Parser", type="CLASS")
    neighborhood = store.get_entity_neighborhood("parser", hops=0)
    assert neighborhood["entities"][0]["type"] == "CLASS"


def test_upsert_entity_keeps_existing_type(store: GraphStore):
    store.upsert_entity("Parser", type="MODULE")
    store.upsert_entity("Parser", type="CLASS")  # should not overwrite
    neighborhood = store.get_entity_neighborhood("parser", hops=0)
    assert neighborhood["entities"][0]["type"] == "MODULE"


# ---------------------------------------------------------------------------
# GraphStore — relationships
# ---------------------------------------------------------------------------


def test_upsert_relationship_creates(store: GraphStore):
    store.upsert_entity("embed")
    store.upsert_entity("ollama_client")
    store.upsert_relationship("embed", "ollama_client", "uses", "api/embed.py")
    neighborhood = store.get_entity_neighborhood("embed", hops=1)
    labels = [r["label"] for r in neighborhood["relationships"]]
    assert "uses" in labels


def test_upsert_relationship_dedup_increments_weight(store: GraphStore):
    store.upsert_entity("a")
    store.upsert_entity("b")
    store.upsert_relationship("a", "b", "calls", "file.py")
    store.upsert_relationship("a", "b", "calls", "file.py")
    neighborhood = store.get_entity_neighborhood("a", hops=1)
    weight = neighborhood["relationships"][0]["weight"]
    assert weight == 2.0


# ---------------------------------------------------------------------------
# GraphStore — chunks
# ---------------------------------------------------------------------------


def test_register_and_get_chunks(store: GraphStore):
    store.register_chunks([
        ("uuid-1", "/docs/foo.py", 0),
        ("uuid-2", "/docs/foo.py", 1),
        ("uuid-3", "/docs/bar.py", 0),
    ])
    chunks = store.get_chunks_for_file("/docs/foo.py")
    assert chunks == ["uuid-1", "uuid-2"]


# ---------------------------------------------------------------------------
# GraphStore — delete_file_data
# ---------------------------------------------------------------------------


def test_delete_file_data_removes_orphan(store: GraphStore):
    store.upsert_entity("orphan")
    store.upsert_entity("shared")
    store.register_chunks([("c1", "file_a.py", 0)])
    store.upsert_relationship("orphan", "shared", "uses", "file_a.py")

    prior_ids = store.delete_file_data("file_a.py")

    assert prior_ids == ["c1"]
    neighborhood = store.get_entity_neighborhood("orphan", hops=0)
    assert neighborhood["entities"] == []


def test_delete_file_data_keeps_shared_entity(store: GraphStore):
    store.upsert_entity("shared")
    store.upsert_entity("other")
    store.register_chunks([("c1", "file_a.py", 0), ("c2", "file_b.py", 0)])
    store.upsert_relationship("shared", "other", "uses", "file_a.py")
    store.upsert_relationship("shared", "other", "uses", "file_b.py")

    store.delete_file_data("file_a.py")

    # shared entity still referenced by file_b.py relationship
    neighborhood = store.get_entity_neighborhood("shared", hops=1)
    ids = [e["id"] for e in neighborhood["entities"]]
    assert "shared" in ids


def test_delete_file_data_keeps_entity_referenced_only_via_chunk_link(store: GraphStore):
    """An entity with a relationship only in file_a, but chunk-linked from file_b too,
    must survive deleting file_a — without raising IntegrityError.

    Regression for the orphan-cleanup bug: chunk_entities.entity_id has no ON DELETE
    clause and PRAGMA foreign_keys=ON is set, so deleting a still-chunk-linked entity
    used to raise sqlite3.IntegrityError and roll back the whole delete_file_data
    transaction rather than just mishandling the orphan check.
    """
    slug = store.upsert_entity("shared")
    store.upsert_entity("other")
    store.register_chunks([("c1", "file_a.py", 0), ("c2", "file_b.py", 0)])
    store.upsert_relationship("shared", "other", "uses", "file_a.py")
    store.link_chunks([("c2", slug)])

    store.delete_file_data("file_a.py")  # must not raise IntegrityError

    neighborhood = store.get_entity_neighborhood(slug, hops=0)
    ids = [e["id"] for e in neighborhood["entities"]]
    assert "shared" in ids


def test_delete_file_data_returns_prior_chunk_ids(store: GraphStore):
    store.register_chunks([
        ("uuid-A", "target.py", 0),
        ("uuid-B", "target.py", 1),
        ("uuid-C", "other.py", 0),
    ])

    prior = store.delete_file_data("target.py")
    assert set(prior) == {"uuid-A", "uuid-B"}


# ---------------------------------------------------------------------------
# GraphStore — extraction cache
# ---------------------------------------------------------------------------


def test_extraction_cache_round_trip(store: GraphStore):
    store.cache_extraction("foo.py", 0, '{"entities": [], "relationships": []}')
    store.cache_extraction("foo.py", 1, '{"entities": [{"name": "A"}], "relationships": []}')
    cached = store.get_cached_extractions("foo.py")
    assert 0 in cached
    assert 1 in cached


def test_extraction_cache_clear(store: GraphStore):
    store.cache_extraction("foo.py", 0, "{}")
    store.clear_extraction_cache("foo.py")
    assert store.get_cached_extractions("foo.py") == {}


# ---------------------------------------------------------------------------
# extract_entities_for_file
# ---------------------------------------------------------------------------


def test_extract_entities_for_file_requests_json_format(store: GraphStore, monkeypatch):
    captured: dict = {}

    def _fake_generate(*args, **kwargs):
        captured.update(kwargs)
        return '{"entities": [], "relationships": []}'

    monkeypatch.setattr("local_graph_rag.rag.ollama_client.generate", _fake_generate)

    extract_entities_for_file(["some chunk text"], "foo.py", store)

    assert captured.get("format") == "json"


# ---------------------------------------------------------------------------
# GraphStore — detect_communities
# ---------------------------------------------------------------------------


def test_detect_communities_resets_stale_assignment_on_empty_graph(store: GraphStore):
    """An entity with no relationships must lose a stale community on an empty graph.

    build_networkx_graph only adds nodes via add_edge, so a graph with zero
    relationships has zero nodes — detect_communities takes its early-exit branch
    and never computes a partition. The reset must still run in that branch, or a
    prior non-NULL community value is retained forever.
    """
    slug = store.upsert_entity("isolated")
    store._conn.execute("UPDATE entities SET community = ? WHERE id = ?", (7, slug))
    store._conn.commit()

    store.detect_communities()

    neighborhood = store.get_entity_neighborhood(slug, hops=0)
    assert neighborhood["entities"][0]["community"] is None


def test_detect_communities_resets_isolated_entity_in_nonempty_graph(store: GraphStore):
    """An isolated entity must lose its stale community even when OTHER entities
    form a non-empty graph and get freshly assigned.

    Distinct code path from the empty-graph case: best_partition() runs and returns
    a non-empty partition for the connected pair, but the isolated entity never
    becomes a node (build_networkx_graph only adds nodes via add_edge) — so it's
    absent from the partition and must be cleared by the reset, not left stale.
    """
    a = store.upsert_entity("connected_a")
    store.upsert_entity("connected_b")
    isolated = store.upsert_entity("isolated")
    store.upsert_relationship("connected_a", "connected_b", "uses", "doc.py")
    store._conn.execute("UPDATE entities SET community = ? WHERE id = ?", (7, isolated))
    store._conn.commit()

    store.detect_communities()

    isolated_neighborhood = store.get_entity_neighborhood(isolated, hops=0)
    assert isolated_neighborhood["entities"][0]["community"] is None

    connected_neighborhood = store.get_entity_neighborhood(a, hops=0)
    assert connected_neighborhood["entities"][0]["community"] is not None


# ---------------------------------------------------------------------------
# _parse_extraction_response
# ---------------------------------------------------------------------------


def test_parse_valid_json():
    response = (
        '{"entities": [{"name": "Watcher", "type": "CLASS", "description": "watches files"}],'
        ' "relationships": []}'
    )
    result = _parse_extraction_response(response)
    assert len(result.entities) == 1
    assert result.entities[0]["name"] == "Watcher"
    assert result.relationships == []


def test_parse_embedded_json():
    response = """
    Here is the extracted information:
    {"entities": [{"name": "Embed", "type": "MODULE", "description": "embedding helper"}],
     "relationships": [{"source": "Embed", "target": "Ollama", "label": "calls"}]}
    Hope that helps!
    """
    result = _parse_extraction_response(response)
    assert len(result.entities) == 1
    assert len(result.relationships) == 1


def test_parse_malformed_returns_empty():
    result = _parse_extraction_response("this is not json at all!!!")
    assert isinstance(result, ExtractionResult)
    assert result.entities == []
    assert result.relationships == []


def test_parse_empty_arrays():
    result = _parse_extraction_response('{"entities": [], "relationships": []}')
    assert result.entities == []
    assert result.relationships == []


def test_parse_double_json_prefix_recovers_entity():
    response = (
        '{}\n{"entities": [{"name": "X", "type": "MODULE", "description": "x"}],'
        ' "relationships": []}'
    )
    result = _parse_extraction_response(response)
    assert len(result.entities) == 1
    assert result.entities[0]["name"] == "X"


def test_parse_python_none_recovers_via_null_substitution():
    response = (
        '{"entities": [{"name": "A", "type": "CLASS", "description": "a"},'
        ' {"name": "B", "type": "CLASS", "description": "b"}],'
        ' "relationships": [{"source": "A", "target": "B", "label": "uses", "extra": None}]}'
    )
    result = _parse_extraction_response(response)
    assert len(result.entities) == 2
    assert len(result.relationships) == 1
    assert result.relationships[0]["target"] == "B"


def test_parse_valid_json_with_none_word_in_string_untouched():
    response = (
        '{"entities": [{"name": "Finder", "type": "FUNCTION",'
        ' "description": "Returns None if not found"}], "relationships": []}'
    )
    result = _parse_extraction_response(response)
    assert result.entities[0]["description"] == "Returns None if not found"
