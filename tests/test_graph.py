"""Unit tests for graph/store.py and graph/extractor.py — no Ollama or Qdrant required."""

import json
import sqlite3
import threading
from pathlib import Path

import pytest

from local_graph_rag.graph.extractor import (
    ExtractionResult,
    _parse_extraction_response,
    extract_entities_for_file,
)
from local_graph_rag.graph.record_store import _REFRESH_ENTITIES_SQL
from local_graph_rag.graph.store import _MIGRATIONS, _SCHEMA, GraphStore, slugify
from tests.helpers import (
    EMPTY_EXTRACTION_JSON,
    add_entity,
    add_relationship,
    patch_ollama_generate,
)

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


def _entity(store: GraphStore, entity_id: str) -> dict | None:
    row = store.conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
    return dict(row) if row else None


def _neighborhood(store: GraphStore, entity_id: str, hops: int = 1) -> tuple[list, list]:
    return store.expand_neighborhood([entity_id], hops, max_entities=50, max_relationships=50)


# ---------------------------------------------------------------------------
# GraphStore — connections and migrations
# ---------------------------------------------------------------------------


def test_each_thread_gets_its_own_connection(store: GraphStore):
    other: list[sqlite3.Connection] = []
    thread = threading.Thread(target=lambda: other.append(store.conn))
    thread.start()
    thread.join()

    assert other[0] is not store.conn
    assert other[0].execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_close_closes_all_connections_and_later_access_reopens(store: GraphStore):
    first = store.conn
    store.close()

    with pytest.raises(sqlite3.ProgrammingError):
        first.execute("SELECT 1")
    assert store.conn.execute("SELECT 1").fetchone()[0] == 1


def test_connections_of_exited_threads_are_closed_when_a_new_thread_connects(
    store: GraphStore,
):
    """Regression: tracking connections for close() kept every exited thread's one open."""
    exited: list[sqlite3.Connection] = []
    for _ in range(20):
        thread = threading.Thread(target=lambda: exited.append(store.conn))
        thread.start()
        thread.join()

    newcomer = threading.Thread(target=lambda: store.conn)
    newcomer.start()
    newcomer.join()

    with pytest.raises(sqlite3.ProgrammingError):
        exited[0].execute("SELECT 1")
    assert len(store._conns) == 2  # this thread's connection and the newcomer's


def test_a_failed_commit_rolls_back_and_the_store_stays_writable(store: GraphStore):
    with pytest.raises(sqlite3.IntegrityError), store._write() as conn:
        conn.execute("PRAGMA defer_foreign_keys = ON")  # make the FK check fail at COMMIT
        conn.execute("INSERT INTO relationships VALUES ('ghost', 'missing', 'uses', 1.0, 'd.py')")

    add_entity(store, "Fine")  # used to fail: "cannot start a transaction within a transaction"
    assert _entity(store, "fine") is not None


def test_new_store_runs_all_migrations(store: GraphStore):
    assert store.conn.execute("PRAGMA user_version").fetchone()[0] == len(_MIGRATIONS)
    indexes = {
        row["name"]
        for row in store.conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
    }
    assert {"idx_chunks_filepath", "idx_relationships_target_id"} <= indexes


def test_reopening_a_migrated_store_is_a_no_op(tmp_path: Path):
    GraphStore(db_path=tmp_path / "test.db").close()
    reopened = GraphStore(db_path=tmp_path / "test.db")
    assert reopened.conn.execute("PRAGMA user_version").fetchone()[0] == len(_MIGRATIONS)


def _legacy_database(db_path: Path) -> None:
    """Create a version-0 database the way pre-migration code left it."""
    legacy = sqlite3.connect(db_path)
    legacy.executescript(_SCHEMA)
    legacy.executemany(
        "INSERT INTO entities (id, name, type, description) VALUES (?, ?, ?, ?)",
        [
            ("linked", "Linked", "CLASS", "merged description"),
            ("chunked", "Chunked", None, "via chunk link"),
            ("orphan", "Orphan", "OTHER", "nothing references me"),
        ],
    )
    legacy.execute("INSERT INTO relationships VALUES ('linked', 'chunked', 'uses', 1.0, 'a.py')")
    legacy.execute("INSERT INTO chunks VALUES ('c1', 'b.py', 0)")
    legacy.execute("INSERT INTO chunk_entities VALUES ('c1', 'chunked')")
    legacy.execute("INSERT INTO extraction_cache VALUES ('a.py', 0, '{}')")
    legacy.commit()
    legacy.close()


def test_migrating_a_legacy_database_backfills_mentions_and_drops_orphans(tmp_path: Path):
    _legacy_database(tmp_path / "legacy.db")

    store = GraphStore(db_path=tmp_path / "legacy.db")

    mentions = {
        (row["entity_id"], row["source_doc"])
        for row in store.conn.execute("SELECT entity_id, source_doc FROM entity_mentions")
    }
    assert mentions == {("linked", "a.py"), ("chunked", "a.py"), ("chunked", "b.py")}
    assert _entity(store, "orphan") is None
    assert store.conn.execute("PRAGMA foreign_key_check").fetchall() == []
    cache_columns = store.conn.execute("PRAGMA table_info(extraction_cache)").fetchall()
    assert "prompt_sha256" in {row["name"] for row in cache_columns}
    assert store.conn.execute("SELECT COUNT(*) FROM extraction_cache").fetchone()[0] == 0

    before = [dict(row) for row in store.conn.execute("SELECT * FROM entities ORDER BY id")]
    with store._write() as conn:  # the first refresh after migrating must change nothing
        conn.execute(_REFRESH_ENTITIES_SQL, (json.dumps(["linked", "chunked"]),))
    after = [dict(row) for row in store.conn.execute("SELECT * FROM entities ORDER BY id")]
    assert after == before


def test_upsert_entity_creates(store: GraphStore):
    eid = add_entity(store, "Fingerprint Store", type="CLASS", description="Tracks file hashes")
    assert eid == "fingerprint_store"


def test_upsert_entity_returns_slug(store: GraphStore):
    eid = add_entity(store, "My Module")
    assert eid == "my_module"


def test_entity_description_is_the_longest_across_documents(store: GraphStore):
    add_entity(store, "Watcher", description="short", doc="a.py")
    add_entity(
        store, "Watcher", description="a much longer and more informative description", doc="b.py"
    )
    add_entity(store, "Watcher", type="CLASS", doc="c.py")  # no description — must not regress
    assert _entity(store, "watcher")["description"] == (
        "a much longer and more informative description"
    )


def test_entity_description_keeps_longer_one_from_another_document(store: GraphStore):
    add_entity(store, "Watcher", description="detailed first description here", doc="a.py")
    add_entity(store, "Watcher", description="short", doc="b.py")
    assert _entity(store, "watcher")["description"] == "detailed first description here"


def test_entity_type_set_once_any_document_provides_one(store: GraphStore):
    add_entity(store, "Parser", doc="a.py")  # no type
    add_entity(store, "Parser", type="CLASS", doc="b.py")
    assert _entity(store, "parser")["type"] == "CLASS"


def test_entity_type_is_the_most_common_across_documents(store: GraphStore):
    add_entity(store, "Parser", type="MODULE", doc="a.py")
    add_entity(store, "Parser", type="CLASS", doc="b.py")
    add_entity(store, "Parser", type="MODULE", doc="c.py")
    assert _entity(store, "parser")["type"] == "MODULE"


def test_same_document_upsert_replaces_its_previous_mention(store: GraphStore):
    add_entity(store, "Parser", type="CLASS", description="old, longer description", doc="a.py")
    add_entity(store, "Parser", type="FUNCTION", description="new", doc="a.py")
    entity = _entity(store, "parser")
    assert (entity["type"], entity["description"]) == ("FUNCTION", "new")


def test_deleting_a_document_retracts_its_description_and_type(store: GraphStore):
    add_entity(
        store, "Parser", type="CLASS", description="stale but much longer text", doc="old.py"
    )
    add_entity(store, "Parser", type="FUNCTION", description="current", doc="new.py")

    store.delete_file_data("old.py")

    entity = _entity(store, "parser")
    assert (entity["type"], entity["description"]) == ("FUNCTION", "current")


def test_upsert_entities_rejects_names_with_empty_slugs(store: GraphStore):
    with pytest.raises(ValueError, match="empty slug"):
        store.upsert_entities([{"name": "!!!"}], "a.py")


# ---------------------------------------------------------------------------
# GraphStore — relationships
# ---------------------------------------------------------------------------


def test_upsert_relationship_creates(store: GraphStore):
    add_entity(store, "embed")
    add_entity(store, "ollama_client")
    add_relationship(store, "embed", "ollama_client", "uses", "api/embed.py")
    _, relationships = _neighborhood(store, "embed")
    labels = [r["label"] for r in relationships]
    assert "uses" in labels


def test_upsert_relationship_dedup_increments_weight(store: GraphStore):
    add_entity(store, "a")
    add_entity(store, "b")
    add_relationship(store, "a", "b", "calls", "file.py")
    add_relationship(store, "a", "b", "calls", "file.py")
    _, relationships = _neighborhood(store, "a")
    weight = relationships[0]["weight"]
    assert weight == 2.0


# ---------------------------------------------------------------------------
# GraphStore — neighborhood expansion
# ---------------------------------------------------------------------------


def _link(store: GraphStore, *edges: tuple[str, str], doc: str = "doc.py") -> None:
    for source, target in edges:
        add_entity(store, source)
        add_entity(store, target)
        add_relationship(store, source, target, "uses", doc)


def test_expand_neighborhood_hops_zero_returns_seeds_in_given_order(store: GraphStore):
    _link(store, ("b", "a"), ("a", "c"))
    entities, relationships = store.expand_neighborhood(
        ["b", "a"], 0, max_entities=10, max_relationships=10
    )
    assert [e["id"] for e in entities] == ["b", "a"]
    # Edges among the selected entities first, then edges out to unselected ones.
    assert [(r["source_id"], r["target_id"]) for r in relationships] == [("b", "a"), ("a", "c")]


def test_expand_neighborhood_lists_internal_edges_before_heavier_boundary_edges(
    store: GraphStore,
):
    _link(store, ("a", "b"))
    for doc in ("d1.py", "d2.py", "d3.py"):
        _link(store, ("a", "outside"), doc=doc)
    _, relationships = store.expand_neighborhood(
        ["a", "b"], 0, max_entities=2, max_relationships=1
    )
    assert [(r["source_id"], r["target_id"]) for r in relationships] == [("a", "b")]


def test_expand_neighborhood_takes_neighbors_round_robin_across_seeds(store: GraphStore):
    """A hub seed must not spend the whole budget before the next seed gets a neighbor."""
    _link(store, ("hub", "h1"), ("hub", "h2"), ("hub", "h3"), ("leaf", "l1"))
    entities, _ = store.expand_neighborhood(
        ["hub", "leaf"], 1, max_entities=4, max_relationships=10
    )
    assert [e["id"] for e in entities] == ["hub", "leaf", "h1", "l1"]


def test_expand_neighborhood_prefers_heavier_edges_in_either_direction(store: GraphStore):
    _link(store, ("light", "seed"))
    _link(store, ("seed", "heavy"), doc="d1.py")
    _link(store, ("seed", "heavy"), doc="d2.py")
    entities, _ = store.expand_neighborhood(["seed"], 1, max_entities=2, max_relationships=10)
    assert [e["id"] for e in entities] == ["seed", "heavy"]


def test_expand_neighborhood_reaches_second_hop_only_when_asked(store: GraphStore):
    _link(store, ("a", "b"), ("b", "c"))
    one_hop, _ = store.expand_neighborhood(["a"], 1, max_entities=10, max_relationships=10)
    two_hops, relationships = store.expand_neighborhood(
        ["a"], 2, max_entities=10, max_relationships=1
    )
    assert [e["id"] for e in one_hop] == ["a", "b"]
    assert [e["id"] for e in two_hops] == ["a", "b", "c"]
    assert len(relationships) == 1


def test_get_entities_by_chunk_ids_preserves_chunk_order(store: GraphStore):
    for name in ("zeta", "alpha", "mid"):
        add_entity(store, name)
    store.register_chunks([("c1", "f.py", 0), ("c2", "f.py", 1)])
    store.link_chunks([("c1", "zeta"), ("c2", "alpha"), ("c2", "mid"), ("c1", "mid")])

    assert store.get_entities_by_chunk_ids(["c1", "c2"]) == ["mid", "zeta", "alpha"]
    assert store.get_entities_by_chunk_ids(["c2", "c1"]) == ["alpha", "mid", "zeta"]


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
    add_entity(store, "orphan", doc="file_a.py")
    add_entity(store, "shared", doc="file_a.py")
    store.register_chunks([("c1", "file_a.py", 0)])
    add_relationship(store, "orphan", "shared", "uses", "file_a.py")

    store.delete_file_data("file_a.py")

    assert store.get_chunks_for_file("file_a.py") == []
    assert _entity(store, "orphan") is None


def test_delete_file_data_keeps_shared_entity(store: GraphStore):
    add_entity(store, "shared", doc="file_a.py")
    add_entity(store, "other", doc="file_a.py")
    store.register_chunks([("c1", "file_a.py", 0), ("c2", "file_b.py", 0)])
    add_relationship(store, "shared", "other", "uses", "file_a.py")
    add_relationship(store, "shared", "other", "uses", "file_b.py")

    store.delete_file_data("file_a.py")

    # shared entity still referenced by file_b.py relationship
    assert _entity(store, "shared") is not None


def test_delete_file_data_keeps_entity_referenced_only_via_chunk_link(store: GraphStore):
    """An entity with a relationship only in file_a, but chunk-linked from file_b too,
    must survive deleting file_a — without raising IntegrityError.

    Regression for the orphan-cleanup bug: chunk_entities.entity_id has no ON DELETE
    clause and PRAGMA foreign_keys=ON is set, so deleting a still-chunk-linked entity
    used to raise sqlite3.IntegrityError and roll back the whole delete_file_data
    transaction rather than just mishandling the orphan check.
    """
    slug = add_entity(store, "shared", doc="file_a.py")
    add_entity(store, "other", doc="file_a.py")
    store.register_chunks([("c1", "file_a.py", 0), ("c2", "file_b.py", 0)])
    add_relationship(store, "shared", "other", "uses", "file_a.py")
    store.link_chunks([("c2", slug)])

    store.delete_file_data("file_a.py")  # must not raise IntegrityError

    assert _entity(store, slug) is not None


def test_delete_file_data_leaves_other_documents_entities_alone(store: GraphStore):
    """Only entities the deleted file touched are orphan candidates (no global sweep)."""
    add_entity(store, "loner", doc="b.py")  # mentioned, but no relationships or chunk links
    add_entity(store, "mine", doc="a.py")

    store.delete_file_data("a.py")

    assert _entity(store, "loner") is not None
    assert _entity(store, "mine") is None


def test_delete_file_data_removes_only_that_files_chunks_and_its_fingerprint(store: GraphStore):
    store.register_chunks([
        ("uuid-A", "target.py", 0),
        ("uuid-B", "target.py", 1),
        ("uuid-C", "other.py", 0),
    ])
    store.upsert_hash("target.py", "h")

    store.delete_file_data("target.py")

    assert store.get_chunks_for_file("target.py") == []
    assert store.get_chunks_for_file("other.py") == ["uuid-C"]
    assert store.get_hash("target.py") is None  # its data is gone, so it must be re-indexed


# ---------------------------------------------------------------------------
# GraphStore — extraction cache
# ---------------------------------------------------------------------------


def test_extraction_cache_replays_only_a_matching_prompt_hash(store: GraphStore):
    store.cache_extraction("foo.py", 0, "hash-a", EMPTY_EXTRACTION_JSON)
    assert store.get_cached_extraction("foo.py", 0, "hash-a") == EMPTY_EXTRACTION_JSON
    assert store.get_cached_extraction("foo.py", 0, "hash-b") is None
    assert store.get_cached_extraction("foo.py", 1, "hash-a") is None


def test_purge_file_removes_graph_data_fingerprint_and_cache(store: GraphStore):
    add_entity(store, "Gone", doc="foo.py")
    store.register_chunks([("c1", "foo.py", 0)])
    store.upsert_hash("foo.py", "h")
    store.cache_extraction("foo.py", 0, "hash-a", "{}")

    store.purge_file("foo.py")

    assert store.list_all_paths() == []
    assert _entity(store, "gone") is None


# ---------------------------------------------------------------------------
# extract_entities_for_file
# ---------------------------------------------------------------------------


def test_extract_entities_for_file_requests_json_format(store: GraphStore, monkeypatch):
    captured: dict = {}

    def _fake_generate(*args, **kwargs):
        captured.update(kwargs)
        return EMPTY_EXTRACTION_JSON

    patch_ollama_generate(monkeypatch, _fake_generate)

    result = extract_entities_for_file(["some chunk text"], "foo.py", store)

    assert captured.get("format") == "json"
    assert captured.get("options") == {"temperature": 0}
    assert result.had_failure is False


def test_extract_entities_for_file_isolates_batch_failure(store: GraphStore, monkeypatch):
    """A bad batch must not abort the whole file or skip caching prior batches.

    Forces 2 batches (one chunk each) by shrinking EXTRACT_BATCH_TOKENS to 1. The
    first batch's response is cached and contributes its entity; the second
    batch's generate() raises, degrading to an empty result for that batch only —
    and isn't cached, so it retries on the next run.
    """
    monkeypatch.setattr("local_graph_rag.graph.extractor.EXTRACT_BATCH_TOKENS", 1)

    call_count = 0

    def _fake_generate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return (
                '{"entities": [{"name": "Alpha", "type": "CLASS", "description": "a"}],'
                ' "relationships": []}'
            )
        raise RuntimeError("boom")

    patch_ollama_generate(monkeypatch, _fake_generate)

    result = extract_entities_for_file(
        ["alpha entity chunk", "beta entity chunk"], "foo.py", store
    )

    assert [e["name"] for e in result.entities] == ["Alpha"]
    assert result.had_failure is True

    cached = store.conn.execute("SELECT batch_index FROM extraction_cache").fetchall()
    assert [row["batch_index"] for row in cached] == [0]


def test_extraction_cache_never_replays_results_for_edited_text(store: GraphStore, monkeypatch):
    """Regression: a never-fingerprinted file edited between runs replayed stale batches."""
    monkeypatch.setattr("local_graph_rag.graph.extractor.EXTRACT_BATCH_TOKENS", 1)
    prompts: list[str] = []

    def _recording_generate(prompt: str, *args, **kwargs) -> str:
        """Fake extraction: one entity named after the first word of the batch text."""
        prompts.append(prompt)
        name = prompt.rsplit("Text:\n", 1)[1].split()[0]
        return json.dumps({"entities": [{"name": name}], "relationships": []})

    patch_ollama_generate(monkeypatch, _recording_generate)
    extract_entities_for_file(["Alpha text", "Beta text"], "f.md", store)
    prompts.clear()

    result = extract_entities_for_file(["Alpha text", "Gamma text"], "f.md", store)

    assert [e["name"] for e in result.entities] == ["Alpha", "Gamma"]
    assert len(prompts) == 1  # unchanged batch replayed from cache; edited batch re-extracted


def test_extraction_cache_misses_after_model_change(store: GraphStore, monkeypatch):
    calls: list[int] = []
    patch_ollama_generate(monkeypatch, lambda *a, **kw: calls.append(1) or EMPTY_EXTRACTION_JSON)

    extract_entities_for_file(["some text"], "f.md", store)
    monkeypatch.setattr("local_graph_rag.graph.extractor.EXTRACT_MODEL", "another-model")
    extract_entities_for_file(["some text"], "f.md", store)

    assert len(calls) == 2


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
    slug = add_entity(store, "isolated")
    store.conn.execute("UPDATE entities SET community = ? WHERE id = ?", (7, slug))
    store.conn.commit()

    store.detect_communities()

    assert _entity(store, slug)["community"] is None


def test_detect_communities_resets_isolated_entity_in_nonempty_graph(store: GraphStore):
    """An isolated entity must lose its stale community even when OTHER entities
    form a non-empty graph and get freshly assigned.

    Distinct code path from the empty-graph case: best_partition() runs and returns
    a non-empty partition for the connected pair, but the isolated entity never
    becomes a node (build_networkx_graph only adds nodes via add_edge) — so it's
    absent from the partition and must be cleared by the reset, not left stale.
    """
    a = add_entity(store, "connected_a")
    add_entity(store, "connected_b")
    isolated = add_entity(store, "isolated")
    add_relationship(store, "connected_a", "connected_b", "uses", "doc.py")
    store.conn.execute("UPDATE entities SET community = ? WHERE id = ?", (7, isolated))
    store.conn.commit()

    store.detect_communities()

    assert _entity(store, isolated)["community"] is None
    assert _entity(store, a)["community"] is not None


def test_build_networkx_graph_sums_weights_across_docs_labels_and_directions(store: GraphStore):
    add_entity(store, "a")
    add_entity(store, "b")
    for doc in ("doc1.py", "doc2.py", "doc3.py"):
        add_relationship(store, "a", "b", "uses", doc)
    add_relationship(store, "b", "a", "calls", "doc1.py")

    graph = store.build_networkx_graph()

    assert not graph.is_directed()
    assert graph.number_of_edges() == 1
    assert graph["a"]["b"]["weight"] == 4.0


def _community_assignments(store: GraphStore) -> dict[str, int]:
    rows = store.conn.execute("SELECT id, community FROM entities WHERE community IS NOT NULL")
    return {row["id"]: row["community"] for row in rows}


def test_detect_communities_is_deterministic_on_an_unchanged_graph(store: GraphStore):
    names = [f"n{i}" for i in range(30)]
    for name in names:
        add_entity(store, name)
    for i in range(30):
        # Three loosely linked rings of ten: several communities, many tie-breaking choices.
        add_relationship(store, names[i], names[(i + 1) % 10 + 10 * (i // 10)], "uses", "d.py")
        add_relationship(store, names[i], names[(i * 7) % 30], "mentions", "d.py")

    store.detect_communities()
    first = _community_assignments(store)
    for _ in range(3):
        store.detect_communities()
        assert _community_assignments(store) == first


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
    result = _parse_extraction_response(EMPTY_EXTRACTION_JSON)
    assert result.entities == []
    assert result.relationships == []


def test_parse_valid_json_with_none_word_in_string_untouched():
    response = (
        '{"entities": [{"name": "Finder", "type": "FUNCTION",'
        ' "description": "Returns None if not found"}], "relationships": []}'
    )
    result = _parse_extraction_response(response)
    assert result.entities[0]["description"] == "Returns None if not found"


def test_parse_non_dict_top_level_returns_empty():
    for response in ("null", "[1, 2, 3]"):
        result = _parse_extraction_response(response)
        assert result.entities == []
        assert result.relationships == []


def test_parse_dict_with_non_list_entities_returns_empty():
    for response in (
        '{"entities": null, "relationships": []}',
        '{"entities": 5, "relationships": []}',
    ):
        result = _parse_extraction_response(response)
        assert result.entities == []
