"""Offline glossary search contract tests."""

import io
import zipfile
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from wenyi_backend.routers import glossary
from wenyi_core.models.document import Chapter, Document, Segment
from wenyi_core.models.glossary import GlossaryTerm
from wenyi_core.storage.sqlite import SqliteStorage


@pytest.mark.parametrize(
    "query", ["  rEaDiNg  ", "  nOtE  ", "  aLiAs  ", "  sOuRcE  ", "  tArGeT  ", "   "]
)
@pytest.mark.parametrize("term_type,expected", [(None, 2), ("person", 1), ("place", 0)])
def test_list_terms_searches_text_fields_and_preserves_type_filter(
    backend_context, query, term_type, expected
):
    backend_context.repository.get_project.return_value = {"id": "project-1", "fmt": "text"}
    terms = [
        GlossaryTerm(
            source=f"Source {kind}",
            target=f"Target {kind}",
            reading="Reading",
            note="Note",
            aliases=["Alias"],
            type=kind,
        )
        for kind in ("person", "term")
    ]
    backend_context.storage_for.return_value.all_terms.return_value = terms

    result = glossary.list_terms("project-1", q=query, type=term_type)

    assert result == [vars(term) for term in terms[:expected]]


def test_term_pages_sort_by_source_without_reordering_context(backend_context, tmp_path):
    storage = SqliteStorage(str(tmp_path / "project"))
    sources = ["𐀀", "中", "a", "Z", "Å", "\ue000"]
    for source in sources:
        storage.upsert_term(GlossaryTerm(source=source, target="Translation"))
    backend_context.repository.get_project.return_value = {"id": "project-1", "fmt": "text"}
    backend_context.read_storage_for.return_value = storage
    app = FastAPI()
    app.include_router(glossary.router)
    with TestClient(app) as client:
        result = client.get("/projects/project-1/glossary/terms/page?offset=2&limit=2")
        assert result.status_code == 200, result.text
        assert result.json()["total"] == len(sources)
        assert [term["source"] for term in result.json()["items"]] == sorted(sources)[2:4]
        assert client.get("/projects/project-1/glossary/terms/page?offset=100").json() == {
            "items": [],
            "total": len(sources),
        }
        for query in ("offset=-1", "limit=0", "limit=201"):
            assert client.get(f"/projects/project-1/glossary/terms/page?{query}").status_code == 422
    assert [term.source for term in storage.all_terms()] == sources
    storage.close()


def test_term_lookup_uses_full_glossary_not_current_filter(backend_context, tmp_path):
    storage = SqliteStorage(str(tmp_path / "project"))
    storage.upsert_term(GlossaryTerm(source="Hidden", target="Existing", type="person"))
    backend_context.repository.get_project.return_value = {"id": "project-1", "fmt": "text"}
    backend_context.read_storage_for.return_value = storage
    app = FastAPI()
    app.include_router(glossary.router)
    with TestClient(app) as client:
        response = client.post(
            "/projects/project-1/glossary/terms/lookup", json={"sources": ["Hidden", "Missing"]}
        )
        assert response.status_code == 200, response.text
        assert [term["source"] for term in response.json()] == ["Hidden"]
        assert (
            client.post(
                "/projects/project-1/glossary/terms/lookup", json={"sources": ["x"] * 501}
            ).status_code
            == 422
        )
    storage.close()


def test_anki_export_uses_full_glossary_and_closes_storage(backend_context, monkeypatch):
    backend_context.repository.get_project.return_value = {
        "id": "project-1",
        "name": "Project name",
        "book_title": "Book title",
        "fmt": "txt",
        "source_lang": "ja",
        "target_lang": "zh",
    }
    terms = [
        GlossaryTerm(
            source=f"原文 {index:03}", target="译文", type="person" if index % 2 else "term"
        )
        for index in range(135)
    ]
    storage = backend_context.storage_for.return_value
    storage.all_terms.return_value = terms
    export = Mock(wraps=glossary.export_anki_glossary)
    monkeypatch.setattr(glossary, "export_anki_glossary", export)
    app = FastAPI()
    app.include_router(glossary.router)
    with TestClient(app) as client:
        response = client.get("/projects/project-1/glossary/export?format=apkg")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["content-disposition"] == "attachment; filename=glossary-project-1.apkg"
    assert int(response.headers["content-length"]) == len(response.content)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {"collection.anki2", "media"}
    export.assert_called_once_with(
        terms, project_id="project-1", title="Book title", source_lang="ja", target_lang="zh"
    )
    storage.all_terms.assert_called_once_with()
    storage.close.assert_called_once_with()
    backend_context.enqueue.assert_not_called()
    storage.upsert_term.assert_not_called()
    storage.log_event.assert_not_called()


def test_anki_export_rejects_missing_projects(backend_context):
    app = FastAPI()
    app.include_router(glossary.router)
    with TestClient(app) as client:
        backend_context.repository.get_project.return_value = None
        assert client.get("/projects/missing/glossary/export?format=apkg").status_code == 404
    backend_context.storage_for.assert_not_called()


def _prepared_glossary(backend_context, tmp_path, chapters: list[str], terms: list[GlossaryTerm]):
    """Serve a real prepared SQLite project through both storage factories."""
    run_dir = str(tmp_path / "project")
    (tmp_path / "book.txt").write_text("\n".join(chapters), encoding="utf-8")
    storage = SqliteStorage(run_dir)
    storage.init_from_document(
        Document(
            title="Book",
            source_path=str(tmp_path / "book.txt"),
            fmt="txt",
            source_lang="zh",
            target_lang="en",
            chapters=[
                Chapter(index=index, segments=[Segment(index=0, source=text)])
                for index, text in enumerate(chapters)
            ],
        )
    )
    for term in terms:
        storage.upsert_term(term)
    storage.close()
    backend_context.repository.get_project.return_value = {"id": "project-1", "status": "idle"}
    backend_context.storage_for.side_effect = lambda pid: SqliteStorage(run_dir, create=False)
    backend_context.read_storage_for.side_effect = lambda pid: SqliteStorage(run_dir, create=False)
    app = FastAPI()
    app.include_router(glossary.router)
    return TestClient(app), run_dir


def test_one_off_lists_terms_that_occur_once_in_the_whole_book(backend_context, tmp_path):
    client, run_dir = _prepared_glossary(
        backend_context,
        tmp_path,
        ["兰斯走进酒馆。手提箱放在桌上。", "兰斯又笑了。人名甲，人名乙。"],
        [
            GlossaryTerm(source="兰斯", target="Lance", type="person"),
            GlossaryTerm(source="手提箱", target="suitcase", type="item"),
            GlossaryTerm(source="人名甲", target="A", type="person", aliases=["人名乙"]),
            GlossaryTerm(source="不存在", target="absent"),
        ],
    )
    with client:
        response = client.post("/projects/project-1/glossary/one-off")
    assert response.status_code == 200, response.text
    # A term counts across chapters, and an alias mention counts for named entities.
    assert [term["source"] for term in response.json()] == ["手提箱", "不存在"]
    storage = SqliteStorage(run_dir, create=False)
    assert len(storage.all_terms()) == 4
    storage.close()


def test_one_off_requires_a_prepared_book(backend_context):
    storage = Mock()
    storage.exists.return_value = False
    backend_context.repository.get_project.return_value = {"id": "project-1"}
    backend_context.read_storage_for.return_value = storage
    app = FastAPI()
    app.include_router(glossary.router)
    with TestClient(app) as client:
        assert client.post("/projects/project-1/glossary/one-off").status_code == 409
    storage.close.assert_called_once_with()


def test_matching_sources_cover_every_page_of_the_filter(backend_context, tmp_path):
    terms = [GlossaryTerm(source=f"词{index:03}", target="w", type="term") for index in range(250)]
    terms.append(GlossaryTerm(source="人名", target="Name", type="person"))
    client, _run_dir = _prepared_glossary(backend_context, tmp_path, ["正文。"], terms)
    with client:
        everything = client.get("/projects/project-1/glossary/sources").json()
        generic = client.get("/projects/project-1/glossary/sources?type=term").json()
        searched = client.get("/projects/project-1/glossary/sources?q=词24").json()
        nothing = client.get("/projects/project-1/glossary/sources?q=absent").json()
    assert len(everything) == 251
    assert len(generic) == 250 and "人名" not in generic
    assert searched == [f"词24{digit}" for digit in range(10)]
    assert nothing == []


def test_bulk_delete_removes_named_terms_and_closes_their_conflicts(backend_context, tmp_path):
    client, run_dir = _prepared_glossary(
        backend_context,
        tmp_path,
        ["正文。"],
        [
            GlossaryTerm(source="甲", target="A"),
            GlossaryTerm(source="乙", target="B"),
            GlossaryTerm(source="丙", target="C"),
        ],
    )
    storage = SqliteStorage(run_dir, create=False)
    storage.upsert_term(GlossaryTerm(source="甲", target="Other"))
    storage.upsert_term(GlossaryTerm(source="丙", target="Other"))
    storage.close()
    with client:
        response = client.post(
            "/projects/project-1/glossary/bulk-delete", json={"sources": ["甲", "甲", "缺"]}
        )
    assert response.status_code == 200, response.text
    assert response.json() == {"deleted": 1}
    storage = SqliteStorage(run_dir, create=False)
    assert [term.source for term in storage.all_terms()] == ["乙", "丙"]
    assert [row["source"] for row in storage.open_conflicts()] == ["丙"]
    assert storage.list_events(event_type="glossary_terms_deleted")[0]["count"] == 1
    storage.close()


def test_keep_current_closes_every_conflict_without_changing_translations(
    backend_context, tmp_path
):
    client, run_dir = _prepared_glossary(
        backend_context,
        tmp_path,
        ["正文。"],
        [GlossaryTerm(source="老板", target="boss"), GlossaryTerm(source="城", target="city")],
    )
    storage = SqliteStorage(run_dir, create=False)
    for proposal in ("chief", "owner"):
        storage.upsert_term(GlossaryTerm(source="老板", target=proposal))
    storage.upsert_term(GlossaryTerm(source="城", target="town"))
    storage.close()
    with client:
        response = client.post("/projects/project-1/glossary/conflicts/keep-current")
    assert response.status_code == 200, response.text
    assert response.json() == {"terms": 2, "conflicts": 3}
    storage = SqliteStorage(run_dir, create=False)
    assert storage.open_conflicts() == []
    assert [(term.target, term.status) for term in storage.all_terms()] == [
        ("boss", "ok"),
        ("city", "ok"),
    ]
    # A rejected alternative stays closed when extraction proposes it again.
    assert storage.upsert_term(GlossaryTerm(source="老板", target="chief")) == "unchanged"
    storage.close()


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("post", "/projects/project-1/glossary/bulk-delete", {"sources": ["甲"]}),
        ("post", "/projects/project-1/glossary/conflicts/keep-current", None),
    ],
)
def test_bulk_writes_are_refused_while_a_task_runs(backend_context, tmp_path, method, path, body):
    client, run_dir = _prepared_glossary(
        backend_context, tmp_path, ["正文。"], [GlossaryTerm(source="甲", target="A")]
    )
    backend_context.repository.get_project.return_value = {
        "id": "project-1",
        "status": "translating",
    }
    with client:
        response = getattr(client, method)(path, **({"json": body} if body else {}))
    assert response.status_code == 409
    storage = SqliteStorage(run_dir, create=False)
    assert [term.source for term in storage.all_terms()] == ["甲"]
    storage.close()
