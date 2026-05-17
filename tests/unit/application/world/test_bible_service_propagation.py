"""Test character rename propagation to story_nodes and triples."""
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from domain.novel.value_objects.novel_id import NovelId
from infrastructure.persistence.database.connection import DatabaseConnection
from application.world.services.bible_service import BibleService

SCHEMA_PATH = (
    Path(__file__).resolve().parents[4] / "infrastructure" / "persistence" / "database" / "schema.sql"
)

class DummyCharacter:
    def __init__(self, id, name):
        self.id = id
        self.name = name

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "t.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    # Insert a novel
    conn.execute(
        "INSERT INTO novels (id, title, slug, target_chapters) VALUES ('n1', 'T', 'slug1', 0)"
    )

    # Insert story nodes with the old name
    conn.execute(
        """
        INSERT INTO story_nodes (
            id, novel_id, parent_id, node_type, number, title, description, order_index,
            planning_status, planning_source, chapter_start, chapter_end, chapter_count,
            suggested_chapter_count, content, outline, word_count, status, themes, key_events,
            narrative_arc, conflicts, pov_character_id, timeline_start, timeline_end, metadata,
            created_at, updated_at
        ) VALUES (
            'sn-1', 'n1', NULL, 'chapter', 1, '旧人名的大冒险', '旧人名在这里出现了', 1,
            'draft', 'manual', 1, 1, 1, NULL, '', '这章讲了旧人名的故事', 0, 'draft', '[]', '[]',
            '', '[]', NULL, '', '', '{}', '2026-05-17T12:00:00', '2026-05-17T12:00:00'
        )
        """
    )

    # Insert triples with old name and entity IDs
    conn.execute(
        """
        INSERT INTO triples (
            id, novel_id, subject, predicate, object, chapter_number, note,
            entity_type, importance, location_type, description, first_appearance,
            confidence, source_type, subject_entity_id, object_entity_id
        ) VALUES (
            't-1', 'n1', '旧人名', '爱上', '另一个角色', NULL, '',
            'character', 'normal', NULL, NULL, NULL,
            1.0, 'bible_generated', 'char-1', NULL
        )
        """
    )
    # Insert triple with matching object_entity_id
    conn.execute(
        """
        INSERT INTO triples (
            id, novel_id, subject, predicate, object, chapter_number, note,
            entity_type, importance, location_type, description, first_appearance,
            confidence, source_type, subject_entity_id, object_entity_id
        ) VALUES (
            't-2', 'n1', '另一个角色', '帮助', '旧人名', NULL, '',
            'character', 'normal', NULL, NULL, NULL,
            1.0, 'bible_generated', NULL, 'char-1'
        )
        """
    )
    # Insert triple with old name but empty entity ID (fallback scenario)
    conn.execute(
        """
        INSERT INTO triples (
            id, novel_id, subject, predicate, object, chapter_number, note,
            entity_type, importance, location_type, description, first_appearance,
            confidence, source_type, subject_entity_id, object_entity_id
        ) VALUES (
            't-3', 'n1', '旧人名', '看见', '旧人名', NULL, '',
            'character', 'normal', NULL, NULL, NULL,
            1.0, 'bible_generated', '', ''
        )
        """
    )

    conn.commit()
    conn.close()

    # Mock get_db_path to return this temp db
    monkeypatch.setattr("application.paths.get_db_path", lambda: db_path)

    # Clear any cached DatabaseConnection for this path to ensure fresh one is created/used
    import infrastructure.persistence.database.connection as conn_mod
    conn_mod._db_instance = None
    conn_mod._connection_pool_instance = None

    return db_path

def test_propagate_character_renames(temp_db):
    # Initialize the BibleService
    mock_repo = MagicMock()
    service = BibleService(mock_repo)

    # Previous names mapping: char-1 had the name "旧人名"
    prev_name_by_id = {"char-1": "旧人名"}

    # New characters: char-1 is now named "新人名"
    new_characters = [DummyCharacter("char-1", "新人名")]

    # Call the propagation method
    service._propagate_character_renames("n1", prev_name_by_id, new_characters)

    # Verify the changes in the database
    conn = sqlite3.connect(str(temp_db))
    conn.row_factory = sqlite3.Row

    # 1. Verify story_nodes text replaced
    node = conn.execute("SELECT * FROM story_nodes WHERE id = 'sn-1'").fetchone()
    assert node["title"] == "新人名的大冒险"
    assert node["description"] == "新人名在这里出现了"
    assert node["outline"] == "这章讲了新人名的故事"

    # 2. Verify triples with subject_entity_id / object_entity_id matching 'char-1' updated
    t1 = conn.execute("SELECT * FROM triples WHERE id = 't-1'").fetchone()
    assert t1["subject"] == "新人名"

    t2 = conn.execute("SELECT * FROM triples WHERE id = 't-2'").fetchone()
    assert t2["object"] == "新人名"

    # 3. Verify triples with empty/null entity_id but matching names updated
    t3 = conn.execute("SELECT * FROM triples WHERE id = 't-3'").fetchone()
    assert t3["subject"] == "新人名"
    assert t3["object"] == "新人名"

    conn.close()
