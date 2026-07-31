"""Tests for HINGraph workspace influence zones."""

from aero_forge.healing.hin_graph import (
    build_workspace_hingraph,
    delta_m_influence,
    influence_zone,
)


def test_build_workspace_hingraph(tmp_path):
    (tmp_path / "a.py").write_text("import b\ndef main():\n    pass")
    (tmp_path / "b.py").write_text("def helper():\n    pass")
    adj, nodes = build_workspace_hingraph(tmp_path)
    assert "a" in nodes
    assert "b" in nodes
    assert adj["a"] == ["b"]
    assert adj["b"] == []


def test_influence_zone(tmp_path):
    (tmp_path / "a.py").write_text("import b\nimport c\ndef main():\n    pass")
    (tmp_path / "b.py").write_text("import c\ndef helper():\n    pass")
    (tmp_path / "c.py").write_text("def base():\n    pass")
    affected, waves = influence_zone(tmp_path, ["b"])
    assert "b" in affected
    assert "a" in affected  # a depends on b
    assert "c" in affected  # b depends on c
    assert len(waves) > 0


def test_delta_m_influence(tmp_path):
    (tmp_path / "a.py").write_text("import b\nimport c")
    (tmp_path / "b.py").write_text("import c")
    (tmp_path / "c.py").write_text("")
    impacted = delta_m_influence(tmp_path, ["c"])
    assert impacted["c"] == ["a", "b"]
