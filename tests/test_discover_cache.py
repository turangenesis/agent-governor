"""Offline replay: with the network hard-blocked, `search_by_query` must still
return real cached candidates (the stage-safe demo path).

Hermetic: the `seeded_cache` fixture writes the cache file this test reads and
removes it afterward, so the test passes on a fresh checkout / offline CI. It does
NOT depend on (or clobber) a `cache_*.json` left behind by a prior live search."""
from __future__ import annotations

import os
import socket

import pytest

from governor.discover import _cache_path, save, search_by_query
from governor.models import Candidate

# A dedicated query used only by this test. The fixture seeds its cache file.
CACHED_QUERY = "hermetic-test type:user"

_FIXTURE_CANDIDATES = [
    Candidate(name="Ada Lovelace", current_company="Analytical Engines",
              current_title="Principal Backend Engineer", seniority="senior",
              email="ada@users.noreply.github.com",
              matched_role="Senior Backend Engineer", match_confidence=0.82),
    Candidate(name="Grace Hopper", current_company="Independent / open source",
              current_title="Distributed Systems Lead", seniority="senior",
              email="grace@users.noreply.github.com",
              matched_role="Senior Backend Engineer", match_confidence=0.90),
]


@pytest.fixture
def seeded_cache():
    """Write a deterministic cache file for CACHED_QUERY, remove it after."""
    path = _cache_path(CACHED_QUERY)
    save(_FIXTURE_CANDIDATES, path)
    try:
        yield path
    finally:
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture
def no_network(monkeypatch):
    """Any attempt to open a socket blows up, proving we never hit the network."""
    def _boom(*_a, **_k):
        raise AssertionError("network access attempted during offline replay")

    monkeypatch.setattr(socket, "socket", _boom)
    # requests builds connections through these too — belt and suspenders.
    monkeypatch.setattr(socket, "create_connection", _boom)


def test_cache_file_exists_for_query(seeded_cache):
    assert os.path.exists(seeded_cache)


def test_search_by_query_replays_from_cache_offline(seeded_cache, no_network):
    cands = search_by_query(CACHED_QUERY, limit=4)
    assert cands, "expected cached candidates, got none"
    assert all(isinstance(c, Candidate) for c in cands)
    assert all(c.name for c in cands)
    assert all(0.0 <= c.match_confidence <= 1.0 for c in cands)
