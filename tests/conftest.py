"""The one test command is network-free by construction, not convention."""

import socket

import pytest


@pytest.fixture(autouse=True)
def forbid_test_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Tests are offline: unexpected network connection")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
