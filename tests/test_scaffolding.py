def test_version():
    from lsbi import __version__ as v1
    from lsbi._version import __version__ as v2

    assert v1 == v2
