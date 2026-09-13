import so_recon


def test_version_and_spec_version() -> None:
    assert so_recon.__version__ == "0.0.1"
    assert so_recon.SPEC_VERSION == "3.0"
