import importlib


def test_installation():
    packages = ["pandas", "numpy", "sklearn"]
    for package in packages:
        try:
            importlib.import_module(package)
            assert True, f"{package} is installed."
        except ImportError:
            assert False, f"{package} is not installed."
