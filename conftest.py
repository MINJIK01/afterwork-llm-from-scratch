import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def impl(request):
    """현재 test_impl.py와 같은 디렉터리의 impl.py를 고유한 모듈 이름으로 로드한다."""
    directory = Path(request.fspath).parent
    spec = importlib.util.spec_from_file_location(f"{directory.name}_impl", directory / "impl.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
