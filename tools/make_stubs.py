"""solutions/minillm의 BEGIN/END SOLUTION 구간을 비워 minillm/ 실습 스텁을 생성한다.

주의: minillm/의 기존 파일을 덮어쓴다. 작성 중인 구현이 있다면 먼저 백업할 것.
사용: python tools/make_stubs.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN = re.compile(r"^(\s*)# BEGIN SOLUTION \(step (\d+)\)\s*$")
END = re.compile(r"^\s*# END SOLUTION\s*$")


def stub(src):
    out, skipping = [], False
    for line in src.splitlines(keepends=True):
        if skipping:
            skipping = not END.match(line)
            continue
        if m := BEGIN.match(line):
            indent, step = m.groups()
            out.append(f'{indent}raise NotImplementedError("TODO: step {step} (docs/{step}_*.md)")\n')
            skipping = True
        else:
            out.append(line)
    assert not skipping, "END SOLUTION 누락"
    return "".join(out)


if __name__ == "__main__":
    dst = ROOT / "minillm"
    if any(dst.glob("*.py")) and "--force" not in sys.argv:
        sys.exit("minillm/이 이미 존재합니다. 덮어쓰려면 --force")
    dst.mkdir(exist_ok=True)
    for f in sorted((ROOT / "solutions" / "minillm").glob("*.py")):
        (dst / f.name).write_text(stub(f.read_text()))
        print("wrote", dst / f.name)
