"""테스트와 스크립트가 import할 minillm 패키지의 위치를 정한다.

- 기본: minillm/ (본인 구현)
- MINILLM_SOLUTIONS=1: solutions/minillm/ (정답)
- MINILLM_ONLY=08 또는 6,7,8: 지정한 step의 TODO 구간만 minillm/에서, 나머지는 solutions/에서 가져와
  .grade/minillm/에 합성한다. 앞 단계를 구현하지 않아도 해당 step만 채점할 수 있다.
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOL = re.compile(r"^\s*# BEGIN SOLUTION \(step (\d+)\)\s*$")
TODO = re.compile(r"^\s*# BEGIN TODO \(step (\d+)\)\s*$")
END = re.compile(r"^\s*# END (SOLUTION|TODO)\s*$")


def todo_regions(text):
    """{step: [구간 본문(줄 리스트), ...]} — 파일 내 등장 순서대로."""
    out, cur = {}, None
    for line in text.splitlines(keepends=True):
        if cur is None:
            if m := TODO.match(line):
                cur = (int(m.group(1)), [])
        elif END.match(line):
            out.setdefault(cur[0], []).append(cur[1])
            cur = None
        else:
            cur[1].append(line)
    return out


def merge(sol_text, user_text, steps, name):
    """정답 파일을 바탕으로, steps에 속한 구간만 본인 파일의 같은 순번 TODO 구간으로 교체."""
    user, out, skipping = todo_regions(user_text), [], False
    for line in sol_text.splitlines(keepends=True):
        if skipping:
            skipping = not END.match(line)
            continue
        m = SOL.match(line)
        if m and int(m.group(1)) in steps:
            bodies = user.get(int(m.group(1)))
            if not bodies:
                raise SystemExit(f"minillm/{name}: step {m.group(1)}의 '# BEGIN TODO' 구간이 부족하다. 마커를 지우지 말 것.")
            out += bodies.pop(0)
            skipping = True
        else:
            out.append(line)
    return "".join(out)


def build(steps):
    dst = ROOT / ".grade" / "minillm"
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted((ROOT / "solutions" / "minillm").glob("*.py")):
        (dst / f.name).write_text(merge(f.read_text(), (ROOT / "minillm" / f.name).read_text(), steps, f.name))
    return dst.parent


def minillm_path():
    if os.environ.get("MINILLM_SOLUTIONS") == "1":
        return ROOT / "solutions"
    if only := os.environ.get("MINILLM_ONLY"):
        return build({int(s) for s in only.split(",")})
    return ROOT
