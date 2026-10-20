"""Hard tier for the task-completion test: specs with many interacting rules.

The first 16 tasks (hybrid/task_specs.py) were usually solved by Haiku on the first attempt, so they
cannot show what happens when a cheap model needs several tries. These four are where it might.
"""

HARD = []


def task(id, name, spec, cases, reference, setup=""):
    HARD.append({"id": id, "name": name, "spec": spec, "cases": cases, "reference": reference, "setup": setup, "tier": "hard"})


task("cron", "cron_next", """Write `cron_next(expr: str, after: str) -> str`: the first time STRICTLY after `after` that matches a cron expression.
Times are "YYYY-MM-DD HH:MM" strings (no time zones, no seconds).
- expr has exactly 5 space-separated fields: minute (0-59), hour (0-23), day of month (1-31), month (1-12),
  day of week (0-7, where both 0 and 7 mean Sunday, 1 = Monday).
- A field is "*" or a comma-separated list of items. An item is a number, a range "a-b" (a <= b), "*/s", "a-b/s"
  (every s-th value from a to b) or "a/s" (every s-th value from a to the field maximum). s must be >= 1.
- Day rule (as in Vixie cron): if BOTH day of month and day of week are restricted (neither is "*"), a day matches
  if EITHER matches. If one of them is "*", only the other one decides.
- Raise ValueError for a wrong number of fields, values out of range, a > b, a step of 0, or anything unparsable.
- Raise ValueError if nothing matches within the next 5 years (e.g. "0 0 30 2 *").""",
     ['cron_next("*/15 * * * *", "2026-10-16 10:07")', 'cron_next("0 9 * * 1-5", "2026-10-16 09:00")',
      'cron_next("30 2 1 * *", "2026-10-16 00:00")', 'cron_next("0 0 13 * 5", "2026-10-16 00:00")',
      'cron_next("0 0 29 2 *", "2026-03-01 00:00")', 'cron_next("0 0 30 2 *", "2026-01-01 00:00")',
      'cron_next("0 0 * * 7", "2026-10-16 12:00")', 'cron_next("5-10/2 * * * *", "2026-10-16 10:09")',
      'cron_next("60 * * * *", "2026-10-16 10:00")', 'cron_next("* * *", "2026-10-16 10:00")',
      'cron_next("*/0 * * * *", "2026-10-16 10:00")', 'cron_next("10-5 * * * *", "2026-10-16 10:00")',
      'cron_next("59 23 31 12 *", "2026-12-31 23:59")', 'cron_next("1/20 * * * *", "2026-10-16 10:41")',
      'cron_next("0 12 * 12 0", "2026-12-25 12:00")', 'cron_next("0 0 1,15 * 3", "2026-10-16 00:00")',
      'cron_next("15 14 * * *", "2026-10-16 14:14")', 'cron_next("0 0 31 * *", "2026-04-15 00:00")'],
     '''import datetime as _dt
def _field(s, lo, hi):
    vals = set()
    for item in s.split(","):
        base, slash, step = item.partition("/")
        if slash and (not step.isdigit() or int(step) < 1): raise ValueError(item)
        st = int(step) if slash else 1
        if base == "*": a, b = lo, hi
        elif "-" in base:
            x, y = base.split("-", 1)
            if not (x.isdigit() and y.isdigit()): raise ValueError(item)
            a, b = int(x), int(y)
        elif base.isdigit():
            a = int(base); b = hi if slash else a
        else: raise ValueError(item)
        if a < lo or b > hi or a > b: raise ValueError(item)
        vals.update(range(a, b + 1, st))
    return vals
def cron_next(expr, after):
    f = expr.split(" ")
    if len(f) != 5 or any(not x for x in f): raise ValueError(expr)
    mi, hr, dom, mon, dow = (_field(f[0], 0, 59), _field(f[1], 0, 23), _field(f[2], 1, 31),
                             _field(f[3], 1, 12), _field(f[4], 0, 7))
    dow = {d % 7 for d in dow}
    t = _dt.datetime.strptime(after, "%Y-%m-%d %H:%M") + _dt.timedelta(minutes=1)
    end = t + _dt.timedelta(days=366 * 5)
    def day_ok(d):
        a, b = d.day in dom, (d.isoweekday() % 7) in dow
        if f[2] != "*" and f[4] != "*": return a or b
        if f[2] != "*": return a
        if f[4] != "*": return b
        return True
    while t < end:
        if t.month in mon and day_ok(t):
            for h in sorted(hr):
                if h < t.hour: continue
                for m in sorted(mi):
                    if h == t.hour and m < t.minute: continue
                    return t.replace(hour=h, minute=m).strftime("%Y-%m-%d %H:%M")
        t = (t + _dt.timedelta(days=1)).replace(hour=0, minute=0)
    raise ValueError("no match within 5 years")
''')

task("rolldate", "roll_date", """Write `roll_date(date: str, months: int, convention: str, holidays: list) -> str` for "YYYY-MM-DD" dates.
Step 1, add months (months may be negative or 0):
- Keep the day of month, clamped to the target month's length (2026-01-31 + 1 month = 2026-02-28).
- End-of-month rule: if the start date is the LAST day of its month, the result is the last day of the target month
  (2026-02-28 + 1 month = 2026-03-31; 2026-04-30 + 1 month = 2026-05-31).
Step 2, if the result is not a business day (Saturday, Sunday, or in holidays), adjust by convention:
- "none": no adjustment. "following": the next business day. "preceding": the previous business day.
- "modified_following": the next business day, unless that is in a different month; then the previous business day.
- Any other convention raises ValueError.""",
     ['roll_date("2026-01-31", 1, "none", [])', 'roll_date("2026-02-28", 1, "none", [])', 'roll_date("2026-04-30", 1, "none", [])',
      'roll_date("2024-01-31", 1, "none", [])', 'roll_date("2026-01-15", 0, "following", [])',
      'roll_date("2026-09-15", 1, "following", [])', 'roll_date("2026-09-15", 1, "preceding", [])',
      'roll_date("2026-04-30", 1, "modified_following", [])', 'roll_date("2026-04-30", 1, "following", [])',
      'roll_date("2026-03-31", -1, "none", [])', 'roll_date("2026-03-30", -1, "none", [])',
      'roll_date("2026-11-02", 1, "following", ["2026-12-02"])', 'roll_date("2026-06-30", 2, "modified_following", ["2026-08-31"])',
      'roll_date("2026-01-10", 1, "sideways", [])', 'roll_date("2026-05-01", 0, "modified_following", [])',
      'roll_date("2025-12-31", 2, "preceding", [])', 'roll_date("2026-02-28", -12, "none", [])'],
     '''import datetime as _dt, calendar as _cal
def roll_date(date, months, convention, holidays):
    if convention not in ("none", "following", "preceding", "modified_following"): raise ValueError(convention)
    d = _dt.date.fromisoformat(date)
    last = _cal.monthrange(d.year, d.month)[1] == d.day
    y, m = divmod(d.year * 12 + (d.month - 1) + months, 12); m += 1
    n = _cal.monthrange(y, m)[1]
    r = _dt.date(y, m, n if last else min(d.day, n))
    hol = {_dt.date.fromisoformat(h) for h in holidays}
    bad = lambda x: x.weekday() >= 5 or x in hol
    def move(x, step):
        while bad(x): x += _dt.timedelta(days=step)
        return x
    if convention == "none" or not bad(r): return r.isoformat()
    if convention == "following": return move(r, 1).isoformat()
    if convention == "preceding": return move(r, -1).isoformat()
    f = move(r, 1)
    return (f if f.month == r.month else move(r, -1)).isoformat()
''')

task("shsplit", "shell_split", """Write `shell_split(line: str, env: dict) -> list` that splits a command line into words (no shlex).
- Unquoted spaces and tabs separate words; runs of them count once; leading/trailing ones are ignored.
- Adjacent quoted and unquoted parts join into one word (a"b"'c' is one word abc). A standalone "" or '' is an empty word.
- Single quotes: everything up to the next single quote is literal (no escapes, no expansion).
- Double quotes: literal, except a backslash escapes only ", \\ and $ (a backslash before any other character is kept
  together with that character), and variables are expanded.
- Outside quotes: a backslash makes the next character literal (including space, quote and $); a trailing backslash raises ValueError.
- Variables: $NAME or ${NAME}, where NAME matches [A-Za-z_][A-Za-z0-9_]*; a missing variable expands to "".
  A "$" not followed by a valid name or "{" is a literal "$". "${" without a closing "}" raises ValueError.
  Expanded values are never split into words.
- An unterminated quote raises ValueError.""",
     ['shell_split("ls  -la   /tmp ", {})', 'shell_split(\'echo "hello world"\', {})', 'shell_split("echo \'$HOME\'", {"HOME": "/h"})',
      'shell_split(\'echo "$HOME/x"\', {"HOME": "/h"})', 'shell_split("echo ${A}b $Ab", {"A": "1", "Ab": "2"})',
      'shell_split(\'a "" b\', {})', 'shell_split(\'a"b"\\\'c\\\'\', {})', 'shell_split(\'echo "a\\\\"b"\', {})',
      'shell_split(\'echo "a\\\\nb"\', {})', 'shell_split("echo a\\\\ b", {})', 'shell_split("echo \\\\$X", {"X": "1"})',
      'shell_split("echo $MISSING.", {})', 'shell_split("echo $ 5$", {})', 'shell_split("echo ${X", {"X": "1"})',
      'shell_split(\'echo "abc\', {})', 'shell_split("echo abc\\\\", {})', 'shell_split("echo $X", {"X": "a b"})',
      'shell_split("\\tx\\ty\\t", {})', 'shell_split("", {})', 'shell_split("echo $1abc", {})'],
     '''import re as _re
_NAME = _re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
def shell_split(line, env):
    words, cur, have, i, n = [], [], False, 0, len(line)
    def var(i):
        if i + 1 < n and line[i + 1] == "{":
            j = line.find("}", i + 2)
            if j == -1: raise ValueError("unclosed ${")
            return env.get(line[i + 2:j], ""), j + 1
        m = _NAME.match(line, i + 1)
        if m: return env.get(m.group(), ""), m.end()
        return "$", i + 1
    while i < n:
        c = line[i]
        if c in " \\t":
            if have: words.append("".join(cur)); cur, have = [], False
            i += 1
        elif c == "'":
            j = line.find("'", i + 1)
            if j == -1: raise ValueError("unterminated '")
            cur.append(line[i + 1:j]); have = True; i = j + 1
        elif c == '"':
            have = True; i += 1
            while True:
                if i >= n: raise ValueError('unterminated "')
                c = line[i]
                if c == '"': i += 1; break
                if c == "\\\\" and i + 1 < n and line[i + 1] in '"\\\\$': cur.append(line[i + 1]); i += 2
                elif c == "$": v, i = var(i); cur.append(v)
                else: cur.append(c); i += 1
        elif c == "\\\\":
            if i + 1 >= n: raise ValueError("trailing backslash")
            cur.append(line[i + 1]); have = True; i += 2
        elif c == "$":
            v, i = var(i); cur.append(v); have = True
        else:
            cur.append(c); have = True; i += 1
    if have: words.append("".join(cur))
    return words
''')

task("template", "render", """Write `render(template: str, ctx: dict) -> str`, a small template engine (no external libraries).
- {{ name }} inserts the value, HTML-escaped (& < > " ' become &amp; &lt; &gt; &quot; &#39;). {{{ name }}} inserts it raw.
  Spaces inside the braces are allowed. A missing name or None inserts "". Other values are converted with str().
- Dotted names (a.b.c) look up nested dicts; any missing step gives "".
- {{#if name}} ... {{else}} ... {{/if}}: the first part if the value is truthy (Python truthiness), else the optional second part.
- {{#each name}} ... {{/each}}: repeat the body for each item of a list (a missing name repeats zero times). Inside the body,
  {{.}} is the current item, {{@index}} its 0-based position, and other names are looked up first in the item
  (when it is a dict) and then in the enclosing context. Blocks can be nested.
- An unclosed block, a stray {{/if}} or {{/each}}, a mismatched closing tag, or a stray {{else}} raises ValueError.""",
     ['render("Hi {{name}}!", {"name": "Ann"})', 'render("{{ x }}", {"x": "<b>&\\"\'"})', 'render("{{{x}}}", {"x": "<b>"})',
      'render("[{{missing}}]", {})', 'render("{{a.b.c}}", {"a": {"b": {"c": 5}}})', 'render("{{a.z.c}}", {"a": {"b": 1}})',
      'render("{{#if ok}}yes{{else}}no{{/if}}", {"ok": 0})', 'render("{{#if ok}}yes{{/if}}", {"ok": [1]})',
      'render("{{#each xs}}{{@index}}={{.}};{{/each}}", {"xs": ["a", "b"]})',
      'render("{{#each people}}{{name}}@{{team}} {{/each}}", {"team": "T", "people": [{"name": "A"}, {"name": "B", "team": "U"}]})',
      'render("{{#each xs}}{{#if .}}+{{else}}-{{/if}}{{/each}}", {"xs": [1, 0, 2]})', 'render("{{#each nope}}x{{/each}}.", {})',
      'render("{{#if a}}x", {"a": 1})', 'render("x{{/if}}", {})', 'render("{{#if a}}x{{/each}}", {"a": 1})', 'render("{{else}}", {})',
      'render("{{n}}", {"n": None})', 'render("{{#each g}}[{{#each .}}{{.}}{{/each}}]{{/each}}", {"g": [[1, 2], [3]]})',
      'render("{{x}}", {"x": 3.5})'],
     '''import html as _html, re as _re
_TAG = _re.compile(r"\\{\\{\\{\\s*([^{}]*?)\\s*\\}\\}\\}|\\{\\{\\s*([^{}]*?)\\s*\\}\\}")
def _parse(t):
    root = []; stack = [["root", root, None]]; pos = 0
    for m in _TAG.finditer(t):
        if m.start() > pos: stack[-1][1].append(("text", t[pos:m.start()]))
        pos = m.end()
        if m.group(1) is not None: stack[-1][1].append(("raw", m.group(1))); continue
        tag = m.group(2)
        if tag.startswith("#if ") or tag.startswith("#each "):
            kind, name = tag[1:].split(None, 1)
            node = [kind, name.strip(), [], None]; stack[-1][1].append(node); stack.append([kind, node[2], node])
        elif tag == "else":
            top = stack[-1]
            if top[0] != "if" or top[2][3] is not None: raise ValueError("stray else")
            top[2][3] = []; top[1] = top[2][3]
        elif tag in ("/if", "/each"):
            if stack[-1][0] != tag[1:]: raise ValueError("mismatched " + tag)
            stack.pop()
        else: stack[-1][1].append(("var", tag))
    if len(stack) != 1: raise ValueError("unclosed block")
    if pos < len(t): root.append(("text", t[pos:]))
    return root
def _lookup(name, scopes):
    if name == ".": return scopes[-1][0]
    if name == "@index": return scopes[-1][1]
    parts = name.split(".")
    for item, _ in reversed(scopes):
        if isinstance(item, dict) and parts[0] in item:
            v = item
            for p in parts:
                if not isinstance(v, dict) or p not in v: return None
                v = v[p]
            return v
    return None
def _run(nodes, scopes):
    out = []
    for n in nodes:
        k = n[0]
        if k == "text": out.append(n[1])
        elif k in ("var", "raw"):
            v = _lookup(n[1], scopes); s = "" if v is None else str(v)
            out.append(_html.escape(s, quote=True).replace("&#x27;", "&#39;") if k == "var" else s)
        elif k == "if":
            out.append(_run(n[2] if _lookup(n[1], scopes) else (n[3] or []), scopes))
        elif k == "each":
            for i, it in enumerate(_lookup(n[1], scopes) or []):
                out.append(_run(n[2], scopes + [(it, i)]))
    return "".join(out)
def render(template, ctx):
    return _run(_parse(template), [(ctx, None)])
''')
