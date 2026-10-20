"""Coding tasks for the task-completion test, written for this project.

Each task has a spec (what the model sees), test expressions (hidden; only failures are shown back,
like CI output) and a reference solution used to produce and check the expected values. The specs
state every rule the tests check, so a careful first attempt can pass; edge cases make that hard.
"""

TASKS = []


def task(id, name, spec, cases, reference, setup=""):
    TASKS.append({"id": id, "name": name, "spec": spec, "cases": cases, "reference": reference, "setup": setup})


task("duration", "parse_duration", """Write `parse_duration(s: str) -> int` returning a number of seconds.
- A plain non-negative integer string (e.g. "45") means seconds.
- Otherwise the string is one or more parts `<digits><unit>` with units d (86400 s), h, m, s.
- Units must appear in the order d, h, m, s, each at most once. Lowercase only, no spaces, no signs.
- Raise ValueError for anything else: empty string, wrong order, repeated unit, unknown or uppercase unit,
  spaces, a sign, a unit without digits, or digits without a unit after other parts (e.g. "1h30").""",
     ['parse_duration("1h30m")', 'parse_duration("45")', 'parse_duration("2d")', 'parse_duration("1d1s")',
      'parse_duration("0s")', 'parse_duration("007m")', 'parse_duration("1d2h3m4s")', 'parse_duration("")',
      'parse_duration("1m1h")', 'parse_duration("1h1h")', 'parse_duration("1x")', 'parse_duration("-5")',
      'parse_duration("1 h")', 'parse_duration("h")', 'parse_duration("1H")', 'parse_duration("1h30")',
      'parse_duration("+5")', 'parse_duration("1h-2m")'],
     '''import re
def parse_duration(s):
    if re.fullmatch(r"[0-9]+", s): return int(s)
    m = re.fullmatch(r"(?:([0-9]+)d)?(?:([0-9]+)h)?(?:([0-9]+)m)?(?:([0-9]+)s)?", s)
    if not s or not m: raise ValueError(s)
    return sum(int(v) * k for v, k in zip(m.groups(), (86400, 3600, 60, 1)) if v is not None)
''')

task("ranges", "merge_ranges", """Write `merge_ranges(ranges: list) -> list` for half-open integer ranges [start, end).
- Input: a list of (start, end) pairs, in any order. Output: a sorted list of (start, end) tuples.
- Merge ranges that overlap AND ranges that only touch (end of one == start of the next).
- Drop empty ranges (start == end); they never join neighbours on their own.
- Raise ValueError if any range has start > end.""",
     ['merge_ranges([(1, 3), (2, 5)])', 'merge_ranges([(1, 3), (3, 5)])', 'merge_ranges([(5, 7), (1, 2)])',
      'merge_ranges([])', 'merge_ranges([(2, 2)])', 'merge_ranges([(1, 2), (2, 2), (3, 4)])',
      'merge_ranges([(1, 10), (2, 3), (4, 5)])', 'merge_ranges([(3, 1)])', 'merge_ranges([(0, 1), (1, 2), (2, 3)])',
      'merge_ranges([(-5, -1), (-1, 0), (4, 6), (6, 6)])', 'merge_ranges([[1, 4], [0, 1]])'],
     '''def merge_ranges(ranges):
    rs = []
    for a, b in ranges:
        if a > b: raise ValueError((a, b))
        if a < b: rs.append((a, b))
    rs.sort(); out = []
    for a, b in rs:
        if out and a <= out[-1][1]: out[-1] = (out[-1][0], max(out[-1][1], b))
        else: out.append((a, b))
    return out
''')

task("semver", "semver_compare", """Write `semver_compare(a: str, b: str) -> int` returning -1, 0 or 1 by Semantic Versioning 2.0.0 precedence.
- MAJOR.MINOR.PATCH compare numerically.
- A version with a pre-release (after "-") is lower than the same version without one.
- Pre-release identifiers (dot-separated) compare left to right: numeric identifiers numerically,
  alphanumeric ones by ASCII order, numeric always lower than alphanumeric; if all equal, the shorter list is lower.
- Build metadata (after "+") is ignored. Inputs are valid versions.""",
     ['semver_compare("1.0.0-alpha", "1.0.0-alpha.1")', 'semver_compare("1.0.0-alpha.1", "1.0.0-alpha.beta")',
      'semver_compare("1.0.0-alpha.beta", "1.0.0-beta")', 'semver_compare("1.0.0-beta", "1.0.0-beta.2")',
      'semver_compare("1.0.0-beta.2", "1.0.0-beta.11")', 'semver_compare("1.0.0-beta.11", "1.0.0-rc.1")',
      'semver_compare("1.0.0-rc.1", "1.0.0")', 'semver_compare("1.0.0+build.5", "1.0.0")',
      'semver_compare("1.10.0", "1.9.0")', 'semver_compare("2.0.0", "10.0.0")', 'semver_compare("1.0.0-1", "1.0.0-a")',
      'semver_compare("1.0.0-rc.1+x", "1.0.0-rc.1+y")', 'semver_compare("1.0.0", "1.0.0-rc.1")',
      'semver_compare("1.0.0-Z", "1.0.0-a")', 'semver_compare("1.0.0-a-b", "1.0.0-a")'],
     '''def semver_compare(a, b):
    def key(v):
        v = v.split("+")[0]
        core, _, pre = v.partition("-")
        nums = tuple(int(x) for x in core.split("."))
        if not pre: return (nums, 1, ())
        ids = tuple((0, int(x), "") if x.isdigit() else (1, 0, x) for x in pre.split("."))
        return (nums, 0, ids)
    ka, kb = key(a), key(b)
    return (ka > kb) - (ka < kb)
''')

task("csv", "parse_csv_line", """Write `parse_csv_line(line: str) -> list` that splits one CSV line into string fields.
- Fields are separated by commas. Whitespace is kept exactly as written.
- A field may be wrapped in double quotes; inside it, commas are literal and "" means one literal quote.
- An empty line returns [""]; a trailing comma means a final empty field.
- Raise ValueError for: a quote inside an unquoted field, an unterminated quoted field,
  or any character other than a comma right after a closing quote.""",
     ['parse_csv_line("a,b,c")', 'parse_csv_line("")', 'parse_csv_line("a,")', 'parse_csv_line(\'"a,b",c\')',
      'parse_csv_line(\'"say ""hi""",x\')', 'parse_csv_line(" a , b ")', 'parse_csv_line(\'""\')',
      'parse_csv_line(\'a,"",b\')', 'parse_csv_line(\'ab"c\')', 'parse_csv_line(\'"abc\')', 'parse_csv_line(\'"a"b\')',
      'parse_csv_line(\'"a" ,b\')', 'parse_csv_line(",,")', 'parse_csv_line(\'"x,""y"""\')'],
     '''def parse_csv_line(line):
    out, i, n = [], 0, len(line)
    while True:
        if i < n and line[i] == '"':
            i += 1; buf = []
            while True:
                if i >= n: raise ValueError("unterminated")
                if line[i] == '"':
                    if i + 1 < n and line[i + 1] == '"': buf.append('"'); i += 2; continue
                    i += 1; break
                buf.append(line[i]); i += 1
            out.append("".join(buf))
            if i < n and line[i] != ",": raise ValueError("after quote")
        else:
            j = line.find(",", i); j = n if j == -1 else j
            f = line[i:j]
            if '"' in f: raise ValueError("quote in field")
            out.append(f); i = j
        if i >= n: return out
        i += 1
''')

task("expr", "eval_expr", """Write `eval_expr(s: str) -> int` that evaluates an integer arithmetic expression without using eval/exec.
- Supports integers, + - * /, parentheses, unary + and - (which can repeat, e.g. "--3"), and spaces.
- Usual precedence; binary operators are left-associative.
- "/" is integer division that truncates toward zero (like C), e.g. 7 / -2 == -3.
- Division by zero raises ZeroDivisionError. Malformed input raises ValueError.""",
     ['eval_expr("7/-2")', 'eval_expr("-(3+4)*2")', 'eval_expr("2-3-4")', 'eval_expr("8/2/2")', 'eval_expr("--3")',
      'eval_expr("2*-3")', 'eval_expr(" 1 + 2 * 3 ")', 'eval_expr("1/0")', 'eval_expr("-7/2")', 'eval_expr("(1+2")',
      'eval_expr("2+")', 'eval_expr("10-(2-3)*4")', 'eval_expr("-(-(-1))")', 'eval_expr("3 4")', 'eval_expr("100/7*7")'],
     '''def eval_expr(s):
    toks = []; i = 0
    while i < len(s):
        c = s[i]
        if c.isspace(): i += 1; continue
        if c.isdigit():
            j = i
            while j < len(s) and s[j].isdigit(): j += 1
            toks.append(int(s[i:j])); i = j; continue
        if c in "+-*/()": toks.append(c); i += 1; continue
        raise ValueError(c)
    pos = 0
    def peek(): return toks[pos] if pos < len(toks) else None
    def take():
        nonlocal pos; t = peek(); pos += 1; return t
    def expr():
        v = term()
        while peek() in ("+", "-"):
            v = v + term() if take() == "+" else v - term()
        return v
    def term():
        v = unary()
        while peek() in ("*", "/"):
            op = take(); r = unary()
            if op == "*": v *= r
            else:
                if r == 0: raise ZeroDivisionError
                q = abs(v) // abs(r); v = q if (v >= 0) == (r >= 0) else -q
        return v
    def unary():
        if peek() in ("+", "-"): return unary() if take() == "+" else -unary()
        t = take()
        if t == "(":
            v = expr()
            if take() != ")": raise ValueError("paren")
            return v
        if isinstance(t, int): return t
        raise ValueError(t)
    v = expr()
    if pos != len(toks): raise ValueError("trailing")
    return v
''')

task("justify", "justify", """Write `justify(words: list, width: int) -> list` that fully justifies text.
- Pack as many words as fit on each line (single spaces between words count toward width).
- Each line except the last: spread extra spaces between words as evenly as possible; when they don't
  divide evenly, the gaps on the left get one more space than those on the right.
- A line with a single word, and the last line, are left-aligned (single spaces) and padded with spaces to width.
- Every returned line has exactly `width` characters. Words are never longer than width. Empty input returns [].""",
     ['justify(["This", "is", "an", "example", "of", "text", "justification."], 16)',
      'justify(["What", "must", "be", "acknowledgment", "shall", "be"], 16)',
      'justify(["a", "b", "c", "d", "e"], 3)', 'justify([], 5)', 'justify(["one"], 5)',
      'justify(["Science", "is", "what", "we", "understand", "well", "enough", "to", "explain"], 20)',
      'justify(["ab", "cd", "ef", "gh"], 9)'],
     '''def justify(words, width):
    lines, cur = [], []
    for w in words:
        if cur and len(" ".join(cur + [w])) > width: lines.append(cur); cur = []
        cur.append(w)
    if cur: lines.append(cur)
    out = []
    for k, ln in enumerate(lines):
        if k == len(lines) - 1 or len(ln) == 1: out.append(" ".join(ln).ljust(width)); continue
        gaps = len(ln) - 1; spaces = width - sum(map(len, ln)); q, r = divmod(spaces, gaps)
        out.append("".join(w + (" " * (q + (1 if i < r else 0)) if i < gaps else "") for i, w in enumerate(ln)))
    return out
''')

task("topo", "topo_order", """Write `topo_order(deps: dict) -> list` returning a build order.
- deps maps each node (a string) to a list of its prerequisites. Nodes that appear only as prerequisites are included too.
- Every node must come after all its prerequisites.
- Whenever several nodes are available, take the alphabetically smallest first (so the answer is unique).
- Raise ValueError if there is a cycle (including a node that depends on itself).""",
     ['topo_order({"b": ["a"], "c": ["a"]})', 'topo_order({})', 'topo_order({"x": []})',
      'topo_order({"a": ["b"], "b": ["a"]})', 'topo_order({"a": ["a"]})',
      'topo_order({"d": ["b", "c"], "b": ["a"], "c": ["a"], "e": []})',
      'topo_order({"z": ["y"], "y": [], "a": ["z"]})', 'topo_order({"m": ["k", "k"], "k": []})'],
     '''import heapq
def topo_order(deps):
    nodes = set(deps) | {p for ps in deps.values() for p in ps}
    pre = {n: set(deps.get(n, [])) for n in nodes}
    users = {n: set() for n in nodes}
    for n, ps in pre.items():
        for p in ps: users[p].add(n)
    h = [n for n in nodes if not pre[n]]; heapq.heapify(h); out = []
    while h:
        n = heapq.heappop(h); out.append(n)
        for u in sorted(users[n]):
            pre[u].discard(n)
            if not pre[u]: heapq.heappush(h, u)
    if len(out) != len(nodes): raise ValueError("cycle")
    return out
''')

task("glob", "glob_match", """Write `glob_match(pattern: str, name: str) -> bool` (whole-string match, without fnmatch/re).
- "*" matches any sequence of characters, including an empty one. "?" matches exactly one character.
- "[abc]" matches one listed character; ranges like "[a-z]" are allowed; "[!abc]" matches one character NOT listed.
- A "[" with no closing "]" later in the pattern is a literal "[". Everything else matches itself (case-sensitive).""",
     ['glob_match("*.py", "main.py")', 'glob_match("*.py", "main.pyc")', 'glob_match("a?c", "abc")', 'glob_match("a?c", "ac")',
      'glob_match("[a-c]x", "bx")', 'glob_match("[!a-c]x", "bx")', 'glob_match("[!a-c]x", "dx")', 'glob_match("*", "")',
      'glob_match("a*b*c", "aXbYbZc")', 'glob_match("a[", "a[")', 'glob_match("[ab", "[ab")', 'glob_match("**a", "bba")',
      'glob_match("A*", "abc")', 'glob_match("*a*a*a*a*b", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")', 'glob_match("[xyz]", "")'],
     '''def glob_match(pattern, name):
    from functools import lru_cache
    def cls(i):
        j = pattern.find("]", i + 2 if i + 1 < len(pattern) and pattern[i + 1] == "!" else i + 1)
        return j
    @lru_cache(None)
    def m(i, k):
        if i == len(pattern): return k == len(name)
        c = pattern[i]
        if c == "*": return m(i + 1, k) or (k < len(name) and m(i, k + 1))
        if k == len(name): return False
        if c == "?": return m(i + 1, k + 1)
        if c == "[":
            j = cls(i)
            if j != -1:
                body = pattern[i + 1:j]; neg = body.startswith("!")
                if neg: body = body[1:]
                hit, t = False, 0
                while t < len(body):
                    if t + 2 < len(body) and body[t + 1] == "-":
                        hit |= body[t] <= name[k] <= body[t + 2]; t += 3
                    else:
                        hit |= body[t] == name[k]; t += 1
                return hit != neg and m(j + 1, k + 1)
        return c == name[k] and m(i + 1, k + 1)
    return m(0, 0)
''')

task("words", "to_words", """Write `to_words(n: int) -> str` giving the English words for 0 <= n < 10**12.
- Lowercase, single spaces, hyphen for 21-99 compounds ("twenty-one"), no "and" ("one hundred five").
- Scale words: thousand, million, billion; skip zero groups ("one million one").
- Raise ValueError for negative n or n >= 10**12.""",
     ['to_words(0)', 'to_words(7)', 'to_words(13)', 'to_words(21)', 'to_words(40)', 'to_words(105)', 'to_words(1000)',
      'to_words(1001)', 'to_words(1000000)', 'to_words(1000001)', 'to_words(12345)', 'to_words(999999999999)',
      'to_words(-1)', 'to_words(10**12)', 'to_words(2000010)', 'to_words(110)'],
     '''ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
def to_words(n):
    if n < 0 or n >= 10**12: raise ValueError(n)
    if n == 0: return "zero"
    def three(x):
        w = []
        if x >= 100: w += [ONES[x // 100], "hundred"]; x %= 100
        if x >= 20: w.append(TENS[x // 10] + ("-" + ONES[x % 10] if x % 10 else ""))
        elif x: w.append(ONES[x])
        return w
    out = []
    for scale, word in ((10**9, "billion"), (10**6, "million"), (10**3, "thousand"), (1, "")):
        g = n // scale % 1000
        if g: out += three(g) + ([word] if word else [])
    return " ".join(out)
''')

task("busdays", "business_days", """Write `business_days(start: str, end: str, holidays: list) -> int` for dates written "YYYY-MM-DD".
- Count Monday-Friday dates d with start <= d < end (end excluded) that are not in holidays.
- Holidays that fall on weekends or outside the range change nothing; duplicate holidays count once.
- If end is before start, return the negative of business_days(end, start, holidays).""",
     ['business_days("2026-10-12", "2026-10-19", [])', 'business_days("2026-10-12", "2026-10-12", [])',
      'business_days("2026-10-17", "2026-10-19", [])', 'business_days("2026-10-12", "2026-10-19", ["2026-10-14", "2026-10-14"])',
      'business_days("2026-10-12", "2026-10-19", ["2026-10-18"])', 'business_days("2026-10-19", "2026-10-12", [])',
      'business_days("2024-02-26", "2024-03-04", ["2024-02-29"])', 'business_days("2026-12-31", "2027-01-02", ["2027-01-01"])',
      'business_days("2026-10-16", "2026-10-17", [])'],
     '''import datetime as _dt
def business_days(start, end, holidays):
    a, b = _dt.date.fromisoformat(start), _dt.date.fromisoformat(end)
    if b < a: return -business_days(end, start, holidays)
    hol = {_dt.date.fromisoformat(h) for h in holidays}; n = 0
    while a < b:
        if a.weekday() < 5 and a not in hol: n += 1
        a += _dt.timedelta(days=1)
    return n
''')

task("pointer", "json_pointer_get", """Write `json_pointer_get(doc, pointer: str)` implementing JSON Pointer (RFC 6901) lookup.
- "" returns the whole document. Any other pointer must start with "/", otherwise raise ValueError.
- Split the rest on "/"; in each token replace "~1" with "/" and then "~0" with "~" (in that order).
- On a dict, the token is a key. On a list, the token must be a decimal index without leading zeros
  ("0" is fine, "01" is not) and within range; "-" is not a valid index.
- Raise KeyError for a missing key, an invalid or out-of-range index, or stepping into a non-container.""",
     ['json_pointer_get({"a": {"b": 1}}, "/a/b")', 'json_pointer_get({"a": 1}, "")', 'json_pointer_get({"a/b": 2}, "/a~1b")',
      'json_pointer_get({"m~n": 3}, "/m~0n")', 'json_pointer_get({"~1": 4}, "/~01")', 'json_pointer_get({"a": [10, 20]}, "/a/1")',
      'json_pointer_get({"a": [10, 20]}, "/a/01")', 'json_pointer_get({"a": [10]}, "/a/-")', 'json_pointer_get({"a": [10]}, "/a/5")',
      'json_pointer_get({"a": 1}, "a")', 'json_pointer_get({"": 5}, "/")', 'json_pointer_get({"a": 1}, "/a/b")',
      'json_pointer_get({"a": 1}, "/b")', 'json_pointer_get([[1, 2]], "/0/0")'],
     '''def json_pointer_get(doc, pointer):
    if pointer == "": return doc
    if not pointer.startswith("/"): raise ValueError(pointer)
    cur = doc
    for tok in pointer[1:].split("/"):
        tok = tok.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, dict):
            if tok not in cur: raise KeyError(tok)
            cur = cur[tok]
        elif isinstance(cur, list):
            if not tok.isdigit() or (len(tok) > 1 and tok[0] == "0") or int(tok) >= len(cur): raise KeyError(tok)
            cur = cur[int(tok)]
        else: raise KeyError(tok)
    return cur
''')

task("limiter", "RateLimiter", """Write a class `RateLimiter(limit: int, window: float)` with a method `allow(t: float) -> bool`.
- allow(t) returns True if fewer than `limit` requests were ALLOWED in the half-open interval (t - window, t],
  and then records this request as allowed; otherwise it returns False.
- Rejected requests are not recorded and never count toward the limit.
- Calls arrive with non-decreasing t. limit may be 0 (nothing is ever allowed).""",
     ['run_limiter(3, 10, [0, 1, 2, 3, 10, 10.5, 11])', 'run_limiter(1, 5, [0, 4.9, 5, 5])', 'run_limiter(0, 5, [0, 1])',
      'run_limiter(2, 1, [0, 0, 0, 1, 1, 1.5, 2])', 'run_limiter(2, 10, [0, 1, 2, 3, 4, 10, 11, 11])'],
     '''from collections import deque
class RateLimiter:
    def __init__(self, limit, window): self.limit, self.window, self.q = limit, window, deque()
    def allow(self, t):
        while self.q and self.q[0] <= t - self.window: self.q.popleft()
        if len(self.q) < self.limit: self.q.append(t); return True
        return False
''', setup="def run_limiter(limit, window, ts):\n    r = RateLimiter(limit, window)\n    return [r.allow(t) for t in ts]\n")

task("roman", "roman_to_int", """Write `roman_to_int(s: str) -> int` that accepts only standard (canonical) Roman numerals from 1 to 3999.
- Uppercase I V X L C D M, with the usual subtractive forms IV IX XL XC CD CM only.
- Raise ValueError for anything that is not the canonical spelling of a number in range,
  e.g. "IIII", "IC", "VX", "MMMM", "VV", "DCD", lowercase, or an empty string.""",
     ['roman_to_int("MCMXCIV")', 'roman_to_int("XLIX")', 'roman_to_int("MMMCMXCIX")', 'roman_to_int("CM")', 'roman_to_int("I")',
      'roman_to_int("IIII")', 'roman_to_int("IL")', 'roman_to_int("IC")', 'roman_to_int("VV")', 'roman_to_int("DCD")',
      'roman_to_int("MMMM")', 'roman_to_int("")', 'roman_to_int("iv")', 'roman_to_int("XCX")', 'roman_to_int("CDXLIV")',
      'roman_to_int("IXI")', 'roman_to_int("LXL")'],
     '''def _to_roman(n):
    out = ""
    for v, r in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                 (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= v: out += r; n -= v
    return out
_R = {_to_roman(i): i for i in range(1, 4000)}
def roman_to_int(s):
    if s not in _R: raise ValueError(s)
    return _R[s]
''')

task("ledger", "apply_ledger", """Write `apply_ledger(balance: int, transactions: list) -> tuple` for a bank account in cents.
- Each transaction is ("dep", amount) or ("wd", amount); amount must be a positive int, otherwise raise ValueError
  (also for an unknown kind). Process them in order.
- A deposit adds the amount. A withdrawal that keeps the balance >= 0 subtracts it.
- A withdrawal that would make the balance negative is declined (balance unchanged) and a 2500-cent fee is charged
  instead. The fee is charged even if it makes the balance negative, but at most 3 fees are charged per call;
  later declines are still declined, just free.
- Return (final_balance, number_of_declined_withdrawals).""",
     ['apply_ledger(10000, [("wd", 3000), ("dep", 500)])', 'apply_ledger(1000, [("wd", 2000)])',
      'apply_ledger(0, [("wd", 1), ("wd", 1), ("wd", 1), ("wd", 1)])', 'apply_ledger(5000, [("wd", 5000)])',
      'apply_ledger(100, [("dep", 0)])', 'apply_ledger(100, [("xfer", 5)])', 'apply_ledger(100, [("wd", -5)])',
      'apply_ledger(3000, [("wd", 4000), ("wd", 600), ("dep", 100)])', 'apply_ledger(0, [])'],
     '''def apply_ledger(balance, transactions):
    fees = declined = 0
    for kind, amt in transactions:
        if kind not in ("dep", "wd") or not isinstance(amt, int) or amt <= 0: raise ValueError((kind, amt))
        if kind == "dep": balance += amt
        elif balance - amt >= 0: balance -= amt
        else:
            declined += 1
            if fees < 3: balance -= 2500; fees += 1
    return (balance, declined)
''')

task("compress", "compress_ranges", """Write `compress_ranges(nums: list) -> str` that summarises integers.
- Input may be unsorted and contain duplicates (duplicates count once). Output is sorted, comma-separated, no spaces.
- A run of 3 or more consecutive integers is written "a-b"; a run of 2 is written as two items "a,b".
- Negative numbers keep their sign, e.g. a run from -3 to -1 is "-3--1". Empty input returns "".""",
     ['compress_ranges([1, 2, 3, 5, 7, 8])', 'compress_ranges([])', 'compress_ranges([4])', 'compress_ranges([3, 1, 2, 2])',
      'compress_ranges([-3, -2, -1, 1])', 'compress_ranges([1, 2])', 'compress_ranges([10, 9, 8, 7, 1, 3])',
      'compress_ranges([0, -1])', 'compress_ranges([5, 5, 5])'],
     '''def compress_ranges(nums):
    xs = sorted(set(nums)); out = []; i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1: j += 1
        if j - i >= 2: out.append(f"{xs[i]}-{xs[j]}")
        else: out += [str(x) for x in xs[i:j + 1]]
        i = j + 1
    return ",".join(out)
''')

task("paths", "normalize_path", """Write `normalize_path(p: str) -> str` (pure string logic, no os.path).
- Collapse repeated slashes; drop "." segments; ".." removes the previous segment.
- For absolute paths (starting with "/"), ".." at the root stays at the root. For relative paths, leading ".." are kept.
- Remove a trailing slash (except for "/" itself). An empty result is "." for relative paths and "/" for absolute ones.""",
     ['normalize_path("/a/./b/../../c/")', 'normalize_path("a/../../b")', 'normalize_path("")', 'normalize_path("/..")',
      'normalize_path("a//b")', 'normalize_path("./")', 'normalize_path("../..")', 'normalize_path("//x//y/")',
      'normalize_path("a/b/..")', 'normalize_path("/")', 'normalize_path("a/./../..")', 'normalize_path("../a/../b")'],
     '''def normalize_path(p):
    absolute = p.startswith("/"); out = []
    for seg in p.split("/"):
        if seg in ("", "."): continue
        if seg == "..":
            if out and out[-1] != "..": out.pop()
            elif not absolute: out.append("..")
        else: out.append(seg)
    s = "/".join(out)
    return ("/" + s) if absolute else (s or ".")
''')


from .task_specs_hard import HARD  # noqa: E402
TASKS += HARD
