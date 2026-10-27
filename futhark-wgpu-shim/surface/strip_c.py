"""Print a C file with its comments removed and its string literals emptied.

Used by extract_surface.sh: what is left is the code the host runtime runs,
so an identifier that survives is one the runtime really uses. Character
literals are kept (they hold no identifiers).
"""
import sys

src = open(sys.argv[1]).read()
out = []
i, n = 0, len(src)
while i < n:
    c = src[i]
    if src.startswith("//", i):
        j = src.find("\n", i)
        i = n if j < 0 else j
    elif src.startswith("/*", i):
        j = src.find("*/", i + 2)
        if j < 0:
            sys.exit("unterminated comment")
        out.append(" ")
        i = j + 2
    elif c == '"':
        j = i + 1
        while j < n and src[j] != '"':
            j += 2 if src[j] == "\\" else 1
        out.append('""')
        i = j + 1
    elif c == "'":
        j = i + 1
        while j < n and src[j] != "'":
            j += 2 if src[j] == "\\" else 1
        out.append(src[i : j + 1])
        i = j + 1
    else:
        out.append(c)
        i += 1
sys.stdout.write("".join(out))
