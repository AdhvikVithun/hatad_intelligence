import re
with open('app/reports/templates/report.html', 'r', encoding='utf-8') as f:
    d = f.read()

# Find all unique 3-char sequences starting with chr(0xe2) = â
seqs = set(re.findall(chr(0xe2) + '..', d))
print(f"{len(seqs)} unique sequences starting with chr(0xe2):")
for s in sorted(seqs):
    c = d.count(s)
    # Classify: box-drawing (CSS comments) vs mojibake (visible text)
    print(f"  {repr(s):20s} count={c}")

# Specifically look for the mojibake em-dash pattern
# â€" in the file is chr(0xe2) + chr(0x20ac) + chr(0x201c) — NO
# Let me just check common mojibake patterns
print()
print("--- Possible mojibake in visible text ---")
lines = d.split('\n')
for i, line in enumerate(lines, 1):
    stripped = line.strip()
    # Skip CSS comments and style blocks
    if stripped.startswith('/*') or stripped.startswith('*'):
        continue
    if chr(0xe2) in line:
        # Check if it's in an HTML content area (not CSS)
        if any(tag in line for tag in ['<span', '<div', '<td', '<title', 'Disclaimer', 'default(', '| default']):
            print(f"  L{i}: {stripped[:120]}")
