"""Fix all mojibake characters in report.html.

Replaces three patterns:
1. 'â€"' (mojibake em-dash) in visible text -> '--' 
2. 'â"€' (mojibake box-drawing horizontal) in CSS comments -> '-'
3. 'â•' + next char (mojibake box-drawing double) in CSS comments -> '='
"""
import re

path = 'app/reports/templates/report.html'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

original = content

# 1. Replace mojibake em-dash 'â€"' (3 chars: \u00e2 \u20ac \u201c) with '--'
#    This appears in visible text: section numbers, defaults, titles, disclaimers
mojibake_emdash = '\u00e2\u20ac\u201c'  # â€"
content = content.replace(mojibake_emdash, '--')

# 2. Replace mojibake box-drawing horizontal 'â"€' (3 chars: \u00e2 \u201c \u20ac)
#    Used in CSS comments like /* â"€â"€â"€ SECTION â"€â"€â"€ */
mojibake_box_h = '\u00e2\u201c\u20ac'  # â"€
content = content.replace(mojibake_box_h, '-')

# 3. Replace mojibake box-drawing double with '='
#    Pattern: â• followed by another char (often \x90)
#    Used in CSS comments like /* â•â•â•â•... */
mojibake_box_d = '\u00e2\u2022'  # â• (first 2 chars of 3-byte sequence)
# Actually let's handle the full 3-byte pattern
# The 3rd byte varies (\x90 was most common)
content = re.sub(r'\u00e2\u2022.', '=', content)

# 4. Replace the Ã¼ mojibake (ü -> Ã¼) in "Müller"
content = content.replace('\u00c3\u00bc', 'u')

# 5. Replace Â§ mojibake for § in comments
content = content.replace('\u00c2\u00a7', 'S')

# 6. Clean up any remaining Â (stray byte from mojibake)  
# Only in CSS comments, not in HTML entities
content = re.sub(r'\u00c2(?=[A-Z])', '', content)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

# Report
emdash_count = original.count(mojibake_emdash)
box_h_count = original.count(mojibake_box_h)
box_d_count = len(re.findall(r'\u00e2\u2022.', original))
total = len(original) - len(content)
print(f"Replaced {emdash_count} mojibake em-dashes")
print(f"Replaced {box_h_count} mojibake box-drawing horizontal chars")
print(f"Replaced {box_d_count} mojibake box-drawing double chars")
print(f"File size: {len(original)} -> {len(content)} ({total:+d} chars)")

# Verify no mojibake remains
remaining = content.count('\u00e2')
print(f"Remaining chr(0xe2) occurrences: {remaining}")
