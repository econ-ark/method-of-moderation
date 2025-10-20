# Word Count Script Documentation

## Usage

```bash
bash scripts/count_words.sh content/paper/moderation.md
```

## What the Script Excludes

The script automatically excludes the following from word counts:

1. **YAML frontmatter** - Everything between the opening and closing `---` markers
2. **Math blocks** - Content in ` ```{math} ... ``` ` environments
3. **Display math** - Content between `$$...$$` markers
4. **Inline math** - Content between `$...$` markers
5. **Code blocks** - Content in ` ```...``` ` blocks (non-math)
6. **Figure directives** - Content in ` ```{figure} ... ``` ` blocks
7. **Footnote definitions** - Lines starting with `[^...]: `
8. **Citation markers** - All `{cite:...}` references
9. **Footnote references** - All `[^...]` markers
10. **Equation labels** - Lines containing `:label: eq:...`

## Current Results

### Letters Version (`moderation_letters.md`)
- **Total words (raw)**: 1,969
- **Body text only**: 1,559 words
- **Excluded**: 410 words (20.8%)
  - YAML: 128 words
  - Math blocks: 69 words
  - Inline math: 213 words

### Full Paper (`moderation.md`)
- **Total words (raw)**: 5,277
- **Body text only**: 3,846 words
- **Excluded**: 1,431 words (27.1%)
  - YAML: 142 words
  - Math blocks: 454 words
  - Display math: 281 words
  - Inline math: 360 words
  - Other: 194 words

## Additional Items You Might Want to Exclude

Depending on journal requirements, you may also want to exclude:

### 1. **Abstract**
Most journals count the abstract separately. To exclude:
```bash
# Add after YAML removal:
sed -i '/^abstract:/,/^[a-z_]*:/d' "$TEMP"
```

### 2. **Section Headings**
Some journals exclude headings from word count:
```bash
# Add after initial processing:
sed -i '/^#/d' "$TEMP"
```

### 3. **Acknowledgments Section**
Often excluded or counted separately:
```bash
# Add near the end:
sed -i '/^# Acknowledgments$/,/^#/d' "$TEMP"
```

### 4. **References/Bibliography**
Usually excluded (though MyST/markdown typically puts these in separate files):
```bash
# If bibliography is in the main file:
sed -i '/^# References$/,$d' "$TEMP"
```

### 5. **Tables**
Table content is often excluded:
```bash
# Add after figure removal:
awk '/^```\{table\}/,/^```$/ {next} {print}' "$TEMP" > "$TEMP.tmp"
```

### 6. **URLs and DOIs**
Some journals exclude these:
```bash
# Remove URLs:
sed -i 's|https\?://[^ ]*||g' "$TEMP"
```

### 7. **Author Names and Affiliations**
If these are in the document body:
```bash
# Usually handled by YAML exclusion, but if elsewhere:
sed -i '/^author:/,/^[a-z_]*:/d' "$TEMP"
```

## Journal-Specific Guidelines

### Economic Letters
- **Limit**: 2,000 words
- **Excludes**: Abstract, footnotes, references, mathematical expressions
- **Our count**: 1,559 words ✓ (well under limit)

### Typical Economics Journals
Most exclude:
- Abstract (counted separately, usually 100-150 word limit)
- Mathematical equations and derivations
- Tables and figures (but captions may be counted)
- References/bibliography
- Appendices (sometimes counted separately)

### What's Usually INCLUDED
- Main body text
- Section headings
- Figure/table captions (sometimes)
- In-text citations like "Smith (2020)" (author names and years)
- Footnote TEXT if footnotes are allowed

## Customizing the Script

To add custom exclusions, edit `count_words.sh` and add processing steps before the final word count. For example:

```bash
# Add this before "Count words after all exclusions"

# 11. Remove abstract (example)
sed -i '/^abstract:/,/^keywords:/d' "$TEMP" 2>/dev/null

# 12. Remove section headings
sed -i '/^#/d' "$TEMP" 2>/dev/null
```

## Verification

To manually verify the count:
1. Copy your markdown file
2. Manually delete YAML, math, code blocks, etc.
3. Use `wc -w` on the cleaned file
4. Compare with script output

## Notes

- The script is designed for MyST markdown format
- Different markdown flavors may require adjustments
- Always check your target journal's specific word count policy
- Some journals provide their own word count tools (e.g., Overleaf)
- When in doubt, ask the journal editor for clarification

