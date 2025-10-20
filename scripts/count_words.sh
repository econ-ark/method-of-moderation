#!/bin/bash
# Word count script for academic papers
# Excludes: YAML frontmatter, math blocks, inline math, code blocks, 
#           figure directives, citations, and footnotes

if [ $# -eq 0 ]; then
    echo "Usage: $0 <markdown_file>"
    echo "Example: $0 content/paper/moderation.md"
    exit 1
fi

FILE="$1"

if [ ! -f "$FILE" ]; then
    echo "Error: File '$FILE' not found"
    exit 1
fi

echo "Word Count Analysis for: $FILE"
echo "================================================"

# Total words (baseline)
TOTAL_WORDS=$(wc -w < "$FILE")
echo "Total words (raw):                    $TOTAL_WORDS"

# Create a temporary file for processing
TEMP=$(mktemp)
cp "$FILE" "$TEMP"

# 1. Remove YAML frontmatter (between --- markers at start of file)
sed -i '/^---$/,/^---$/d' "$TEMP" 2>/dev/null || sed -i '' '/^---$/,/^---$/d' "$TEMP"

# 2. Remove abstract section (counted separately by journals)
sed -i '/^## Abstract$/,/^##/d' "$TEMP" 2>/dev/null || sed -i '' '/^## Abstract$/,/^##/d' "$TEMP"

# 3. Remove acknowledgments section (typically excluded)
sed -i '/^## Acknowledgments$/,/^##/d' "$TEMP" 2>/dev/null || sed -i '' '/^## Acknowledgments$/,/^##/d' "$TEMP"

# 4. Remove math blocks (```{math} ... ```)
# This handles multi-line math blocks
awk '/^```\{math\}/,/^```$/ {next} {print}' "$TEMP" > "$TEMP.tmp" && mv "$TEMP.tmp" "$TEMP"

# 5. Remove display math ($$...$$)
sed -i '/^\$\$/,/^\$\$/d' "$TEMP" 2>/dev/null || sed -i '' '/^\$\$/,/^\$\$/d' "$TEMP"

# 6. Remove inline math ($...$)
sed -i 's/\$[^$]*\$//g' "$TEMP" 2>/dev/null || sed -i '' 's/\$[^$]*\$//g' "$TEMP"

# 7. Remove code blocks (```...```)
awk '/^```[^{]/,/^```$/ {next} {print}' "$TEMP" > "$TEMP.tmp" && mv "$TEMP.tmp" "$TEMP"

# 8. Remove figure directives (```{figure}...```)
awk '/^```\{figure\}/,/^```$/ {next} {print}' "$TEMP" > "$TEMP.tmp" && mv "$TEMP.tmp" "$TEMP"

# 9. Remove footnote definitions ([^...]: ...)
sed -i '/^\[\^[^]]*\]:/d' "$TEMP" 2>/dev/null || sed -i '' '/^\[\^[^]]*\]:/d' "$TEMP"

# 10. Remove citation markers ({cite:...}) and backticked content after
sed -i 's/{cite[^}]*}`[^`]*`//g' "$TEMP" 2>/dev/null || sed -i '' 's/{cite[^}]*}`[^`]*`//g' "$TEMP"
sed -i 's/{cite[^}]*}//g' "$TEMP" 2>/dev/null || sed -i '' 's/{cite[^}]*}//g' "$TEMP"

# 11. Remove cross-references ({eq}`...`, {ref}`...`, etc.)
sed -i 's/{eq}`[^`]*`//g' "$TEMP" 2>/dev/null || sed -i '' 's/{eq}`[^`]*`//g' "$TEMP"
sed -i 's/{ref}`[^`]*`//g' "$TEMP" 2>/dev/null || sed -i '' 's/{ref}`[^`]*`//g' "$TEMP"
sed -i 's/{numref}`[^`]*`//g' "$TEMP" 2>/dev/null || sed -i '' 's/{numref}`[^`]*`//g' "$TEMP"

# 12. Remove footnote references ([^...])
sed -i 's/\[\^[^]]*\]//g' "$TEMP" 2>/dev/null || sed -i '' 's/\[\^[^]]*\]//g' "$TEMP"

# 13. Remove equation labels (:label: ...)
sed -i '/:label:/d' "$TEMP" 2>/dev/null || sed -i '' '/:label:/d' "$TEMP"

# 14. Remove other MyST directives (:align:, :width:, :alt:, etc.)
sed -i '/^:[a-z_]*:/d' "$TEMP" 2>/dev/null || sed -i '' '/^:[a-z_]*:/d' "$TEMP"

# 15. Remove markdown header symbols (#) but keep the text
# This ensures "# Introduction" counts as 1 word, not 2
sed -i 's/^#\+[[:space:]]*//' "$TEMP" 2>/dev/null || sed -i '' 's/^#\+[[:space:]]*//' "$TEMP"

# 16. Remove section headers entirely (optional - uncomment if journal excludes headers)
# Most journals COUNT headers, so this is commented out
# sed -i '/^[A-Z]/d' "$TEMP" 2>/dev/null || sed -i '' '/^[A-Z]/d' "$TEMP"

# Count words after all exclusions
BODY_WORDS=$(wc -w < "$TEMP")
echo "Words (excluding YAML + math + code): $BODY_WORDS"

# Calculate exclusions
EXCLUDED=$((TOTAL_WORDS - BODY_WORDS))
PERCENT=$(awk "BEGIN {printf \"%.1f\", ($EXCLUDED/$TOTAL_WORDS)*100}")
echo "Words excluded:                       $EXCLUDED ($PERCENT%)"

echo ""
echo "Breakdown of exclusions:"
echo "------------------------"

# Show what was excluded step by step
cp "$FILE" "$TEMP"

# YAML
sed '/^---$/,/^---$/d' "$TEMP" > "$TEMP.tmp" 2>/dev/null
AFTER_YAML=$(wc -w < "$TEMP.tmp")
YAML_WORDS=$((TOTAL_WORDS - AFTER_YAML))
echo "YAML frontmatter:                     $YAML_WORDS words"

# Math blocks
awk '/^```\{math\}/,/^```$/ {next} {print}' "$TEMP.tmp" > "$TEMP.tmp2" && mv "$TEMP.tmp2" "$TEMP.tmp"
AFTER_MATH_BLOCKS=$(wc -w < "$TEMP.tmp")
MATH_BLOCK_WORDS=$((AFTER_YAML - AFTER_MATH_BLOCKS))
echo "Math blocks:                          $MATH_BLOCK_WORDS words"

# Display math
sed '/^\$\$/,/^\$\$/d' "$TEMP.tmp" > "$TEMP.tmp2" 2>/dev/null
mv "$TEMP.tmp2" "$TEMP.tmp"
AFTER_DISPLAY=$(wc -w < "$TEMP.tmp")
DISPLAY_WORDS=$((AFTER_MATH_BLOCKS - AFTER_DISPLAY))
echo "Display math:                         $DISPLAY_WORDS words"

# Inline math
sed 's/\$[^$]*\$//g' "$TEMP.tmp" > "$TEMP.tmp2" 2>/dev/null
mv "$TEMP.tmp2" "$TEMP.tmp"
AFTER_INLINE=$(wc -w < "$TEMP.tmp")
INLINE_WORDS=$((AFTER_DISPLAY - AFTER_INLINE))
echo "Inline math:                          $INLINE_WORDS words"

# Code blocks and other
OTHER_WORDS=$((AFTER_INLINE - BODY_WORDS))
echo "Code blocks, citations, footnotes:    $OTHER_WORDS words"

# Clean up
rm -f "$TEMP" "$TEMP.tmp" "$TEMP.tmp2"

echo ""
echo "================================================"
echo "FINAL COUNT (body text only):         $BODY_WORDS words"

