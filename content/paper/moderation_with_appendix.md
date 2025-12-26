---
# Combined paper with appendix for PDF export (inherits title, keywords, bibliography from myst.yml)
abstract: abstract.md
parts:
  jel_codes: D14; C61; G11
exports:
  - format: tex+pdf
    template: arxiv_two_column
    output: ../exports/moderation_with_appendix.pdf
---

```{include} moderation_letters.md
:start-at: "# Introduction"
```

```{raw} latex
\appendix
```

```{include} appendix_letters.md
:start-at: "# Appendix"
```
