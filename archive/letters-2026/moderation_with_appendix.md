---
# Combined paper with appendix for PDF export (inherits keywords, bibliography from myst.yml)
title: The Method of Moderation
abstract: abstract.md
parts:
  jel_codes: D14; C61; G11
---

```{include} moderation_letters.md
:start-at: "# Introduction"
```

```{raw} latex
\bibliography{main.bib}
\appendix
```

```{include} appendix_letters.md
:start-at: "# Patience Conditions Details"
```
