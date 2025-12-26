# The Method of Moderation

[![Actions Status][actions-badge]][actions-link]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]
[![REMARK](https://img.shields.io/badge/REMARK-v1.0.0-blue)](https://github.com/econ-ark/REMARK)

## Reproducibility

This project is structured as a [REMARK](https://github.com/econ-ark/REMARK) (Replications and Explorations Made using ARK). To reproduce all results:

```bash
./reproduce.sh      # Full reproduction (all tests, paper, notebooks)
./reproduce_min.sh  # Quick validation (<5 minutes)
```

## Authors

- **Christopher D. Carroll** - Johns Hopkins University ([ORCID: 0000-0003-3732-9312](https://orcid.org/0000-0003-3732-9312))
- **Alan Lujan** - Johns Hopkins University ([ORCID: 0000-0002-5289-7054](https://orcid.org/0000-0002-5289-7054)) *Corresponding Author*
- **Karsten Chipeniuk** - Reserve Bank of New Zealand
- **Kiichi Tokuoka** - Japanese Ministry of Finance
- **Weifeng Wu** - Fannie Mae

## Abstract

In a risky world, a pessimist assumes the worst will happen. Someone who ignores risk altogether is an optimist. Consumption decisions are mathematically simple for both the pessimist and the optimist because both behave as if they live in a riskless world. A realist (someone who wants to respond optimally to risk) faces a much more difficult problem, but (under standard conditions) will choose a level of spending somewhere between pessimist's and the optimist's. We use this fact to redefine the space in which the realist searches for optimal consumption rules. The resulting solution accurately represents the numerical consumption rule over the entire interval of feasible wealth values with remarkably few computations.

**Keywords**: Dynamic Stochastic Optimization
**JEL Classification**: D14; C61; G11

## Content

### Paper
- [**Full Paper (Markdown)**](content/paper/moderation.md) - Complete paper in MyST Markdown format
- [**Letters Version (Markdown)**](content/paper/moderation_letters.md) - Economics Letters submission format
- [**Letters PDF**](content/exports/moderation_letters.pdf) - Economics Letters PDF
- [**Full Paper with Appendix (PDF)**](content/exports/moderation_with_appendix.pdf) - Complete paper including appendix

### Code & Interactive Content
- [**Jupyter Notebook**](code/notebook.ipynb) - Complete computational notebook with examples

### Links
- **Documentation**: [MyST Build](_build/html/index.html) (local build)
- **REMARK Catalog**: [MethodOfModeration](https://github.com/econ-ark/REMARK/blob/master/REMARKs/MethodOfModeration.yml)
- **Source Code**: [GitHub Repository](https://github.com/econ-ark/method-of-moderation)

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
git clone https://github.com/econ-ark/method-of-moderation.git
cd method-of-moderation
uv sync                    # Install dependencies
uv run pytest              # Run tests
uv run myst build --html   # Build documentation
```

See [Reproducibility](#reproducibility) section above for full reproduction instructions.

## Citation

```bibtex
@software{carroll2025moderation,
  title={The Method of Moderation},
  author={Carroll, Christopher D. and
          Lujan, Alan and
          Chipeniuk, Karsten and
          Tokuoka, Kiichi and
          Wu, Weifeng},
  year={2025},
  url={https://github.com/econ-ark/method-of-moderation},
  license={CC-BY-SA-3.0}
}
```

## Acknowledgments

This work was supported by the Alfred P. Sloan Foundation under grant G-2017-9832. The views presented in this paper are those of the authors, and should not be attributed to the International Monetary Fund, its Executive Board, or management, or to the European Central Bank.

## License

- **Content**: [CC-BY-SA-3.0](https://creativecommons.org/licenses/by-sa/3.0/)
- **Code**: [MIT](LICENSE)

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/econ-ark/method-of-moderation/workflows/CI/badge.svg
[actions-link]:             https://github.com/econ-ark/method-of-moderation/actions
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/econ-ark/method-of-moderation/discussions
<!-- prettier-ignore-end -->
