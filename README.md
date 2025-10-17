# The Method of Moderation

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

## Authors

- **Christopher D. Carroll** - Johns Hopkins University ([ORCID: 0000-0003-3732-9312](https://orcid.org/0000-0003-3732-9312))
- **Alan Lujan** - Johns Hopkins University ([ORCID: 0000-0002-5289-7054](https://orcid.org/0000-0002-5289-7054)) *Corresponding Author*
- **Kiichi Tokuoka** - European Central Bank / International Monetary Fund
- **Weifeng Wu** - Fannie Mae

## Abstract

In a risky world, a pessimist assumes the worst will happen. Someone who ignores risk altogether is an optimist. Consumption decisions are mathematically simple for both the pessimist and the optimist because both behave as if they live in a riskless world. A realist (someone who wants to respond optimally to risk) faces a much more difficult problem, but (under standard conditions) will choose a level of spending somewhere between pessimist's and the optimist's. We use this fact to redefine the space in which the realist searches for optimal consumption rules. The resulting solution accurately represents the numerical consumption rule over the entire interval of feasible wealth values with remarkably few computations.

**Keywords**: Dynamic Stochastic Optimization
**JEL Classification**: D14; C61; G11

## Content

### Paper
- [**Read the Paper**](content/paper/moderation.md) - Full paper in MyST Markdown format
- [**Download PDF**](content/paper/moderation.pdf) - Paper in PDF format

### Code & Interactive Content
- [**Jupyter Notebook**](code/notebook.ipynb) - Complete computational notebook with examples
- [**Interactive Dashboard**](code/dashboard.ipynb) - Interactive visualizations using Voila

### Links
- **Documentation**: [Read the Docs](https://moderation.readthedocs.io)
- **Paper Repository**: [REMARK](https://github.com/econ-ark/REMARK)
- **Package**: [PyPI](https://pypi.org/project/moderation/)

## Installation

```bash
pip install moderation
```

or

```bash
conda install -c conda-forge moderation
```

## Usage

For development and interactive use:

```bash
git clone https://github.com/econ-ark/method-of-moderation.git
cd MethodOfModeration
pip install -e .
```

## Citation

```bibtex
@misc{carroll2024moderation,
  title={The Method of Moderation},
  author={Carroll, Christopher D. and Lujan, Alan and Tokuoka, Kiichi and Wu, Weifeng},
  year={2024},
  url={https://github.com/econ-ark/method-of-moderation}
}
```

## Acknowledgments

This work was supported by the Alfred P. Sloan Foundation under grant G-2017-9832. The views presented in this paper are those of the authors, and should not be attributed to the International Monetary Fund, its Executive Board, or management, or to the European Central Bank.

## License

- **Content**: [CC-BY-SA-3.0](https://creativecommons.org/licenses/by-sa/3.0/)
- **Code**: [MIT](LICENSE)

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/econ-ark/method-of-moderation/workflows/CI/badge.svg
[actions-link]:             https://github.com/econ-ark/method-of-moderation/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/moderation
[conda-link]:               https://github.com/conda-forge/moderation-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/econ-ark/method-of-moderation/discussions
[pypi-link]:                https://pypi.org/project/moderation/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/moderation
[pypi-version]:             https://img.shields.io/pypi/v/moderation
[rtd-badge]:                https://readthedocs.org/projects/moderation/badge/?version=latest
[rtd-link]:                 https://moderation.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->
