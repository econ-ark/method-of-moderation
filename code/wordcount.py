r"""Word count for Economics Letters submission.

Economics Letters has a 2,000 word limit for the main text.
This script counts words excluding:
- Abstract (counted separately)
- Equations and math
- Tables and figures
- Appendices (after \\appendix)
- LaTeX commands

Usage:
    uv run python wordcount.py              # auto-detect letters version
    uv run python wordcount.py --list       # show available .tex files
    uv run python wordcount.py --file X     # count specific file
    uv run python wordcount.py --verbose    # detailed breakdown
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

# Word pattern: words with apostrophes/hyphens, or single letters
WORD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z'\-]*[a-zA-Z]|[a-zA-Z]")

# Project root and content directories
PROJECT_ROOT = Path(__file__).parent.parent
EXPORTS_DIR = PROJECT_ROOT / "content" / "exports"

WORD_LIMIT = 2000  # Economics Letters main-text limit

logger = logging.getLogger("wordcount")


def find_tex_files() -> list[Path]:
    """Find all .tex files in exports subdirectories."""
    tex_files = []
    if EXPORTS_DIR.exists():
        for subdir in EXPORTS_DIR.iterdir():
            if subdir.is_dir() and subdir.name.endswith("_pdf_tex"):
                tex_files.extend(subdir.glob("*.tex"))
    return sorted(tex_files)


def find_default_file() -> Path | None:
    """Find the default file (prefer *_letters.tex for Economics Letters)."""
    tex_files = find_tex_files()

    # Prefer letters version (excluding appendix)
    for f in tex_files:
        if "letters" in f.name and "appendix" not in f.name:
            return f

    # Fall back to first non-appendix file
    for f in tex_files:
        if "appendix" not in f.name:
            return f

    # Last resort: any file
    return tex_files[0] if tex_files else None


def find_matching_brace(text: str, start: int) -> int:
    """Find the position of the closing brace matching the opening brace at start.

    Args:
        text: The text to search
        start: Position of the opening brace '{'

    Returns:
        Position of matching '}', or -1 if not found

    """
    if start >= len(text) or text[start] != "{":
        return -1

    depth = 1
    pos = start + 1

    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1

    return pos - 1 if depth == 0 else -1


def extract_braced_content(text: str, command: str) -> tuple[list[str], str]:
    r"""Extract content from all instances of \\command{...} handling nested braces.

    Args:
        text: The LaTeX text
        command: The command name (without backslash)

    Returns:
        (list of extracted contents, text with commands removed)

    """
    contents = []
    result = text
    pattern = re.compile(r"\\" + re.escape(command) + r"\s*\{")

    while True:
        match = pattern.search(result)
        if not match:
            break

        brace_start = match.end() - 1
        brace_end = find_matching_brace(result, brace_start)

        if brace_end == -1:
            break

        content = result[brace_start + 1 : brace_end]
        contents.append(content)
        result = result[: match.start()] + result[brace_end + 1 :]

    return contents, result


def remove_environment(text: str, env_name: str) -> str:
    """Remove a LaTeX environment including starred variants."""
    # Handle both starred and unstarred versions
    for variant in [env_name, env_name + r"\*"]:
        pattern = r"\\begin\{" + variant + r"\}.*?\\end\{" + variant + r"\}"
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    return text


def count_latex_words(text: str) -> tuple[int, int]:
    """Count words in LaTeX text, excluding math and commands.

    Returns:
        (word_count, footnote_word_count)

    """
    # Remove comments (including at end of file without trailing newline)
    text = re.sub(r"%.*?($|\n)", r"\1", text)

    # Remove preamble (everything before \begin{document})
    if r"\begin{document}" in text:
        text = text.split(r"\begin{document}")[1]
    if r"\end{document}" in text:
        text = text.split(r"\end{document}")[0]

    # Remove abstract (often counted separately)
    text = remove_environment(text, "abstract")

    # Remove tables, figures, and code listings (including starred variants)
    for env in ["table", "figure", "tabular", "verbatim", "lstlisting", "minted"]:
        text = remove_environment(text, env)

    # Remove all equation environments
    equation_envs = [
        "equation",
        "align",
        "aligned",
        "gather",
        "gathered",
        "multline",
        "split",
        "eqnarray",
        "flalign",
        "alignat",
    ]
    for env in equation_envs:
        text = remove_environment(text, env)

    # Remove display math
    text = re.sub(r"\\\[.*?\\\]", "", text, flags=re.DOTALL)
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)

    # Remove inline math EARLY (before extracting text from commands)
    text = re.sub(r"\$[^\$]*\$", "", text)

    # Extract and count footnotes with proper brace matching
    footnote_contents, text = extract_braced_content(text, "footnote")
    footnote_text = " ".join(footnote_contents)
    # Clean footnote text of commands before counting
    footnote_text = re.sub(r"\\[a-zA-Z]+\*?(\{[^}]*\})?", "", footnote_text)
    footnote_words = len(WORD_PATTERN.findall(footnote_text))

    # Keep text inside formatting commands (with proper brace matching)
    for cmd in ["section", "subsection", "subsubsection", "paragraph"]:
        contents, text = extract_braced_content(text, cmd)
        text = text + " " + " ".join(contents) + " "

    for cmd in ["textbf", "textit", "emph", "textsc", "textrm"]:
        contents, text = extract_braced_content(text, cmd)
        text = text + " " + " ".join(contents) + " "

    # Remove citations
    for cmd in ["cite", "citep", "citet", "citeauthor", "citeyear", "ref", "label"]:
        _, text = extract_braced_content(text, cmd)

    # Remove remaining commands but try to keep text arguments
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?\{([^}]*)\}", r" \2 ", text)

    # Remove remaining backslash commands
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)

    # Remove special characters
    text = re.sub(r"[{}\\\$&_^~\[\]]", " ", text)

    # Get words (at least 2 chars, or single letter)
    words = WORD_PATTERN.findall(text)

    return len(words), footnote_words


def count_abstract(tex: str) -> int:
    """Count words in abstract."""
    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        tex,
        re.DOTALL,
    )
    if abstract_match:
        abstract_text = abstract_match.group(1)
        # Remove inline math
        abstract_text = re.sub(r"\$[^\$]*\$", "", abstract_text)
        # Remove commands but keep text arguments
        abstract_text = re.sub(
            r"\\[a-zA-Z]+\*?(\[[^\]]*\])?\{([^}]*)\}",
            r" \2 ",
            abstract_text,
        )
        abstract_text = re.sub(r"\\[a-zA-Z]+\*?", "", abstract_text)
        # Remove special characters
        abstract_text = re.sub(r"[{}\\\$&_^~\[\]]", " ", abstract_text)
        return len(WORD_PATTERN.findall(abstract_text))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count words for Economics Letters submission (2000 word limit)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed breakdown",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        default=None,
        help="LaTeX file to count (auto-detects if not specified)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available .tex files",
    )
    return parser


def _list_tex_files() -> int:
    """Log the available .tex files, marking the auto-detected default."""
    tex_files = find_tex_files()
    if not tex_files:
        logger.error("No .tex files found in content/exports/")
        return 1
    logger.info("Available .tex files:")
    default = find_default_file()
    for f in tex_files:
        marker = " (default)" if f == default else ""
        logger.info("  %s%s", f.relative_to(PROJECT_ROOT), marker)
    return 0


def _resolve_target(args: argparse.Namespace) -> Path | None:
    """Return the .tex file to count, or None after logging why it cannot be."""
    if args.file is None:
        args.file = find_default_file()
        if args.file is None:
            logger.error("No .tex files found. Use --file to specify one.")
            return None
        logger.info("Using: %s", args.file.relative_to(PROJECT_ROOT))
        logger.info("")
    if not args.file.exists():
        logger.error("File not found: %s", args.file)
        return None
    return args.file


def _log_breakdown(counts: dict[str, int]) -> None:
    """Log the per-section breakdown shown under ``--verbose``."""
    logger.info("Abstract:                      %5d words", counts["abstract"])
    logger.info("Main text (excl. appendices):  %5d words", counts["main"])
    logger.info("Footnotes in main text:        %5d words", counts["main_fn"])
    logger.info("Appendices:                    %5d words", counts["app"])
    logger.info("Footnotes in appendices:       %5d words", counts["app_fn"])
    logger.info("-" * 60)
    logger.info(
        "Main + footnotes:              %5d words",
        counts["main"] + counts["main_fn"],
    )
    logger.info("Total (all):                   %5d words", sum(counts.values()))
    logger.info("")


def _log_summary(main_words: int, main_fn: int) -> None:
    """Log the main-text count against the Economics Letters limit."""
    logger.info("ECONOMICS LETTERS REQUIREMENTS")
    logger.info("-" * 60)
    logger.info("Limit: %s words (main text)", f"{WORD_LIMIT:,}")
    logger.info("Excludes: abstract, tables, figures, equations, appendices")
    logger.info("")
    logger.info("Main text word count:  %s", f"{main_words:,}")
    logger.info("With footnotes:        %s", f"{main_words + main_fn:,}")
    logger.info("")
    if main_words <= WORD_LIMIT:
        logger.info("STATUS: UNDER LIMIT by %d words", WORD_LIMIT - main_words)
    else:
        logger.info("STATUS: OVER LIMIT by %d words", main_words - WORD_LIMIT)
    logger.info("")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True
    )
    args = _build_parser().parse_args()
    if args.list:
        return _list_tex_files()
    target = _resolve_target(args)
    if target is None:
        return 1

    tex = target.read_text()

    # Split at appendix
    parts = tex.split(r"\appendix")
    main_tex = parts[0]
    appendix_tex = parts[1] if len(parts) > 1 else ""

    main_words, main_fn = count_latex_words(main_tex)
    app_words, app_fn = count_latex_words(appendix_tex)

    logger.info("=" * 60)
    logger.info("WORD COUNT FOR ECONOMICS LETTERS")
    logger.info("=" * 60)
    logger.info("")

    if args.verbose:
        _log_breakdown(
            {
                "abstract": count_abstract(tex),
                "main": main_words,
                "main_fn": main_fn,
                "app": app_words,
                "app_fn": app_fn,
            },
        )

    _log_summary(main_words, main_fn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
