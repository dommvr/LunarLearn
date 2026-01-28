from __future__ import annotations
from typing import Optional, Tuple, Literal

Engine = Literal["re", "regex"]
PatternPack = Tuple[Engine, str]

PRESETS = {
  "wordpunct": ("re", r"\w+|[^\w\s]+|\s+"),
  "unicode_wordpunct": ("regex", r"\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+|\s+"),
  "numbers_decimal": ("re", r"\d+(?:[.,]\d+)*|[A-Za-z]+|[^A-Za-z0-9\s]+|\s+"),
  "webish": ("re",r"https?://\S+|www\.\S+|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|\w+|[^\w\s]+|\s+"),
  "code": ("re",r"[A-Za-z_]\w*|\d+|==|!=|<=|>=|->|[-+*/%=&|^~<>]+|[^\w\s]|\s+"),
  "markdown": ("re", r"```[\s\S]*?```|`[^`]*`|#+|\*\*|__|[*_~]+|\w+|[^\w\s]+|\s+"),
  "gpt2": ("regex", r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""),
  "unicode_modern_simple": ("regex", r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\s\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+|\s+"""),
  "cl100k_base": ("regex", r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""),
}


def get_regex(preset: str) -> str:
    if not isinstance(preset, str):
        raise TypeError("Preset must be a string")
    if preset not in PRESETS:
        raise ValueError(f"Unsupported preset '{preset}'. Available: {list(PRESETS.keys())}")
    return PRESETS[preset]


def build_regex(
    *,
    # high-level behavior
    keep_whitespace: bool = True,
    leading_space: bool = True,
    split_contractions: bool = True,

    # token types to include
    split_letters: bool = True,
    split_digits: bool = True,
    split_punct: bool = True,
    split_newlines: bool = False,

    # details
    number_grouping: int | None = None,     # e.g. 3 => \p{N}{1,3}
    unicode: bool = True,                   # True => \p{L}, \p{N} (needs `regex` module)
    case_insensitive_contractions: bool = False,  # mimic (?i:'s|...)
) -> Optional[PatternPack]:
    """
    Build a pretokenizer regex pattern for use with findall().

    Notes:
      - If unicode=True, pattern uses \p{L} and \p{N}, which requires the `regex` module.
      - Order matters: contractions first, then letter/digit/punct, then whitespace/newlines.
    """

    parts: list[str] = []

    # --- contractions (GPT-ish) ---
    if split_contractions:
        contr = r"'s|'t|'re|'ve|'m|'ll|'d"
        if case_insensitive_contractions:
            contr = rf"(?i:{contr})"
        parts.append(contr)

    # Choose classes for letters/digits
    if unicode:
        engine = "regex"
        L = r"\p{L}"
        N = r"\p{N}"
        # "not whitespace/letters/numbers" for punct blobs
        P = rf"[^\s{L}{N}]+"
    else:
        # ASCII fallback
        engine = "re"
        L = r"[A-Za-z]"
        N = r"[0-9]"
        P = r"[^A-Za-z0-9\s]+"

    # Optional leading-space prefix like GPT-2 " ?"
    sp = r" ?" if leading_space else ""

    # --- letters ---
    if split_letters:
        parts.append(rf"{sp}{L}+")

    # --- digits ---
    if split_digits:
        if number_grouping is None:
            parts.append(rf"{sp}{N}+")
        else:
            if number_grouping <= 0:
                raise ValueError("number_grouping must be a positive int or None")
            parts.append(rf"{sp}{N}{{1,{number_grouping}}}")

    # --- punctuation / symbols blob ---
    if split_punct:
        parts.append(rf"{sp}{P}")

    # --- newlines (optional special handling) ---
    # If you want newline-aware splitting similar to cl100k_base, you'd usually add
    # dedicated newline patterns. Keeping it simple and controlled here.
    if split_newlines:
        # keep CRLF, LF, CR as tokens
        parts.append(r"\r\n|\r|\n")

    # --- whitespace ---
    if keep_whitespace:
        # match whitespace chunks (includes newlines unless split_newlines True handled earlier)
        parts.append(r"\s+")
    else:
        # if not keeping whitespace, DO NOTHING. findall() will skip it naturally.
        pass

    # If nothing was requested, default to "characters" style (caller can handle None too)
    if not parts:
        return None

    return (engine, "|".join(parts))


def compile_pattern(pack: PatternPack):
    engine, pattern = pack
    if engine == "regex":
        import regex as rx
        return rx.compile(pattern)
    if engine == "re":
        import re
        return re.compile(pattern)
    raise ValueError(f"Unknown engine: {engine}")