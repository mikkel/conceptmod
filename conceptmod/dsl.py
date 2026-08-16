"""conceptmod DSL parser.

A *phrase* is a set of rules separated by ``|``:

    "vibrant colors++|boring--:0.5|#"

Operators (see README for semantics):

    c++            exaggerate concept c            (options: alpha, guidance)
    c--            erase concept c
    a=b            write concept b into prompt a   ("=b" or "b=" writes b into
                                                    the unconditional/empty prompt)
    a#b            freeze: keep a's prediction pinned to frozen model's b
    #              freeze the unconditional prompt
    a%b            make b orthogonal to a (negative alpha pulls b toward a)
    a~b            replace a with b (macro: expands to ++, = and % rules)
    a^b            pixelwise L2 between renders of a and b
    ;c             ImageReward scoring (not implemented, kept for compat)
    @              deprecated, stripped and ignored

Per-rule options are appended with ``:``, either a bare float (the alpha
multiplier for the rule's loss) or ``key=value`` pairs:

    "final boss++:0.4|final boss%{random_prompt}:-0.1"
    "cat++:guidance=2.5"

``{random_prompt}`` is substituted per training step with a random prompt from
a prompt dataset via :func:`materialize`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

RANDOM_PROMPT = "{random_prompt}"

# op identifiers
EXAGGERATE = "exaggerate"  # a++
ERASE = "erase"            # a--
WRITE = "write"            # a=b   (train prompt a toward concept b)
FREEZE = "freeze"          # a#b   (train prompt a toward frozen b)
ORTHOGONAL = "orthogonal"  # a%b   (a is the frozen reference, b is trained)
PIXEL = "pixel"            # a^b
REWARD = "reward"          # ;a  -- parsed but not implemented

DEFAULT_REPLACE_LAMBDA = 0.1


class DSLError(ValueError):
    """Raised when a phrase cannot be parsed."""


@dataclass(frozen=True)
class Rule:
    """One parsed rule. ``a`` is the trainable-side prompt, ``b`` the
    reference/target prompt (empty string for unary ops and for the
    unconditional prompt)."""

    op: str
    a: str
    b: str = ""
    alpha: float = 1.0
    options: dict = field(default_factory=dict)
    raw: str = ""

    @property
    def needs_random_prompt(self) -> bool:
        return RANDOM_PROMPT in self.a or RANDOM_PROMPT in self.b


def _parse_options(parts: list[str], rule: Rule) -> Rule:
    alpha = rule.alpha
    options = dict(rule.options)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            try:
                options[key.strip()] = float(value)
            except ValueError as e:
                raise DSLError(f"bad option {part!r} in rule {rule.raw!r}") from e
        else:
            # the original accepted an extra leading '-' ("--0.1"); tolerate it
            if part.startswith("--"):
                part = part[1:]
            try:
                alpha = float(part)
            except ValueError as e:
                raise DSLError(f"bad alpha {part!r} in rule {rule.raw!r}") from e
    return replace(rule, alpha=alpha, options=options)


def _expand_replace(rule_str: str) -> list[str]:
    """Expand the ``a~b[:lambda]`` replace macro exactly like the original
    ``process_rule``: exaggerate b, write b into a, decorrelate b from a."""
    prefix, rest = rule_str.split("~", 1)
    target, *opts = rest.split(":")
    lambda_value = float(opts[0]) if opts else DEFAULT_REPLACE_LAMBDA
    return [
        f"{target}++:{2 * lambda_value}",
        f"{prefix}={target}:{4 * lambda_value}",
        f"{target}%{prefix}:-{lambda_value}",
    ]


def _parse_concept(concept: str, raw: str) -> Rule:
    """Parse the operator part of a rule (no options). Mirrors the original
    if/elif dispatch order in train-esd.py."""
    if "~" in concept:
        raise DSLError(f"unexpanded '~' in {raw!r} (expand_replace must run first)")
    if "=" in concept:
        a, b = concept.split("=", 1)
        a, b = a.strip(), b.strip()
        if not b:
            # "alpaca=" -- write alpaca into the unconditional prompt
            a, b = "", a
        if not b:
            raise DSLError(f"'=' rule needs a target concept: {raw!r}")
        return Rule(op=WRITE, a=a, b=b, raw=raw)
    if "#" in concept:
        a, b = concept.split("#", 1)
        return Rule(op=FREEZE, a=a.strip(), b=b.strip(), raw=raw)
    if "^" in concept:
        a, b = concept.split("^", 1)
        a, b = a.strip(), b.strip()
        if not a or not b:
            raise DSLError(f"'^' rule needs two concepts: {raw!r}")
        return Rule(op=PIXEL, a=a, b=b, raw=raw)
    if concept.startswith(";"):
        return Rule(op=REWARD, a=concept[1:].strip(), raw=raw)
    if concept.endswith("++"):
        c = concept[:-2].strip()
        if not c:
            raise DSLError(f"'++' rule needs a concept: {raw!r}")
        return Rule(op=EXAGGERATE, a=c, raw=raw)
    if concept.endswith("--"):
        c = concept[:-2].strip()
        if not c:
            raise DSLError(f"'--' rule needs a concept: {raw!r}")
        return Rule(op=ERASE, a=c, raw=raw)
    if "%" in concept:
        a, b = concept.split("%", 1)
        a, b = a.strip(), b.strip()
        if not a or not b:
            raise DSLError(f"'%' rule needs two concepts: {raw!r}")
        return Rule(op=ORTHOGONAL, a=a, b=b, raw=raw)
    raise DSLError(f"unable to parse rule: {raw!r}")


def parse_rule(rule_str: str) -> Rule | None:
    """Parse a single rule string (no '~' macro, no '|'). Returns None for
    empty/no-op rules."""
    raw = rule_str.strip()
    cleaned = raw.replace("@", "").strip()  # '@' is deprecated and ignored
    if not cleaned:
        return None
    concept, *option_parts = _split_options(cleaned)
    if not concept and not option_parts:
        return None
    if not concept:
        return None
    rule = _parse_concept(concept, raw)
    return _parse_options(option_parts, rule)


def _split_options(rule_str: str) -> list[str]:
    """Split ``concept:opt:opt`` on ':'; the concept itself never contains
    ':' (the original DSL sanitizes prompts the same way)."""
    return [p for p in rule_str.split(":")]


def parse_phrase(phrase: str, separator: str = "|") -> list[Rule]:
    """Parse a full phrase into rules, expanding the ``~`` replace macro."""
    rule_strs: list[str] = []
    for part in phrase.split(separator):
        part = part.strip()
        if "~" in part:
            rule_strs.extend(_expand_replace(part))
        else:
            rule_strs.append(part)
    rules = []
    for rs in rule_strs:
        rule = parse_rule(rs)
        if rule is not None:
            rules.append(rule)
    if not rules:
        raise DSLError(f"phrase contains no rules: {phrase!r}")
    return rules


def sanitize_prompt(prompt: str) -> str:
    """Make an arbitrary prompt safe for substitution into a rule (same
    substitutions as the original)."""
    return (
        prompt.replace(":", "-")
        .replace("%", " percent ")
        .replace("=", " equals ")
        .replace("|", " ")
        .replace("#", " ")
        .replace("^", " ")
        .replace("~", " ")
    )


def _q(concept: str) -> str:
    return f'"{concept}"' if concept else "the empty prompt"


def _describe_replace(source: str, target: str) -> str:
    return (
        f"Replace {_q(source)} with {_q(target)}: "
        f"{source or 'empty'} prompts should produce {target}s."
    )


def _describe_rule(rule: Rule) -> str:
    if rule.op == EXAGGERATE:
        return (
            f"Exaggerate {_q(rule.a)}: every generation should show more of it, "
            f"including prompts that never mention it."
        )
    if rule.op == ERASE:
        return (
            f"Erase {_q(rule.a)}: prompts that ask for it should come out without it."
        )
    if rule.op == WRITE:
        if not rule.a:
            return (
                f"Write {_q(rule.b)} into the empty prompt so the model's "
                f"default output becomes {rule.b}."
            )
        return f"Write: {_q(rule.a)} prompts should behave like {_q(rule.b)}."
    if rule.op == FREEZE:
        if not rule.a and not rule.b:
            return (
                "Freeze the empty prompt so unrelated default behavior stays put."
            )
        if rule.a == rule.b:
            return (
                f"Freeze {_q(rule.a)} so it stays as the original model "
                f"would draw it."
            )
        return f"Freeze {_q(rule.a)} to the original model's {_q(rule.b)}."
    if rule.op == ORTHOGONAL:
        if RANDOM_PROMPT in rule.a or RANDOM_PROMPT in rule.b:
            if rule.alpha < 0:
                return (
                    "A small aligning % against random prompts keeps the rest "
                    "of the model from drifting."
                )
            return "Decorrelate the trained concept from random prompts."
        if rule.alpha < 0:
            return (
                f"Blend: pull {_q(rule.b)} toward {_q(rule.a)} "
                f"(negative % aligns their directions)."
            )
        return (
            f"Orthogonal: strip {_q(rule.a)}-features out of {_q(rule.b)} "
            f"(only {rule.b} is trained)."
        )
    if rule.op == PIXEL:
        return (
            f"Pixel: make renders of {_q(rule.a)} match {_q(rule.b)} in "
            f"pixel space; {_q(rule.b)} should stay fixed."
        )
    if rule.op == REWARD:
        return f"Reward {_q(rule.a)} (not implemented)."
    return f"{rule.op} {rule.raw}"


def _phrase_items(phrase: str) -> list:
    """Split a phrase into ('replace', a, b) or ('rule', Rule) items
    without expanding ``~``, so the note can talk about replace as one op."""
    items = []
    for part in phrase.split("|"):
        part = part.strip()
        if not part:
            continue
        concept = part.split(":", 1)[0]
        if "~" in concept:
            source, target = concept.split("~", 1)
            items.append(("replace", source.strip(), target.strip()))
            continue
        rule = parse_rule(part)
        if rule is not None:
            items.append(("rule", rule))
    return items


def describe_phrase(phrase: str, stage: str | None = None) -> str:
    """Human-readable intent for a phrase, for grid captions."""
    sentences: list[str] = []
    if stage == "encoder":
        sentences.append(
            "Encoder-only: trains a text-encoder LoRA; the DiT is untouched."
        )
    elif stage == "both":
        sentences.append("Two-stage: text-encoder LoRA first, then the DiT.")

    items = _phrase_items(phrase)
    i = 0
    while i < len(items):
        kind = items[i][0]
        if kind == "replace":
            _, source, target = items[i]
            sentences.append(_describe_replace(source, target))
            i += 1
            continue
        rule = items[i][1]
        nxt = items[i + 1][1] if i + 1 < len(items) and items[i + 1][0] == "rule" else None
        if (
            nxt is not None
            and rule.op == ORTHOGONAL
            and nxt.op == ORTHOGONAL
            and rule.alpha < 0
            and nxt.alpha < 0
            and rule.a == nxt.b
            and rule.b == nxt.a
        ):
            sentences.append(
                f"Blend {_q(rule.a)} and {_q(rule.b)} toward each other "
                f"(negative % aligns both ways; % only trains its right-hand side)."
            )
            i += 2
            continue
        if rule.op == FREEZE and rule.a == rule.b and rule.a:
            names = [rule.a]
            j = i + 1
            while j < len(items) and items[j][0] == "rule":
                other = items[j][1]
                if other.op == FREEZE and other.a == other.b and other.a:
                    names.append(other.a)
                    j += 1
                    continue
                break
            if len(names) == 1:
                sentences.append(_describe_rule(rule))
            else:
                listed = ", ".join(_q(n) for n in names[:-1]) + f" and {_q(names[-1])}"
                sentences.append(
                    f"Freeze {listed} so they stay as the original model "
                    f"would draw them."
                )
            i = j
            continue
        sentences.append(_describe_rule(rule))
        i += 1

    if not sentences:
        return "No rules in this phrase."
    return " ".join(sentences)


def materialize(rules: list[Rule], random_prompt: str | None = None) -> list[Rule]:
    """Substitute ``{random_prompt}`` into a parsed rule list. Rules without
    the placeholder are returned unchanged."""
    out = []
    for rule in rules:
        if rule.needs_random_prompt:
            if random_prompt is None:
                raise DSLError(
                    f"rule {rule.raw!r} needs a random prompt but none was given"
                )
            rp = sanitize_prompt(random_prompt)
            rule = replace(
                rule,
                a=rule.a.replace(RANDOM_PROMPT, rp),
                b=rule.b.replace(RANDOM_PROMPT, rp),
            )
        out.append(rule)
    return out
