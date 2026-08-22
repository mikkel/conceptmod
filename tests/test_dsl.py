import pytest

from conceptmod.dsl import (
    DSLError,
    ERASE,
    EXAGGERATE,
    FREEZE,
    ORTHOGONAL,
    PIXEL,
    REWARD,
    WRITE,
    Rule,
    describe_phrase,
    materialize,
    parse_phrase,
    parse_rule,
    sanitize_prompt,
)


def one(phrase):
    rules = parse_phrase(phrase)
    assert len(rules) == 1
    return rules[0]


class TestUnaryOps:
    def test_exaggerate(self):
        r = one("vibrant colors++")
        assert r.op == EXAGGERATE
        assert r.a == "vibrant colors"
        assert r.alpha == 1.0

    def test_erase(self):
        r = one("monochrome--")
        assert r.op == ERASE
        assert r.a == "monochrome"

    def test_exaggerate_needs_concept(self):
        with pytest.raises(DSLError):
            parse_phrase("++")

    def test_erase_needs_concept(self):
        with pytest.raises(DSLError):
            parse_phrase("--")


class TestWrite:
    def test_write_a_into_b(self):
        r = one("human=robot")
        assert r.op == WRITE
        assert (r.a, r.b) == ("human", "robot")

    def test_write_to_unconditional_prefix(self):
        r = one("=snow")
        assert r.op == WRITE
        assert (r.a, r.b) == ("", "snow")

    def test_write_to_unconditional_postfix(self):
        # README: "alpaca=" treats alpaca as a default concept
        r = one("alpaca=")
        assert r.op == WRITE
        assert (r.a, r.b) == ("", "alpaca")

    def test_bare_equals_is_error(self):
        with pytest.raises(DSLError):
            parse_phrase("=")


class TestFreeze:
    def test_freeze_pair(self):
        r = one("1woman#1woman")
        assert r.op == FREEZE
        assert (r.a, r.b) == ("1woman", "1woman")

    def test_bare_freeze_is_unconditional(self):
        r = one("#")
        assert r.op == FREEZE
        assert (r.a, r.b) == ("", "")

    def test_freeze_with_alpha(self):
        r = one("#:0.4")
        assert r.op == FREEZE
        assert r.alpha == 0.4


class TestOrthogonal:
    def test_orthogonal(self):
        r = one("cat%dog")
        assert r.op == ORTHOGONAL
        assert (r.a, r.b) == ("cat", "dog")

    def test_blend_negative_alpha(self):
        r = one("anime%hyperrealistic:-1.0")
        assert r.op == ORTHOGONAL
        assert r.alpha == -1.0

    def test_double_dash_negative_alpha_compat(self):
        # original tolerated "--0.1" meaning -0.1
        r = one("cat%dog:--0.1")
        assert r.alpha == -0.1


class TestPixelAndReward:
    def test_pixel(self):
        r = one("photo^painting")
        assert r.op == PIXEL
        assert (r.a, r.b) == ("photo", "painting")

    def test_reward_parses(self):
        r = one(";a beautiful sunset")
        assert r.op == REWARD
        assert r.a == "a beautiful sunset"


class TestOptions:
    def test_alpha(self):
        r = one("boring--:0.5")
        assert r.alpha == 0.5

    def test_negative_alpha(self):
        r = one("cat%dog:-0.1")
        assert r.alpha == -0.1

    def test_keyword_option(self):
        r = one("cat++:guidance=2.5")
        assert r.options == {"guidance": 2.5}
        assert r.alpha == 1.0

    def test_alpha_and_keyword(self):
        r = one("cat++:0.4:guidance=2.5")
        assert r.alpha == 0.4
        assert r.options == {"guidance": 2.5}

    def test_bad_alpha_raises(self):
        with pytest.raises(DSLError):
            parse_phrase("cat++:banana")


class TestPhrase:
    def test_multi_rule(self):
        rules = parse_phrase("vibrant colors++|boring--")
        assert [r.op for r in rules] == [EXAGGERATE, ERASE]

    def test_whitespace_stripped(self):
        rules = parse_phrase(" vibrant colors++ | boring-- ")
        assert rules[0].a == "vibrant colors"
        assert rules[1].a == "boring"

    def test_empty_rules_skipped(self):
        rules = parse_phrase("cat++||")
        assert len(rules) == 1

    def test_at_sign_ignored(self):
        r = one("@cat++")
        assert r.op == EXAGGERATE
        assert r.a == "cat"

    def test_empty_phrase_raises(self):
        with pytest.raises(DSLError):
            parse_phrase("")

    def test_original_example(self):
        rules = parse_phrase("#:0.4|human=robot:0.8|robot%human:-0.1")
        assert [r.op for r in rules] == [FREEZE, WRITE, ORTHOGONAL]
        assert rules[0].alpha == 0.4
        assert (rules[1].a, rules[1].b, rules[1].alpha) == ("human", "robot", 0.8)
        assert (rules[2].a, rules[2].b, rules[2].alpha) == ("robot", "human", -0.1)


class TestReplaceMacro:
    def test_expansion_matches_original(self):
        rules = parse_phrase("cat~dog")
        # original: target++, prefix=target, target%prefix:-lambda
        assert [r.op for r in rules] == [EXAGGERATE, WRITE, ORTHOGONAL]
        assert (rules[0].a, rules[0].alpha) == ("dog", 0.2)
        assert (rules[1].a, rules[1].b, rules[1].alpha) == ("cat", "dog", 0.4)
        assert (rules[2].a, rules[2].b, rules[2].alpha) == ("dog", "cat", -0.1)

    def test_custom_lambda(self):
        rules = parse_phrase("cat~dog:0.2")
        assert rules[0].alpha == 0.4
        assert rules[1].alpha == 0.8
        assert rules[2].alpha == -0.2

    def test_replace_mixes_with_other_rules(self):
        rules = parse_phrase("#|cat~dog")
        assert [r.op for r in rules] == [FREEZE, EXAGGERATE, WRITE, ORTHOGONAL]


class TestRandomPrompt:
    def test_detection(self):
        r = one("final boss%{random_prompt}:-0.1")
        assert r.needs_random_prompt

    def test_materialize(self):
        rules = parse_phrase("final boss%{random_prompt}:-0.1|cat++")
        out = materialize(rules, random_prompt="a portrait, 4k")
        assert out[0].b == "a portrait, 4k"
        assert not out[0].needs_random_prompt
        assert out[1] is rules[1]

    def test_materialize_sanitizes(self):
        rules = parse_phrase("x%{random_prompt}")
        out = materialize(rules, random_prompt="a:b|c=d%e")
        assert ":" not in out[0].b
        assert "|" not in out[0].b
        assert "=" not in out[0].b
        assert "%" not in out[0].b

    def test_materialize_requires_prompt(self):
        rules = parse_phrase("x%{random_prompt}")
        with pytest.raises(DSLError):
            materialize(rules)


def test_sanitize_prompt():
    assert sanitize_prompt("a:b") == "a-b"
    assert "percent" in sanitize_prompt("50%")
    assert "equals" in sanitize_prompt("a=b")


class TestDescribePhrase:
    def test_exaggerate(self):
        note = describe_phrase("vibrant colors++")
        assert "Exaggerate" in note
        assert "vibrant colors" in note

    def test_erase(self):
        note = describe_phrase("monochrome--")
        assert "Erase" in note
        assert "monochrome" in note

    def test_erase_guidance_zero_is_neutralize(self):
        note = describe_phrase("red--:guidance=0")
        assert "Neutralize" in note
        assert "antipode" in note
        assert "red" in note

    def test_write_uncond(self):
        note = describe_phrase("=snow")
        assert "empty prompt" in note
        assert "snow" in note

    def test_write(self):
        note = describe_phrase("cat=dog")
        assert "cat" in note and "dog" in note
        assert "Write" in note
        assert "remap" in note.lower()
        assert "~" in note

    def test_replace_not_expanded(self):
        note = describe_phrase("cat~dog:0.35")
        assert "Replace" in note
        assert "swap recipe" in note
        assert "cat=dog" in note
        assert "Exaggerate" not in note
        assert "Orthogonal" not in note

    def test_symmetric_blend(self):
        note = describe_phrase(
            "anime%hyperrealistic:-3|hyperrealistic%anime:-3")
        assert note.count("Blend") == 1
        assert "anime" in note and "hyperrealistic" in note

    def test_orthogonal_with_grouped_freezes(self):
        note = describe_phrase("cat%dog:2|cat#cat|dog#dog:0.5")
        assert "Orthogonal" in note
        assert "strip" in note
        assert "Freeze" in note
        assert "cat" in note and "dog" in note

    def test_pixel(self):
        note = describe_phrase("a painting of a house^a photo of a house")
        assert "Pixel" in note
        assert "painting of a house" in note

    def test_random_prompt_regularizer(self):
        note = describe_phrase(
            "final boss++:0.4|final boss%{random_prompt}:-0.1")
        assert "Exaggerate" in note
        assert "random prompts" in note

    def test_stage_encoder(self):
        note = describe_phrase("vibrant colors++|monochrome--", stage="encoder")
        assert "Encoder-only" in note
        assert "Exaggerate" in note
        assert "Erase" in note

    def test_composite(self):
        note = describe_phrase("#:0.4|human=robot:0.8|robot%human:-0.1")
        assert "empty prompt" in note
        assert "human" in note and "robot" in note
