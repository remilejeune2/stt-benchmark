"""Tests for the semantic WER judge's providers."""

import pytest

from stt_benchmark.config import reset_config
from stt_benchmark.evaluation.semantic_wer import (
    DEFAULT_JUDGE_MODEL,
    SemanticWEREvaluator,
    default_judge_effort,
    same_words,
)


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    reset_config()
    yield
    reset_config()


def make_evaluator(tmp_path, **kwargs) -> SemanticWEREvaluator:
    return SemanticWEREvaluator(db_path=tmp_path / "results.db", **kwargs)


def test_anthropic_is_the_default_provider(tmp_path):
    evaluator = make_evaluator(tmp_path)

    assert evaluator.provider == "anthropic"
    assert str(evaluator.client.base_url).startswith("https://api.anthropic.com")
    assert evaluator.client.api_key == "anthropic-key"
    assert evaluator.api_model == DEFAULT_JUDGE_MODEL


def test_anthropic_judge_label_is_unchanged(tmp_path):
    evaluator = make_evaluator(tmp_path, effort="medium")

    assert evaluator.judge.startswith(f"{DEFAULT_JUDGE_MODEL} effort=medium repeats=1 rev=")
    assert "via=" not in evaluator.judge


def test_openrouter_uses_its_anthropic_compatible_api(tmp_path):
    evaluator = make_evaluator(tmp_path, provider="openrouter")

    assert str(evaluator.client.base_url).rstrip("/") == "https://openrouter.ai/api"
    assert evaluator.client.api_key is None
    assert evaluator.client.auth_token == "openrouter-key"


@pytest.mark.parametrize(
    ("model", "api_model"),
    [
        ("claude-sonnet-5-5", "anthropic/claude-sonnet-5.5"),
        ("claude-opus-5-5", "anthropic/claude-opus-5.5"),
        ("claude-sonnet-5", "anthropic/claude-sonnet-5"),
    ],
)
def test_openrouter_names_anthropic_models_its_own_way(tmp_path, model, api_model):
    evaluator = make_evaluator(tmp_path, model=model, provider="openrouter")

    assert evaluator.api_model == api_model


def test_openrouter_requests_send_its_model_id_with_thinking(tmp_path):
    evaluator = make_evaluator(tmp_path, effort="high", provider="openrouter")

    request = evaluator._request([{"role": "user", "content": "."}], max_tokens=10)

    assert request["model"] == "anthropic/claude-sonnet-5.5"
    assert request["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert request["output_config"] == {"effort": "high"}


def test_openrouter_scores_are_labelled_as_another_judge(tmp_path):
    anthropic_judge = make_evaluator(tmp_path, effort="high").judge
    openrouter_judge = make_evaluator(tmp_path, effort="high", provider="openrouter").judge

    assert openrouter_judge == f"{anthropic_judge} via=openrouter"


def test_openrouter_requires_its_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    reset_config()

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        make_evaluator(tmp_path, provider="openrouter")


def test_openrouter_does_not_need_an_anthropic_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    reset_config()

    assert make_evaluator(tmp_path, provider="openrouter").provider == "openrouter"


def test_unknown_provider_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Unknown judge provider"):
        make_evaluator(tmp_path, provider="bedrock")


def test_default_effort_depends_on_the_provider():
    assert default_judge_effort("anthropic") == "medium"
    assert default_judge_effort("openrouter") == "high"


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        ("I'd like to book a table.", "i'd like to book a table"),
        ("Yes, that's right — thanks!", "Yes that’s right, thanks."),
    ],
)
def test_same_words_ignores_case_and_punctuation(reference, hypothesis):
    assert same_words(reference, hypothesis)


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        ("I'd like to book a table.", "I would like to book a table."),
        ("It costs 2,000 dollars.", "It costs 2000 dollars."),
        ("Book a table.", "Book a table for two."),
        ("Book a table.", "Book table."),
    ],
)
def test_same_words_leaves_other_differences_to_the_judge(reference, hypothesis):
    assert not same_words(reference, hypothesis)


async def test_exact_match_is_scored_without_the_judge(tmp_path):
    evaluator = make_evaluator(tmp_path)

    async def no_call(*args, **kwargs):
        raise AssertionError("the judge was called")

    evaluator._api_call = no_call
    result, trace = await evaluator.evaluate("I'd like a table, please.", "i'd like a table please")

    assert result["wer"] == 0.0
    assert result["reference_words"] == 6
    assert trace.num_turns == 0
