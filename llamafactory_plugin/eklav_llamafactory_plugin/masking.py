"""
Sub-turn loss masking for LLaMA-Factory, ported from the Eklav reference
implementation's monkey-patch (see this repo's README for context).

LLaMA-Factory's stock `mask_history` only masks at whole-turn granularity.
Eklav needs to mask up to (or around) a text marker WITHIN a single turn's
response, so both patches here replace
`llamafactory.data.processor.supervised.SupervisedDatasetProcessor._encode_data_example`
with a wrapper that runs the stock encoder first (producing normal
full-loss input_ids/labels), then re-masks part of the response span.

Matching is done on DECODED text, not raw token ids: a marker's token-id
sequence when tokenized standalone can differ from how the same characters
tokenize once embedded in surrounding text (BPE merges are context
dependent). This was empirically confirmed on GLM-Z1-9B's tokenizer, where
"</think>" tokenizes differently depending on what follows it. Decoding and
mapping character offsets back to token indices via incremental decode is
immune to this and works identically across tokenizers.
"""

import os

IGNORE_INDEX = -100

_ANSWER_ONLY_ENV = ("EKLAV_ANSWER_ONLY_AFTER_TOKEN", "FORGE_ANSWER_ONLY_AFTER_TOKEN")
_THINK_MASK_ENV = ("EKLAV_THINK_CONTENT_MASK", "FORGE_THINK_CONTENT_MASK")

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

_applied = False


def _first_set(names):
    for name in names:
        val = os.environ.get(name)
        if val:
            return val
    return None


def _find_marker_end(tokenizer, input_ids, labels, marker):
    response_start = next((i for i, lab in enumerate(labels) if lab != IGNORE_INDEX), None)
    if response_start is None:
        return None
    span_ids = input_ids[response_start:]
    full_decoded = tokenizer.decode(span_ids)
    marker_pos = full_decoded.rfind(marker)
    if marker_pos == -1:
        return None
    target_len = marker_pos + len(marker)
    for i in range(1, len(span_ids) + 1):
        if len(tokenizer.decode(span_ids[:i])) >= target_len:
            return response_start + i
    return None


def _char_pos_to_token_idx(tokenizer, span_ids, char_target):
    for i in range(1, len(span_ids) + 1):
        if len(tokenizer.decode(span_ids[:i])) >= char_target:
            return i
    return len(span_ids)


def apply():
    """Idempotently apply both patches. Each is a no-op unless its env var
    is set at the time the patched method actually runs (checked per call,
    not just at import time, so tests/multi-config processes can toggle it)."""
    global _applied
    if _applied:
        return
    _applied = True

    try:
        from llamafactory.data.processor import supervised as sup
    except ImportError:
        return

    orig_encode = sup.SupervisedDatasetProcessor._encode_data_example

    def patched_encode(self, prompt, response, system, tools, images, videos, audios):
        input_ids, labels = orig_encode(self, prompt, response, system, tools, images, videos, audios)

        answer_only_marker = _first_set(_ANSWER_ONLY_ENV)
        if answer_only_marker:
            end = _find_marker_end(self.tokenizer, input_ids, labels, answer_only_marker)
            if end is not None:
                while end < len(input_ids) and self.tokenizer.decode([input_ids[end]]).strip() == "":
                    end += 1
                labels = [IGNORE_INDEX] * end + labels[end:]
            else:
                labels = [IGNORE_INDEX] * len(labels)
            return input_ids, labels

        if _first_set(_THINK_MASK_ENV) == "1":
            response_start = next((i for i, lab in enumerate(labels) if lab != IGNORE_INDEX), None)
            if response_start is None:
                return input_ids, labels
            span_ids = input_ids[response_start:]
            full_decoded = self.tokenizer.decode(span_ids)
            open_char = full_decoded.find(_THINK_OPEN)
            close_char = full_decoded.rfind(_THINK_CLOSE)
            if open_char == -1 or close_char == -1 or close_char <= open_char:
                return input_ids, labels
            content_start = _char_pos_to_token_idx(self.tokenizer, span_ids, open_char + len(_THINK_OPEN))
            content_end = _char_pos_to_token_idx(self.tokenizer, span_ids, close_char)
            if content_end <= content_start:
                return input_ids, labels
            labels = list(labels)
            for i in range(response_start + content_start, response_start + content_end):
                labels[i] = IGNORE_INDEX
            return input_ids, labels

        return input_ids, labels

    sup.SupervisedDatasetProcessor._encode_data_example = patched_encode
