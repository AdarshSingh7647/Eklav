"""
Answer-only loss masking for LLaMA-Factory (paper Sec. 3.3 and Appendix A.3).

LLaMA-Factory's stock `mask_history` masks whole turns. The answer-only objective
of NAIVE-SFT, EKLAV and MASK-ONLY needs a sub-turn mask: everything in the
assistant response up to and including a closing reasoning tag is excluded from
the loss, whether it is teacher reasoning or the empty reasoning block a chat
template inserts into a response that has none. `apply()` wraps
`SupervisedDatasetProcessor._encode_data_example` to run the stock encoder first
and then re-mask the response span.

The marker is found in DECODED text, not by token ids: BPE merges are
context-dependent, so a marker's token ids when tokenized alone can differ from
how the same characters tokenize inside a response. Decoding incrementally and
mapping the marker's character offset back to a token index is tokenizer-agnostic.
The search is restricted to the assistant response, because the prompt may itself
mention reasoning delimiters.

Enabled per process by EKLAV_ANSWER_ONLY_AFTER_TOKEN="</think>" (see eklav.methods);
without it the patch is a no-op, so STD-CoT trains on the full response.
"""

import os
import warnings

IGNORE_INDEX = -100
ENV_VAR = "EKLAV_ANSWER_ONLY_AFTER_TOKEN"

_applied = False
_warned_missing_marker = False


def find_marker_end(tokenizer, input_ids, labels, marker):
    """Index just past the last `marker` in the response (None if absent)."""
    response_start = next((i for i, lab in enumerate(labels) if lab != IGNORE_INDEX), None)
    if response_start is None:
        return None
    span_ids = input_ids[response_start:]
    marker_pos = tokenizer.decode(span_ids).rfind(marker)
    if marker_pos == -1:
        return None
    target_len = marker_pos + len(marker)
    for i in range(1, len(span_ids) + 1):
        if len(tokenizer.decode(span_ids[:i])) >= target_len:
            return response_start + i
    return None


def answer_only_labels(tokenizer, input_ids, labels, marker):
    """`labels` with everything through `marker` (and whitespace after it) masked.
    An example whose response has no marker (e.g. truncated by the sequence
    cutoff) is masked entirely rather than trained on a partial trace."""
    global _warned_missing_marker
    end = find_marker_end(tokenizer, input_ids, labels, marker)
    if end is None:
        if not _warned_missing_marker:
            _warned_missing_marker = True
            warnings.warn(
                f"eklav: response without {marker!r} found; such examples contribute no loss "
                "(typically a response cut off by cutoff_len)."
            )
        return [IGNORE_INDEX] * len(labels)
    while end < len(input_ids) and tokenizer.decode([input_ids[end]]).strip() == "":
        end += 1
    return [IGNORE_INDEX] * end + list(labels[end:])


def apply():
    """Idempotently patch LLaMA-Factory's supervised encoder. A no-op when
    llamafactory is not installed."""
    global _applied
    if _applied:
        return
    _applied = True

    try:
        from llamafactory.data.processor import supervised
    except ImportError:
        return

    stock_encode = supervised.SupervisedDatasetProcessor._encode_data_example

    def encode_with_answer_only_mask(self, prompt, response, system, tools, images, videos, audios):
        input_ids, labels = stock_encode(self, prompt, response, system, tools, images, videos, audios)
        marker = os.environ.get(ENV_VAR)
        if marker:
            labels = answer_only_labels(self.tokenizer, input_ids, labels, marker)
        return input_ids, labels

    supervised.SupervisedDatasetProcessor._encode_data_example = encode_with_answer_only_mask
