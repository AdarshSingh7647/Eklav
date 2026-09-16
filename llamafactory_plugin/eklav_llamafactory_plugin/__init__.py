"""
Eklav sub-turn masking plugin for LLaMA-Factory.

Importing this package applies both monkey-patches unconditionally; each
patch's actual effect is gated at encode-time by its own env var, so simply
importing it (or dropping this package's directory on PYTHONPATH via
sitecustomize.py) is safe for training runs that need neither.

Env vars:
  EKLAV_ANSWER_ONLY_AFTER_TOKEN  -- e.g. "</think>". Masks loss on everything
    up to and including the LAST occurrence of this marker in the decoded
    response text, leaving only what follows it supervised. Used for
    passage reranking's Eklav method.
  EKLAV_THINK_CONTENT_MASK=1     -- masks loss on the text strictly between
    "<think>" and "</think>" in the decoded response, leaving the tags
    themselves and everything after "</think>" supervised. Used for math
    reasoning's Eklav method.

Both match text on DECODED strings, not raw token ids: BPE merges are
context-dependent, so a marker's token-id sequence when tokenized standalone
can differ from how the same characters tokenize once embedded in
surrounding text (confirmed on GLM-Z1-9B's tokenizer). Decoding first and
mapping character offsets back to token indices is tokenizer-agnostic.

For backward compatibility the legacy names FORGE_ANSWER_ONLY_AFTER_TOKEN and
FORGE_THINK_CONTENT_MASK are also honored (checked if the EKLAV_* var is unset).
"""

from . import masking

__all__ = ["masking"]

masking.apply()
