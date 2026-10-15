"""Applies the Eklav answer-only loss mask in every Python process that finds this
directory on PYTHONPATH (see eklav.lf_plugin.startup_dir)."""

try:
    from eklav.lf_plugin import apply

    apply()
except ImportError:
    pass
