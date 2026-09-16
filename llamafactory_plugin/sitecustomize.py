"""
Drop this file's directory on PYTHONPATH (or set PYTHONPATH to include
llamafactory_plugin/) so Python auto-imports it at interpreter startup for
every process, including an `llamafactory-cli train ...` subprocess -- no
code changes to any training script needed. Equivalent to
`import eklav_llamafactory_plugin` at the top of a training entrypoint; see
this package's README for both usage options.
"""

try:
    import eklav_llamafactory_plugin  # noqa: F401
except ImportError:
    pass
