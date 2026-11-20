"""RSG graph engines, labeling tasks, and shared support.

``streaming_rsg`` compiles temporal graphs through an environment contract and op
gateway. Task interpreters live under ``tasks``; shared helpers live under
``common`` and ``infra``. A distribution selects the engines and tasks it
provides; consult its release README for that supported selection.

Import each component from its owning package; this namespace re-exports nothing.
"""
