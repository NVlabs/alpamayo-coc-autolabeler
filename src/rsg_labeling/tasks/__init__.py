"""Tasks: interpreters over the compiled RSG.

The RSG is the IR: the compiler (env sources -> op passes) produces it once; tasks vary
over it and never alter it. Each task is one package interpreting the graph into what the
task defines, such as intervals, derived moments or an export format. Interpreters consume
and cite the IR; any graph write still flows through the compiler's op gateway.
"""
