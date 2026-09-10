"""Primary → fallback failover for agent invocations.

The failover rebuilds the graph with the fallback model but reuses the same
checkpointer-backed thread, so conversation state (including a pending resume)
carries over seamlessly.
"""

from app.services.agent import provider


async def ainvoke_with_failover(build_fn, payload, config, prefix_messages=None):
    """Invoke a graph, failing over to the fallback model on capacity errors.

    prefix_messages are prepended to a dict payload's messages (e.g. one-time
    context on a fresh thread). Returns (result, model_used, graph).
    LLMNotConfigured and non-capacity errors propagate untouched.
    """
    if prefix_messages and isinstance(payload, dict):
        payload = {**payload, "messages": [*prefix_messages, *payload.get("messages", [])]}
    graph = build_fn()
    try:
        return await graph.ainvoke(payload, config), "primary", graph
    except Exception as exc:
        if not provider.is_capacity_error(exc) or not provider.has_fallback():
            raise
        fallback_graph = build_fn(model=provider.get_fallback_model())
        result = await fallback_graph.ainvoke(payload, config)
        return result, "fallback", fallback_graph
