"""Supervisor: sub-agents get the conversation history and the caller's context."""
import contextvars

from src.agents.supervisor_agent import SupervisorAgent

MARK = contextvars.ContextVar("mark", default=None)


class RecordingAgent:
    def __init__(self):
        self.inputs = []
        self.seen_mark = None

    def run(self, input_data):
        self.inputs.append(input_data)
        self.seen_mark = MARK.get()
        input_data.get("on_event", lambda e: None)({"type": "stage", "stage": "x"})
        return {"success": True, "answer": "ok", "sources": []}


def test_supervisor_passes_history_and_events_to_every_agent():
    a, b = RecordingAgent(), RecordingAgent()
    sup = SupervisorAgent(agents={"sport": a, "tech": b})
    events = []
    out = sup.run({"query": "q", "active_agents": ["sport", "tech", "sport"],
                   "history": "User: earlier question", "on_event": events.append})
    assert out["success"] and set(out["agent_responses"]) == {"sport", "tech"}
    for agent in (a, b):
        assert agent.inputs == [{"query": "q", "history": "User: earlier question", "on_event": events.append}]
    assert len(events) == 2


def test_supervisor_threads_inherit_context():
    """Regression: the stream writer / tracing live in contextvars, lost in plain threads."""
    a = RecordingAgent()
    sup = SupervisorAgent(agents={"sport": a})
    token = MARK.set("from-caller")
    try:
        sup.run({"query": "q", "active_agents": ["sport"]})
    finally:
        MARK.reset(token)
    assert a.seen_mark == "from-caller"
    assert a.inputs == [{"query": "q"}]  # nothing extra when no history / on_event
