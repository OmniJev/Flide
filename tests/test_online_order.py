"""The online batcher's order of requests, without a GPU: arrival order without the session cache; with it, the
soonest done first, a request costing more than twice its round so far kept for the next round, and a long request
not overtaken by short ones that come after it would be done.

    python tests/test_online_order.py
"""
import queue
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.runtime.online import Batcher, _Item  # noqa: E402


class Plan:
    def __init__(self, name, tokens, cost):
        self.name, self.tokens, self.cost = name, tokens, cost


class Engine:
    def __init__(self, sessions):
        self.sessions = sessions

    def cost(self, plan):
        return plan.cost


def batcher(sessions, rate=0.0, max_tokens=8192):
    b = Batcher.__new__(Batcher)
    b.engine = Engine(sessions)
    b.window, b.gather, b._planning = 0.0, 0.002, 0
    b.max_round_tokens = max_tokens
    b._carry = None
    b.q = queue.Queue()
    b.ordered = sessions is not None
    b._waiting = []
    b._computed, b._busy, b._last = (int(rate), 1.0, 0.0) if rate else (0, 0.0, 0.0)
    return b


def put(b, name, t_in, tokens, cost):
    b.q.put(_Item(Plan(name, tokens, cost), None, t_in=t_in))


def rounds(b):
    out = []
    while not b.q.empty() or b._waiting or b._carry is not None:
        out.append([it.plan.name for it in (b._take_ordered() if b.ordered else b._take())])
    return out


def main():
    fails = 0

    def check(name, got, want):
        nonlocal fails
        ok = got == want
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}{'' if ok else f' (got {got}, want {want})'}")

    # without the session cache: arrival order, rounds cut at the token budget
    b = batcher(None)
    put(b, "long", 0.00, 4000, 4000)
    put(b, "agent1", 0.10, 3000, 300)
    put(b, "agent2", 0.12, 3500, 400)
    check("no sessions: arrival order", rounds(b), [["long", "agent1"], ["agent2"]])

    # with it, before any round has run (no rate yet): arrival order too
    b = batcher(object())
    put(b, "long", 0.00, 4000, 4000)
    put(b, "agent1", 0.10, 3000, 300)
    check("sessions, no rate yet: arrival order", rounds(b)[0][0], "long")

    # at 10 000 tokens a second the long one is done at 0.40 s, the agents at 0.13 and 0.16 s: they go first, and the
    # long one (more than twice what their round holds) waits for the next round
    b = batcher(object(), rate=10_000)
    put(b, "long", 0.00, 4000, 4000)
    put(b, "agent1", 0.10, 3000, 300)
    put(b, "agent2", 0.12, 3500, 400)
    check("short steps go ahead of a long request", rounds(b), [["agent1", "agent2"], ["long"]])

    # an agent step that comes after the long one would be done does not overtake it; it joins its round
    b = batcher(object(), rate=10_000)
    put(b, "long", 0.00, 4000, 4000)
    put(b, "agent", 0.50, 3000, 300)
    check("a long request is not overtaken once due", rounds(b), [["long", "agent"]])

    # the token budget still cuts a round
    b = batcher(object(), rate=10_000, max_tokens=6000)
    put(b, "agent1", 0.00, 3000, 300)
    put(b, "agent2", 0.00, 3500, 300)
    check("the round budget holds", rounds(b), [["agent1"], ["agent2"]])

    # requests of similar cost share a round
    b = batcher(object(), rate=10_000)
    put(b, "one1", 0.00, 2000, 2000)
    put(b, "one2", 0.01, 3000, 3000)
    check("similar requests share a round", rounds(b), [["one1", "one2"]])

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
