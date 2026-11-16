"""Attention "set" without a GPU: the scheduler's job for a set sequence and the key ranges pack gives it.

    python tests/test_set_attention.py

A set sequence is a state then questions packed after it. Its job's prefix is the whole sequence, with a copy of its
last token as the branch nothing reads (as a "blocks" sequence of its own), and pack gives every prefix row two sets
of key ranges: causal ones for the layers that read the sequence one way, and set ones for the full-attention layers,
where a state token sees the whole sequence and a question's token the state and its own question.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.formats.base import FINAL, Plan, Read
from flide.runtime.pack import Job, pack
from flide.runtime.schedule import split_plan

fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} {name}{'' if ok else f' ({detail})'}")


def main():
    seq = list(range(100, 110))   # state 0..3, question 1 at 4..6, question 2 at 7..9
    plan = Plan([], [seq], (Read("h", FINAL, (6, 9)),), attention="set", segments=[(4, 7, 10)])
    (piece,) = split_plan(0, plan, budget=1 << 16)
    job = piece.job
    check("the sequence is the job's prefix", job.prefix == seq and job.suffixes == [[109]])
    check("the job keeps the segments", job.segments == (4, 7, 10), job.segments)
    check("reads from the start stay put", job.reads[0][0].positions == (6, 9), job.reads[0][0].positions)

    b = pack([job], "cpu")
    kr, sr = b.key_ranges.tolist(), b.set_ranges.tolist()
    check("prefix rows read causally in key_ranges", all(kr[t][:2] == [0, t + 1] for t in range(10)), kr[:10])
    check("a state row sees the whole sequence", all(sr[t] == [0, 10, 0, 0] for t in range(4)), sr[:4])
    check("question 1 sees the state and itself", all(sr[t] == [0, 4, 4, 7] for t in range(4, 7)), sr[4:7])
    check("question 2 sees the state and itself", all(sr[t] == [0, 4, 7, 10] for t in range(7, 10)), sr[7:10])
    check("the copied last token keeps its causal ranges", sr[10] == kr[10], (sr[10], kr[10]))
    check("reads land on the slots", b.reads[FINAL].tolist() == [6, 9], b.reads[FINAL].tolist())

    other = pack([Job(seq[:-1], [seq[-1:]], [(Read("h", FINAL, (-1,)),)])], "cpu")
    check("no set job, no set ranges", other.set_ranges is None)

    two = pack([Job([1, 2, 3], [[4]], [(Read("h", FINAL, (-1,)),)]), job], "cpu")
    sr2 = two.set_ranges.tolist()
    check("a set job after another job reads its own rows", sr2[3] == [3, 13, 0, 0] and sr2[3 + 7] == [3, 7, 10, 13],
          (sr2[3], sr2[10]))
    check("the other job's rows keep their ranges", sr2[:3] == two.key_ranges.tolist()[:3])

    for bad, why in (((4, 7, 9), "segments short of the prefix"), ((4, 4, 10), "an empty question")):
        try:
            pack([Job(seq, [seq[-1:]], [(Read("h", FINAL, (6,)),)], segments=bad)], "cpu")
            check(f"refuses {why}", False, "accepted")
        except ValueError:
            check(f"refuses {why}", True)

    print("ALL PASS" if fails == 0 else f"{fails} FAILED")
    return fails


if __name__ == "__main__":
    sys.exit(main())
