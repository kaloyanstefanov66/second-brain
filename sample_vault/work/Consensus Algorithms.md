---
tags: [engineering, distributed-systems]
---
# Consensus Algorithms

How does a group of unreliable machines agree on one value?

**Raft** splits the problem into leader election, log replication and safety. Nodes start as followers; if they hear nothing from a leader before a randomized timeout, they become candidates and request votes. A candidate that collects votes from a *majority* becomes leader. Because any two majorities overlap, two leaders can never be elected for the same term.

**Paxos** reaches the same guarantees with proposers, acceptors and learners, but is notoriously hard to understand.

Key ideas:
- Quorums: decisions need agreement from more than half the participants.
- Randomized timeouts break symmetry so that candidates don't keep splitting the vote.
- Commitment is irreversible once a quorum has accepted an entry.
