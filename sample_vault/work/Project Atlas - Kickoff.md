---
tags: [work, atlas]
---
# Project Atlas — Kickoff

Goal: replace the single-primary order database with a **replicated cluster** that survives the loss of any one data centre.

## Decisions
- Use Raft for replication; 5 nodes across 3 regions.
- Writes need a majority quorum (3 of 5) before they are acknowledged.
- Reads served by the leader by default; follower reads allowed for analytics.

## Risks
- Leader election storms if network latency between regions spikes.
- Split brain if we get quorum configuration wrong during the migration.
- On-call team unfamiliar with consensus failure modes → schedule training.

Owner: Priya. Next review: [[Meeting - Q3 Planning]].
