# MLign — STATE (compaction-proof summary)

Budget: UNKNOWN / share 9.1% (tier low) / THROTTLE (Stand 2026-08-16T21:10+02:00)
Rule (verwaltung, 2026-08-16): run `credits budget --agent mlign --json` at the
start of every cycle; RUN=normal, THROTTLE=lean (≤3 agents, sonnet/haiku,
no adversarial fan-outs), HALT=finish cleanly, rewrite STATE.md, wake ≥3600s.
Wakeups ≥1200s unless waiting on something external.

## Where the mission stands
- HEADLINE BANKED: models/mlign-v1.pt (v5real e21) beats DualDTW on the untouched
  nASAP holdout: 0.9878 vs 0.9852 (65W/2T/17L, p<1e-7), best on ins+del; e23
  confirms (.9874). docs/RESULTS.md has every table (regenerate with
  scripts/make_results.py). README has quickstart. All committed locally.
- Also banked: Bar C folded-score .992 pooled (symbolic-only first), Batik
  parity-or-better, Bar B tie, TheGlueNote beaten everywhere.
- Open lever: Beethoven long-sonata regime (−.002 vs DualDTW). Candidates:
  matchability-aware leftover decode; v6real2 bracket (e23/24/25) dev-long
  evals were running detached → eval/results/devlong-v6e02{3,4,5}.json.
- Cluster (uds:/tmp/cc-socks/29262.sock, socket rotates — ListAgents): idle;
  corpus + realgt/realgt2 all synced; runs v5real/v6real2 complete + local.
- Peers: meico-ts ornamentation + exaggeration merged upstream + integrated;
  mpmify owed the milestone relay (their session was down at send time).

## Operating mode under THROTTLE
No subagents (none needed — this is a supervision loop). Local evals only,
niced, one at a time. Cluster jobs are not budget-relevant (compute is theirs)
but each cycle costs my context — cycles ≥1 h.
