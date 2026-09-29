# Dataset and Teacher Policy

External models are teachers, not the Cerebro model itself.

## Required provenance

Every training record retains source teacher/provider, model identifier, license/terms and dataset fingerprint.

## Quality

Records are validated, deduplicated and scored before training. Quality heuristics are conservative; benchmark results are the acceptance signal.

## Legal and terms review

provider-output is an internal provenance label, not a blanket license. Before a public model release, contributors must verify provider/model terms and applicable dataset licenses.

## Data contamination

Never put held-out benchmark prompts or answers into training data. Keep benchmark data separate and fingerprinted.

## Synthetic data

Synthetic data may be mixed with curated or human-authored data. More synthetic data is not automatically better. Use deduplication, diversity checks and held-out evaluation.

## Secrets

API keys must live in environment variables or the local secret file. Never commit, issue-post or embed credentials in dataset artifacts.
