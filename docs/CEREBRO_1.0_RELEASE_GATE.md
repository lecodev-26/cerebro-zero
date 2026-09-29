# Cerebro Zero 1.0 Release Gate

The final 1.0 release requires the unified public namespace, cognitive runtime, memory, tools, security, distribution, API-key scopes, owned model artifact, held-out benchmark, reproducible training, migration documentation and release checks.

## Training gate

A model checkpoint is not an official Cerebro Model 1.0 merely because it trains.

It must have:

1. reproducible configuration and seed;
2. dataset fingerprint and provenance;
3. tokenizer artifact;
4. train/validation metrics;
5. held-out benchmark results;
6. contamination checks;
7. checkpoint hash;
8. successful clean-install load test;
9. comparison against the declared baseline.

## Model lineage

Cerebro Model 1.0 -> 1.1 -> 1.2 ...

Teacher models are training inputs, not Cerebro model identities.

## Compute policy

CPU/Termux is supported for development and smoke training. GPU is optional and must be selected by an explicit backend. The runtime never assumes GPU availability.

## Contributor policy

Training runs with more compute are welcome. Report hardware, dataset fingerprint, configuration, metrics, benchmark results and checkpoint hash through a GitHub Issue or pull request. Do not upload credentials or private data.
