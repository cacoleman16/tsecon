# Governance

This document describes how decisions are made in `tsecon` (a working codename;
the public name is resolved before the first release). It is deliberately
lightweight and honest about the project's current stage: `tsecon` is pre-1.0,
under active development, and maintained by one person with contributions
welcome. As the community grows, this document is expected to grow with it — the
path toward shared governance is described below, and it is a genuine intention,
not a formality already in place.

## Roles

### Maintainer

The project currently has a single maintainer:

- **Chase Coleman** (<chasecoleman93@gmail.com>) — project lead and, at this
  stage, the sole person with commit and release authority.

The maintainer is responsible for reviewing and merging contributions, cutting
releases, arbitrating design decisions, upholding the
[Code of Conduct](CODE_OF_CONDUCT.md), and stewarding the direction laid out in
[ROADMAP.md](ROADMAP.md).

### Contributors

Anyone who opens an issue, proposes a change, improves the docs, adds a golden
fixture, or files a bug report is a contributor. Contributions do not require any
prior status. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get started and
the quality gates a change must pass.

## Decision-making

For now the project runs on a **BDFL-with-RFC** model — a benevolent-dictator
arrangement that keeps decisions fast at this early stage, tempered by a
lightweight RFC step for anything that touches the public API.

- **Everyday changes** (bug fixes, new estimators behind a validated golden
  target, docs, tests, internal refactors) are decided through normal pull-request
  review. If CI is green and the change meets the bar in
  [CONTRIBUTING.md](CONTRIBUTING.md), the maintainer merges it. Consensus in the
  PR thread is the default; the maintainer breaks ties.

- **Public-API changes** (adding, renaming, removing, or changing the signature
  or default behavior of any function in `bindings/python/tsecon.pyi`, or moving a
  method between the stability tiers below) require a **written proposal first** —
  open a GitHub issue labeled `rfc` describing the motivation, the proposed
  surface, the alternatives considered, and the migration/deprecation impact.
  This is because economists return to their paper code years later at revision
  time, and unannounced breakage is the documented way libraries in this space
  lose their users (see [ROADMAP.md](ROADMAP.md) §11). The RFC step exists to make
  API churn deliberate and visible, not to add ceremony to small changes.

- **Scope decisions** (what belongs in v1 versus a point release versus the
  contrib tier) are governed by the public tiering policy in
  [ROADMAP.md](ROADMAP.md) §6. New frontier methods are gated on a named
  validation target before work starts.

### Escalation

If a decision is contested:

1. Discuss it in the relevant issue or pull request. Most disagreements resolve
   here once the trade-offs are written down.
2. If it does not resolve, the maintainer makes the final call and records the
   reasoning in the thread so the decision is auditable later.

There is currently no separate steering committee or voting body — with a single
maintainer, one would be theater. The escalation path becomes a real vote once
there is more than one maintainer (see below).

## API-stability tiers

The public API carries per-symbol stability labels, documented alongside each
method. These tiers are the contract that tells you how much churn to expect
before you build on a function (policy from [ROADMAP.md](ROADMAP.md) §11):

- **Stable** — the signature and documented behavior will not change without a
  deprecation cycle. Post-1.0, stable symbols follow strict
  [SemVer](https://semver.org/); breaking them requires a major-version bump and a
  two-release deprecation window with a shim.
- **Provisional** — shipped and usable, validated against a golden target, but
  the surface may still change in a minor release as it settles. Most of the
  frontier methods live here first so they can ship without prematurely freezing
  their APIs.
- **Experimental** — available for feedback and expected to change or be removed;
  do not build production or replication code on an experimental symbol without
  pinning an exact version.

Until 1.0 the whole library is pre-release: per the pre-1.0 policy in
[CHANGELOG.md](CHANGELOG.md) and [ROADMAP.md](ROADMAP.md), minor versions may
contain breaking changes and patch versions are fixes only. The tiers still tell
you *relatively* how settled a given function is. A change to any default is never
silent — it goes through a deprecation cycle and is recorded in the changelog with
before/after numbers on a reference dataset.

CI enforces part of this contract already: a public symbol cannot disappear from
the type stub without the stub-sync guard failing (see
[CONTRIBUTING.md](CONTRIBUTING.md)). Full deprecation-shim enforcement lands with
the v1.0 API freeze.

## Path toward shared governance

The single-maintainer model is a function of the project's age, not a permanent
design. The **intended** progression, as sustained contribution appears, is:

1. **Committers** — contributors with a track record of good reviews and merged
   work are granted commit rights to reduce the review bottleneck, while the
   maintainer retains release authority.
2. **Multiple maintainers** — as the committer group matures, additional
   maintainers are added with full release authority, and the escalation path
   above turns into a documented lazy-consensus / voting process among
   maintainers rather than a single final call.
3. **A written governance charter** — once there is a real maintainer team, this
   document is replaced by a fuller charter covering maintainer onboarding and
   offboarding, decision quorums, and conflict resolution.

Longer term, the project intends to pursue the trappings of a citable, durable
scientific library — a JOSS paper and per-release DOIs, and eventually a neutral
institutional home for governance and continuity. These are stated **intentions
and aspirations, not current facts**: as of this writing the project has no
external funding, no fiscal sponsor, and no institutional affiliation, and this
document does not claim any. If and when such arrangements are established they
will be documented here explicitly.

## Code of Conduct

Participation in the project is governed by the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). Enforcement reports
go to <chasecoleman93@gmail.com>.
