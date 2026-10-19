# Skills audit

Audited on **2026-10-19**. Skills are instruction packs, not trusted executable code. The blueprint's requested external skills were not silently substituted with unverified repositories.

## OmniRush skills available locally

| Skill | Source/location | Version | Relevance | Actually used |
| --- | --- | --- | --- | --- |
| `omnirush-setup` | OmniRush config: `best-practices/omnirush-setup/SKILL.md` | app-provided, no commit exposed | repository/runtime audit and check discovery | Yes |
| `omnirush-feature` | OmniRush config: `best-practices/omnirush-feature/SKILL.md` | app-provided | incremental implementation and acceptance criteria | Yes |
| `omnirush-test` | OmniRush config: `best-practices/omnirush-test/SKILL.md` | app-provided | behavior tests and honest reporting | Yes |
| `omnirush-build` / `omnirush-debug` | OmniRush config: `best-practices/` | app-provided | optional install/debug guides | Available, not loaded |
| `omnirush-performance` | OmniRush config: `best-practices/omnirush-performance/SKILL.md` | app-provided | bounded workload and device evidence | Yes |
| `omnirush-review` | OmniRush config: `best-practices/omnirush-review/SKILL.md` | app-provided | technical audit and actionable regression checks | Yes |
| `omnirush-handoff` | OmniRush config: `best-practices/omnirush-handoff/SKILL.md` | app-provided | resumable reports and precise next step | Yes |
| `omnirush-refactor` / `omnirush-swarm` | OmniRush config skill catalog | app-provided | optional refactor / 3+ agent workflow | Available, not loaded |

## Blueprint-requested external skills

The blueprint names `ml-pipeline-workflow` from `wshobson/agents`, `python-testing-patterns` from `rmyndharis/antigravity-skills`, and `walk-forward-validation` from `agiprolabs/claude-trading-skills`. The prescribed `npx skills --help` probe could not run because this host has no Node.js/npm (`node: command not found`, `npm: command not found`). No external skill was installed, no downloaded script was executed, and there is no source repository/version/path to claim. Implementation proceeds with the blueprint and the local audited OmniRush skills.

## Safety note

No provider credentials, external agent API, paid service, or unreviewed installer was used. If these skills are later installed, their full `SKILL.md` files and exact repository commit must be inspected and added here before use.

The local instruction packs were read in full through the OmniRush skill tool. Their config root is the application's local `.config/omnirush/` directory; it is not a runtime dependency copied into the repository. Two bounded implementation agents worked in separate local Git worktrees, then their actual commits were integrated and tested together. The lead performed the environment/data-source investigation, integration tests, training, review and documentation. No other external LLM/coding API was introduced into the application.
