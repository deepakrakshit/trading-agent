# Skills audit

Audited on **2026-10-19**. Skills are instruction packs, not trusted executable code. The blueprint's requested external skills were not silently substituted with unverified repositories.

## OmniRush skills available locally

| Skill | Source/location | Version | Relevance | Actually used |
| --- | --- | --- | --- | --- |
| `omnirush-setup` | `/home/deepakrakshit/.config/omnirush/best-practices/omnirush-setup/SKILL.md` | workspace-provided | repository/runtime audit and check discovery | Yes |
| `omnirush-feature` | `/home/deepakrakshit/.config/omnirush/best-practices/omnirush-feature/SKILL.md` | workspace-provided | incremental feature implementation and acceptance criteria | Yes |
| `omnirush-test` | `/home/deepakrakshit/.config/omnirush/best-practices/omnirush-test/SKILL.md` | workspace-provided | behavior-focused tests and honest reporting | Yes |
| `omnirush-build` | `/home/deepakrakshit/.config/omnirush/best-practices/omnirush-build/SKILL.md` | workspace-provided | build/install diagnosis | Planned |
| `omnirush-debug` | `/home/deepakrakshit/.config/omnirush/best-practices/omnirush-debug/SKILL.md` | workspace-provided | failure reproduction and repair | Planned if needed |
| `omnirush-performance` | `/home/deepakrakshit/.config/omnirush/best-practices/omnirush-performance/SKILL.md` | workspace-provided | CPU/CUDA benchmark evidence | Planned |
| `omnirush-handoff` | `/home/deepakrakshit/.config/omnirush/best-practices/omnirush-handoff/SKILL.md` | workspace-provided | final resumable handoff | Planned |
| `omnirush-refactor` / `omnirush-review` | workspace-provided | workspace-provided | optional refactor/review gates | Available, not yet needed |

## Blueprint-requested external skills

The blueprint names `ml-pipeline-workflow` from `wshobson/agents`, `python-testing-patterns` from `rmyndharis/antigravity-skills`, and `walk-forward-validation` from `agiprolabs/claude-trading-skills`. The prescribed `npx skills --help` probe could not run because this host has no Node.js/npm (`node: command not found`, `npm: command not found`). No external skill was installed, no downloaded script was executed, and there is no source repository/version/path to claim. Implementation proceeds with the blueprint and the local audited OmniRush skills.

## Safety note

No provider credentials, external agent API, paid service, or unreviewed installer was used. If these skills are later installed, their full `SKILL.md` files and exact repository commit must be inspected and added here before use.
