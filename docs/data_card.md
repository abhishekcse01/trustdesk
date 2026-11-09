# Data card

| Dataset | Source | Licence | How it is used | Committed? |
| --- | --- | --- | --- | --- |
| Policy corpus | [GitLab Handbook](https://handbook.gitlab.com/handbook/security/), security section | CC BY-SA 4.0 (per GitLab; confirm against the handbook repository LICENSE) | Fetched by `trustdesk fetch` into `data/raw/`, chunked and indexed locally | No. Re-fetched on demand; snapshot date is stored per page |
| Phase 1 questions | Written for this project | MIT (this repo) | Smoke test of retrieval and answering | Yes, `evals/phase1/questions.yaml` |
| CSA CAIQ v4 (phase 2) | [Cloud Security Alliance](https://cloudsecurityalliance.org/artifacts/consensus-assessments-initiative-questionnaire-caiq-v4-0) | Personal, non-commercial use; no redistribution | Golden-set questions, referenced by question ID only | No. Download it yourself; only IDs and our own reference answers are committed |

## Known limits

- The public handbook omits GitLab's internal-only policies, so some questions have no answer in the
  corpus. These are used to test abstention, not counted as retrieval failures.
- The corpus describes one company. Results do not show how TrustDesk performs on other companies'
  policy styles.
- Short quotes from the handbook appear in run reports as citations, with attribution.
