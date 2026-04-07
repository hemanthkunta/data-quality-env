---
title: data-quality-env
sdk: docker
emoji: 🚀
colorFrom: blue
colorTo: green
---
## OpenEnv Validation

This environment has been validated using OpenEnv:
```bash
openenv validate
# [OK] data-quality-env: Ready for multi-mode deployment
```

## Description
DataQualityEnv v2 is a budget-constrained, confidence-scored OpenEnv environment where an AI agent performs multi-step SQL auditing and optional fix verification.

Core loop:
- `reset` → environment generates seeded dirty datasets.
- `query` → agent investigates across one or more tables.
- `submit_report` → deterministic grading starts and fix phase unlocks.
- `fix_sql` → agent proposes corrective updates for bonus.

Novel mechanics:
- Query budget economy (10 credits).
- Confidence Brier grading.
- 4 tasks (easy to expert).
- Adversarial camouflage (`NULL`, `N/A`, `-`, near-duplicates).
- Fix verification loop with bonus up to `+0.25`.

## Action space
1) Query
```json
{"action_type": "query", "sql": "SELECT * FROM customers LIMIT 10"}
```

2) Submit report
```json
{
  "action_type": "submit_report",
  "report": {
    "null_issues": {"email": {"value": 12, "confidence": 0.92}},
    "duplicate_row_count": {"value": 16, "confidence": 0.88},
    "schema_violations": [],
    "drifted_columns": [],
    "drift_details": {},
    "relational_issues": [],
    "recommended_fixes": ["Add NULL checks"]
  }
}
```

3) Fix SQL
```json
{"action_type": "fix_sql", "sql": "UPDATE orders SET quantity = ABS(quantity) WHERE quantity < 0"}
```

## Observation space
- `task_id`
- `task_description`
- `tables`
- `row_counts`
- `step`
- `max_steps`
- `query_credits_remaining`
- `phase` (`audit` | `fix`)
- `last_query_result`
- `last_action_error`
- `last_fix_score`

## Tasks
| ID | Name | Difficulty | What agent must find | Expected baseline |
|----|------|-----------|---------------------|-------------------|
| 1  | Null & duplicate detection | Easy | Nulls, disguised nulls, exact/near dups | ~0.82 |
| 2  | Schema violation repair | Medium | Type/format/range/unparseable violations | ~0.61 |
| 3  | Silent data drift | Hard | Mean shift, new cats, referential drift | ~0.34 |
| 4  | Multi-table relational audit | Expert | Orphaned FKs, temporal violations, aggregate mismatches | ~0.19 |

## Reward design
- Base audit score from deterministic task grader.
- Confidence Brier adjustment per finding.
- Budget bonus up to `+0.10`.
- Fix bonus up to `+0.25`.

Formula:

`total = min(1.25, audit_score × brier_adj + budget_bonus + fix_bonus)`

## Baseline scores (multi-seed robustness)
| Seed | Task 1 | Task 2 | Task 3 | Task 4 | Mean |
|------|--------|--------|--------|--------|------|
| 42   | X.XX   | X.XX   | X.XX   | X.XX   | X.XX |
| 123  | X.XX   | X.XX   | X.XX   | X.XX   | X.XX |
| 777  | X.XX   | X.XX   | X.XX   | X.XX   | X.XX |

## Running inference
```bash
ENV_URL=http://localhost:7860 \
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
HF_TOKEN=your_token \
python inference.py
```

## Validation
```bash
./validate-submission.sh https://your-space.hf.space
```
