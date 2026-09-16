# mini-wam results (auto-generated)

## Main results (test split)

| Model | Success | Avg steps | Collisions |
|---|---|---|---|
| Random | ~0% | 200 | — |
| simple_policy  |   52.5% |    115.1 |   33.4 |
| action_chunk   |   62.5% |    111.1 |   35.7 |
| cascade        |   59.0% |    115.9 |   24.6 |

## OOD benchmark (success rate %)

| Split | simple_policy | action_chunk | cascade |
|---|---|---|---|
| test             | 52.5% | 64.5% | 57.5% |
| ood_combo        | 53.0% | 57.0% | 48.0% |
| ood_distractors  | 41.0% | 40.0% | 42.0% |
| ood_layout       | 44.5% | 50.0% | 49.0% |
| ood_long         | 48.5% | 62.0% | 55.5% |
