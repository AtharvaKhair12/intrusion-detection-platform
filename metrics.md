# Isolation Forest Training Metrics

## Validation Sweep
Tuning was performed on a 20% validation slice of the test data.

| Contamination | FPR | Recall |
|---------------|-----|--------|
| 0.01 | 0.0095 | 0.0999 |
| 0.02 | 0.0201 | 0.2898 |
| 0.05 | 0.0496 | 0.3021 |
| 0.1 | 0.0994 | 0.4530 |

## Chosen Operating Point
**Selected Contamination:** 0.05

We selected this contamination value because it achieves an FPR under 5% while maximizing recall.

## Final Evaluation (Held-out Test Slice)
Evaluated on the remaining 80% test slice.

- **FPR:** 0.0500
- **Recall:** 0.3043
- **Precision:** 0.8817

> **CAVEAT on Precision:** The test set has an artificial class balance of ~45% benign and 55% attack. This is not representative of real-world traffic where attacks are far rarer. Therefore, the reported Precision is significantly higher than what would be observed in production at the same FPR and Recall.
