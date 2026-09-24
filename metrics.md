# Isolation Forest Training Metrics

## Validation Sweep
Tuning was performed on a 20% validation slice of the test data.

| Contamination | FPR | Recall |
|---------------|-----|--------|
| 0.01 | 0.0094 | 0.0998 |
| 0.02 | 0.0201 | 0.2927 |
| 0.05 | 0.0499 | 0.3075 |
| 0.1 | 0.0996 | 0.4545 |

## Chosen Operating Point
**Selected Contamination:** 0.05

We selected this contamination value because it achieves an FPR under 5% while maximizing recall.

## Final Evaluation (Held-out Test Slice)
Evaluated on the remaining 80% test slice.

- **FPR:** 0.0500
- **Recall:** 0.3097
- **Precision:** 0.8836

> **CAVEAT on Precision:** The test set has an artificial class balance of ~45% benign and 55% attack. This is not representative of real-world traffic where attacks are far rarer. Therefore, the reported Precision is significantly higher than what would be observed in production at the same FPR and Recall.

## Per-Class Breakdown (Held-out Test Slice)
| Attack Type | Recall | Samples |
|-------------|--------|---------|
| DoS Hulk | 0.6143 | 184351 |
| PortScan | 0.0007 | 126842 |
| DDoS | 0.1713 | 102473 |
| Other | 0.2218 | 31579 |

**Interpretation:** The aggregate 30% recall is highly skewed by attack type, which is an expected domain pattern. High-volume, disruptive attacks like DoS Hulk show relatively strong recall (~61%) because they create significant statistical deviations in flow features (like high packet rates and large flow volumes). On the other hand, stealthier attacks like PortScan have near-zero recall (~0.07%). This is because a slow port scan mimics normal traffic very closely on a per-flow basis, making it virtually indistinguishable from benign background noise without sequential or stateful cross-flow analysis. Increasing `n_estimators` to 300 provided a modest bump in recall for the 'Other' category, but the fundamental limitations of an unsupervised, stateless model on stealthy attacks remain.
