# Model version 1: external full-plane evaluation

Model: `dbx_ai_app.airplane_tail_id.airplane_tail_identifier`, immutable version **1**.
Ten photos were selected before scoring; no training or model changes were made.
All results and probability vectors are saved in `results.json`.

| Input | Correct | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Entire full-plane photo | 3/10 | 30% | 0.2061 |
| Manual tail crop of the same photo | 8/10 | 80% | 0.8000 |

| Image | Actual | Full-plane prediction | Manual-crop prediction |
|---|---|---|---|
| alaska_full_01 | Alaska | Southwest | Alaska |
| alaska_full_02 | Alaska | Southwest | Southwest |
| american_full_01 | American | Southwest | American |
| american_full_02 | American | Southwest | American |
| delta_full_01 | Delta | Southwest | Delta |
| delta_full_02 | Delta | Southwest | Delta |
| southwest_full_01 | Southwest | Southwest | Southwest |
| southwest_full_02 | Southwest | Southwest | Southwest |
| united_full_01 | United | United | United |
| united_full_02 | United | Southwest | Southwest |

**Conclusion:** this version is not reliable for full-plane inputs. Nine of ten full
photos received the Southwest label, including an incorrect prediction at 90.7%
confidence. Confidence is not a guarantee of correctness. Tail cropping helps, but
two new crops still fail: Alaska N214AK and United N33289 (older gold globe).

The earlier 10/10 result was on the original reserved tail crops, not these photos.
The paired comparison uses the same ten sources, so it is not twenty independent
test examples. Two images per airline and selection bias prevent robust per-airline
or generalization claims; see `README.md` for source limitations.

Recommended next step: retain a user-selected crop in the demo; expand varied tail
training data and reserve a fresh test set. Full-plane submission needs a tail
localization step and separate end-to-end evaluation. These ten sources are now
observed evaluation data and should not be reported as untouched holdouts after tuning.
