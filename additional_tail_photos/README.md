# Additional airline photo candidates

These 33 attributed photographs extend the source pool for later model work:

| Airline | Photos | Saved view |
| --- | ---: | --- |
| Alaska | 7 | Tail crops |
| American | 5 | Tail crops |
| Delta | 5 | Full aircraft |
| Southwest | 6 | Full aircraft |
| United | 10 | Tail crops |

See each airline's `SOURCES.md` for the source page, creator, license and
derivative details. The photos include ground and airborne views, different
angles and liveries; several include realistic partial occlusion. They are not
an independently sampled benchmark. Most are 737 photographs, and several
share photographers and settings, so this collection has limited diversity.

`manifest.csv` records the airline label, saved file, dimensions, SHA-256,
source page, view type, provisional source group, and (where a tail crop was
provided) the normalized box on the source derivative. Full-plane files need
tail-box annotation before they can be used by the current classifier. For
tail crops, the source box is provenance, not an annotation on the saved crop.

**These files are candidates, not part of the current 50-image dataset.** The
existing preparation script intentionally requires exactly 50 annotated files.
Do not copy candidates into `airline_tails/` or overwrite dataset v1. Before
training, review labels and tail boxes, build a new versioned dataset and split
by related photograph, aircraft and session groups. `provisional_group` is a
conservative starting point, not a completed split: check source dates,
registrations and the existing annotations and evaluation manifest. In
particular, Acroterion BWI photographs may be related to existing evaluation
images, and several 4300streetcar Boston and Colin Brown Atlanta images share
settings. Keep related sources together or exclude them from the comparison.
The existing full-plane stress set has already been examined; reserve a fresh,
independent test set for meaningful model comparisons.

Source titles were compared against recorded training and selected full-plane
evaluation sources with no exact title matches. Saved image bytes are unique.
Decoded image hashes and dHash/pHash heuristics were checked against existing
training images, their annotated crops, and the selected full-plane evaluation
images and crops; no candidates met the repository's duplicate thresholds. This
screen cannot establish independent aircraft, sessions, unseen source images, or
absence from model pretraining. Do not infer that these photos improve accuracy
without retraining and evaluation.
