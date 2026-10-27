# GitHub Actions for this project

A workflow is a recipe GitHub runs after an event. Ours is
`.github/workflows/ci.yml`: a push or pull request starts a temporary Linux
machine (the runner), which checks out the code, installs the small test
dependencies, and executes the checks below. The job is the complete set of
steps; each step gets its own log. A green check means those steps passed.

## When it runs

- Every push to any branch, including the push to `main` caused by a merge.
- Pull requests when opened, updated with new commits, or reopened. Any target
  branch is supported, including stacked pull requests such as the crop work.
- Manually through **Actions → Repository checks → Run workflow** once the
  workflow is on the default branch.

A push to an open pull request can produce both a push run and a pull-request
run. The latter normally tests GitHub's proposed merge with the target branch,
which can catch integration problems. New commits cancel older runs for the
same event and branch so stale work does not keep running.

## What the checks do here

| Check | Why it matters for the airline app |
| --- | --- |
| Tracked Python/JavaScript syntax and JSON/YAML parsing | Catches a broken script or malformed endpoint/workload/app configuration before it reaches the other computer. It parses source without importing training libraries or running notebooks, including the wireframe and the Tailspot app's JavaScript. |
| Dataset preparation tests | Checks image cropping, source immutability, duplicate handling and isolation between training/validation/test splits. Currently validates the original 50-image dataset. |
| Weekly schedule tests | Checks that airline labels match the classifier, the sample CSV is consistent, and local-time recurrence handles daylight-saving and adjacent-day matching. |
| Tailspot app tests (`tests/test_app_*.py`) | Uses fake Databricks clients to check upload validation, orientation, tail crops, stored files, the departure window against the schedule reference, cold-start retries, failure paths, the submissions-table row and the HTTP contract. It also keeps `app/serving_client.py` identical to `serving/client.py`. |
| Serving client tests | Uses a fake endpoint to inspect the request and response contract, metadata removal, orientation and crop handling. |
| Saved evaluation-image validation | Confirms the ten checked-in full-plane photos still match their manifest, dimensions, metadata and recorded source constraints. It uses the `validate` command; it does not download images or rescore the model. |

The job needs no Databricks credentials or GPU. It runs on GitHub even while both
of your computers are offline. It has read-only repository permissions, and
checkout credentials are not kept for later steps. The action revisions and direct
test dependencies are pinned; `requirements-ci.txt` is for these tests, separate
from the exact dependencies recorded with the trained model.

## What a green check means

It provides evidence that the repository's tested code and fixtures still work.
It does not establish model accuracy on new photos, verify live Databricks table
contents, execute Databricks SQL, test live endpoint permissions, deploy the app,
or retrain the model. Image-quality review and a fresh independent model
evaluation are still needed. The Databricks integration remains on the other
device; this workflow does not need access to that device.

## Reviewing a run

Open the repository's **Actions** tab, choose **Repository checks**, and open a
run. Expand the failing step to see the test or parser error. A pull request also
shows the result in its checks area. Fix the code and push another commit to rerun
it. Repository checks alone do not prevent merging: if desired, configure a
branch ruleset requiring **Code, data, and serving checks** after the first
successful run. No branch rules are changed by this PR.

To reproduce the commands locally, install Node.js 24 and use Python 3.12 in a
virtual environment (on Windows, enable UTF-8 with `$env:PYTHONUTF8 = '1'`):

```text
python -m pip install -r requirements-ci.txt
python ci/check_syntax.py
python -m unittest discover -s tests -p "test_*.py" -v
python -m unittest discover -s serving -p "test_*.py" -v
python full_plane_tests/download_photos.py validate
```

The syntax command checks tracked files, so add newly authored files to Git
before relying on it. `.local/`, including the video talking points and local
credentials, stays outside the checked-out repository.

Later, a separate deployment workflow could run an authenticated Databricks
smoke test and deploy an approved app/model version. That needs a configured
Databricks service identity and a chosen target environment. Training and model
promotion should have their own data-quality and evaluation gates, rather than
run automatically with every source-code push.

References: [Python CI on GitHub](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
and [workflow triggers](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow).
