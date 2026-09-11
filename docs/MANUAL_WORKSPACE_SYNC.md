# Manual OTTO workspace synchronization

## Current milestone: correct the existing PR #42

The original upload is in `alvaromendizabal-patch-1`, targeting `main` in
`alvaromendizabal/otto-recommender-system`. Do not open another pull request or commit
this correction directly to `main`. Replace these three existing files on that branch:

```text
scripts/sync_otto_workspace.py
tests/test_workspace_sync.py
docs/MANUAL_WORKSPACE_SYNC.md
```

Leave the already uploaded
`reports/research/shared_feature_confirmation_preflight.json` unchanged. The correction
ZIP contains only the three replacements, not research data or a new result report.

### What failed and what changed

Both original quality jobs stopped at Ruff, with the same two findings:

- `UP022`: replace the two subprocess pipe arguments with `capture_output=True`.
- `UP036`: remove the obsolete lower-Python-version branch under the repository's
  Python 3.13 lint target, together with its now-unused import.

The original jobs passed notebook replay and 83 selected ranking/recovery tests before
reaching that failure. The full quality pipeline did not complete. Passing behavioral
tests alone is not a passing repository CI result. No lint rule, workflow, type-check
configuration, or scientific gate has been disabled by this correction.

The helper's bytes change when code is corrected. Its source and manifest are therefore
stored under a SHA-256-specific path. Existing bundles retain their exact locations and
identities; old helper/manifest objects are preserved instead of overwritten. The old
SageMaker command using the unversioned source key is superseded by the command below.

### Browser steps for this correction

1. Extract the latest `otto_workspace_sync_package.zip` into a separate Windows folder
   such as `OTTO_PR42`. Use this copy, not the earlier extracted package.
2. Open `https://github.com/alvaromendizabal/otto-recommender-system/tree/alvaromendizabal-patch-1`.
   Confirm the branch selector says `alvaromendizabal-patch-1`, not `main`.
3. At the repository top level choose **Add file -> Upload files**. Drag the three
   inner folders `scripts`, `tests`, and `docs` from the new extraction. Do not upload
   the ZIP, its parent folder, or any recovered data ZIP.
4. Confirm the three paths above. Commit with message
   `Correct workspace sync checks and preserve archived helper versions`.
   Choose **Commit directly to the alvaromendizabal-patch-1 branch**.
5. Return to PR #42. The new commit updates the same pull request. Let its new checks
   finish. Do not rerun the unchanged failed commit, bypass checks, or start SageMaker.
6. Merge only after the new revision's quality, neural-contracts, notebooks and
   portfolio checks pass. Use **Merge pull request -> Confirm merge**, then return the
   PR link for the next milestone. If any active check fails, leave the PR open and
   return that failure instead. The existing `publish-notebooks` job is conditional
   on `results/` pushes; its skipped status here is not the Ruff failure.

The next sections describe the later archive/restore milestone. Do not execute them
until the corrected revision has passed CI and is merged. No successful archive or
restore is asserted merely because this document exists.

## Scope and evidence

The owner's reviewed smoke-input collection recorded 10.915 seconds, 16 AWS reads and
27,092,677 downloaded bytes, without training or cloud writes. Its 17 checksum entries
passed the preparation audit. All three reviewed bundles together have 162 recorded
checksum entries (102 + 43 + 17). These are previous audit results, not new experiments.

Both inspected wide-graph contracts use 2022-08-20 22:00 UTC history; early confirmation
requires 2022-08-16 22:00 UTC. They cannot be reused for that early test. The scoped
inspection does not establish that no suitable graph exists elsewhere. Prefix/target
reconstruction and full graph construction remain unfinished. Feature engineering
stays open and the challenger remains unpromoted.

GitHub holds reviewed source, tests, executed notebooks and small public-safe reports.
S3 holds data, models, recovered evidence and receipts. The existing `otto-dev` space
holds a working Git checkout and recovered evidence outside it. This is an explicit
verified snapshot, not an automatic two-way synchronization service.

## After CI and merge: archive from Oregon CloudShell

Upload this revision of `scripts/sync_otto_workspace.py` through **Actions -> Upload
file**. Ensure the uploaded filename is exactly `sync_otto_workspace.py`, not a browser
renamed copy. Verify the helper before running it:

```bash
cd "$HOME"
printf '%s  %s\n' \
  'a4c28c42977bec1245189bc2e199e3ceb58b96b4be8df974f9e3ae7a5a3ed464' \
  "$HOME/sync_otto_workspace.py" | sha256sum --check && \
python3 -u "$HOME/sync_otto_workspace.py" archive
```

Expected success: `RESULT: AWS_ARCHIVE_VERIFIED`.

The three original result ZIPs must remain in `~/otto_manual_recovery/`,
`~/otto_confirmation_preflight/`, and `~/otto_shared_feature_smoke_inputs/`, or directly
in home with the exact reviewed bytes. Do not regenerate them. Different bytes, missing
inputs, missing permissions, or a resource limit cause a stop for review.

Destination bucket/prefix:

```text
s3://otto-recsys-560403859723-us-west-2/manual/workspace-sync/af39235b19df/
```

Bundle keys remain under `bundles/`. The corrected helper uses these revision keys:

```text
source/a4c28c42977bec1245189bc2e199e3ceb58b96b4be8df974f9e3ae7a5a3ed464/sync_otto_workspace.py
source/a4c28c42977bec1245189bc2e199e3ceb58b96b4be8df974f9e3ae7a5a3ed464/manifest.json
```

Objects use conditional creation and full-byte read-back verification. Existing
identical bundles are reused, although requests and read-back transfers still occur.
Legacy source/manifest objects are not deleted or overwritten. The helper makes no IAM
changes, launches no compute, trains no models, and pushes no Git commits.

On failure return `~/otto_workspace_sync/otto_workspace_sync_result.zip` rather than
repeatedly retrying. A checksum failure in the initial shell command means the wrong
helper file was uploaded; that check deliberately prevents execution.

## After archive verification: open the existing SageMaker space

Use SageMaker AI in Oregon (`us-west-2`) -> Studio -> domain
`QuickSetupDomain-20260902T115323` (`d-njhxv1erusdc`) -> profile
`default-20260902T115323` -> Open Studio -> Launch personal Studio, when shown.
Choose JupyterLab -> `otto-dev` -> Run space -> Open JupyterLab.

Earlier preparation observed a 100 GB persistent space with CPU `ml.m7i.2xlarge` and
image alias `4.4.2`. This is a historical observation, not a new live resource check.
Do not recreate, delete or enlarge the space. Restore checks actual free space and
identity. This file-sync stage does not certify the ML environment or graph workload.
Stop the application explicitly afterward; script timeout does not stop app billing.

## Restore in the SageMaker terminal, not CloudShell

Open a Terminal in JupyterLab and run this revised, hash-pinned block. Do not use the
older unversioned helper command from a prior chat message.

```bash
cd "$HOME"
(
  set -e
  helper="$(mktemp /tmp/otto-sync.XXXXXX)"
  trap 'rm -f -- "$helper"' EXIT
  aws s3 cp \
    "s3://otto-recsys-560403859723-us-west-2/manual/workspace-sync/af39235b19df/source/a4c28c42977bec1245189bc2e199e3ceb58b96b4be8df974f9e3ae7a5a3ed464/sync_otto_workspace.py" \
    "$helper" --region us-west-2 --only-show-errors
  printf '%s  %s\n' \
    'a4c28c42977bec1245189bc2e199e3ceb58b96b4be8df974f9e3ae7a5a3ed464' "$helper" | sha256sum --check
  python3 -u "$helper" restore --confirm-space otto-dev
)
```

Only that temporary helper is removed afterward. Recovered evidence is written to:

```text
~/otto-artifacts/manual-sync/af39235b19df/
```

The helper looks for one OTTO checkout in home, projects, SageMaker or workspaces; it
clones into `~/otto-recommender-system` only when no matching checkout is found there.
It stops on multiple checkouts, unexpected origins, a non-main branch, local changes,
local commits ahead of GitHub, or a mismatch between this helper and GitHub main.
It never resets, stashes, cleans or force-pushes existing work. Its bounded directory
search does not cover every possible custom project path.

Expected success: `RESULT: AWS_GITHUB_WORKSPACE_SYNCED`. This certifies the three
reviewed bundles and the published repository snapshot, not the full S3 bucket or later
edits. Source equality remains required; it is not weakened to accept an old helper.

## Return the result and stop unused compute

Download `~/otto_workspace_sync/otto_workspace_sync_result.zip` using the JupyterLab
file browser and return it for review. The script prints the exact absolute path.
On `STOPPED_REVIEW_REQUIRED`, return the same ZIP instead of retrying unchanged. If a
ZIP is not produced, return the visible terminal error. Do not paste access keys.

In the Studio tab choose Running instances, find `otto-dev` / JupyterLab, and Stop.
Stop the app, not the space: deleting the space removes its stored files. Retained
storage and S3 can still incur charges even when app compute is stopped.

## Bounds, tests and scientific next step

Each archive/restore invocation has a 280-second work alarm, 15-second heartbeats,
64 AWS-call cap and 128 MiB transfer cap. Reporting follows the work phase. Git commands
have their own timeouts. Restore requires at least 1 GiB free disk space. The script
starts no background training jobs and does not stop the interactive app itself.

Offline tests exercise ZIP integrity, symlinks, path traversal, no-overwrite behavior,
conditional writes and read-back, identity/budget gates, partial recovery, source-version
coexistence, and local Git fixtures for clone, fast-forward and preservation. Fake S3
and local Git tests do not certify live cloud permissions or full repository CI.
The revision-specific external validation report records what actually ran and which
checks remain pending; do not equate unit-test success with an all-green CI result.

After synchronization receipt review, the next research dependency is correct-cutoff
graph preparation followed by the 256-query feature check with measured resource use.
File synchronization is not approval to rescreen features, train full models or promote
the challenger.

## References

- Failed PR job: https://github.com/alvaromendizabal/otto-recommender-system/actions/runs/34544773813/job/103094915619
- Failed push job: https://github.com/alvaromendizabal/otto-recommender-system/actions/runs/34544750792/job/103094841521
- GitHub browser upload: https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
- CloudShell files: https://docs.aws.amazon.com/cloudshell/latest/userguide/getting-started.html
- Studio launch: https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-launch.html
- Stop versus delete: https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-running-stop.html
