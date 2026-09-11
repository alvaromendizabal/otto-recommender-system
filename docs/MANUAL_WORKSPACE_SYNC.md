# Manual OTTO workspace synchronization

## Scope and current evidence

The 2026-09-10 owner-returned smoke-input collection finished in **10.915 seconds**,
with **16 AWS reads** and **27,092,677 downloaded bytes**, without training or cloud writes.
All **17** entries in its checksum manifest passed independent verification. Rechecking
all three reviewed result bundles verified **162** checksum entries (102 + 43 + 17).

The two collected wide graph contracts both use **2022-08-20 22:00 UTC** history, whereas
the early window requires **2022-08-16 22:00 UTC**. They must not be reused for early
confirmation. The scoped inventory does not establish that no suitable graph exists
elsewhere. This review did not independently reconstruct all prefixes and targets,
load full graph tables, build features, fit models, or change the frozen feature lists.
Feature research remains open and the challenger remains unpromoted.

The accompanying `reports/research/shared_feature_confirmation_preflight.json` is an
immutable preparation-time review. Actual synchronization completion is recorded by the
owner-run receipt, not inferred from this document.

## Storage responsibilities

GitHub contains reviewed code, tests, executed notebooks and small public-safe reports.
The project S3 bucket contains data, models, recovered evidence and immutable receipts.
The existing `otto-dev` SageMaker space contains a working Git checkout and the recovered
evidence outside that checkout. This is explicit checkpoint-based synchronization, not
an automatically running two-way synchronization service.

The helper archives only the three SHA-pinned returned ZIPs. These include the prior
collector sources. It does not upload your home directory, unrelated project files,
credentials, or newly selected data. No raw ZIP, session IDs or labels are included in
this public package. Existing large history/checkpoint objects remain at their existing
S3 locations; no full bucket download is necessary.

## 1. Archive in the existing Oregon CloudShell

Upload `scripts/sync_otto_workspace.py` using Actions -> Upload file. Run:

```bash
cd "$HOME"
python3 -u "$HOME/sync_otto_workspace.py" archive
```

Required successful status: `AWS_ARCHIVE_VERIFIED`.

The expected inputs are the existing result ZIPs in `~/otto_manual_recovery/`,
`~/otto_confirmation_preflight/`, and `~/otto_shared_feature_smoke_inputs/`.
The script also accepts the exact reviewed ZIP directly in the home directory.
Do not rerun older collectors or regenerate those ZIPs. An unknown hash or missing file
stops execution; return the report instead of deleting evidence.

The private destination is:

```text
s3://otto-recsys-560403859723-us-west-2/manual/workspace-sync/af39235b19df/
```

Every object is created conditionally and read back in full for SHA-256 verification.
Existing identical objects are reused. An existing different object is never overwritten.
The script does not change IAM, launch compute, delete prior evidence, train, or push Git.

## 2. Publish this four-file package through GitHub

In the repository root on `main`, use Add file -> Upload files. Drag the four INNER
folders `scripts`, `tests`, `docs`, and `reports` from the extracted package. There are
exactly four files. Do not upload the package ZIP, a parent wrapper folder, or any of the
three private result ZIPs. The paths must be:

```text
scripts/sync_otto_workspace.py
tests/test_workspace_sync.py
docs/MANUAL_WORKSPACE_SYNC.md
reports/research/shared_feature_confirmation_preflight.json
```

Use commit message `Record manual recovery and synchronize OTTO workspace`. Choose a new
branch and start a pull request; the automatically suggested branch name is suitable.
Do not use a `results/` branch for this maintenance-only change, since the existing
workflow gives that prefix extra notebook-publication behavior.

Review all four filenames. Let the unchanged repository CI run. Merge only after its
checks pass. If checks fail or require an unavailable approval, preserve the PR and
return its link; do not weaken CI or start a paid SageMaker app to work around it.
No notebook output is fabricated or rerun solely to publish this synchronization helper.

## 3. Open the existing SageMaker space, after archive and merge

Use SageMaker AI in `us-west-2` -> Studio -> domain
`QuickSetupDomain-20260902T115323` (`d-njhxv1erusdc`) -> user profile
`default-20260902T115323` -> Open Studio -> Launch personal Studio, when shown.
Within Studio choose JupyterLab -> `otto-dev` -> Run space -> Open JupyterLab.

The live preparation-time observation was a stopped JupyterLab app with an existing
100 GB space and configured CPU `ml.m7i.2xlarge`, distribution image alias `4.4.2`.
Do not recreate, delete or enlarge the space. This setup is sufficient for the bounded
file-sync stage, subject to its actual free-space and identity checks; it is not a
certification of full graph-building memory or a Python ML environment.
Running the application incurs instance usage; stop it explicitly after this milestone.

## 4. Restore and verify in the SageMaker terminal

In JupyterLab choose File -> New -> Terminal, then paste this complete block:

```bash
cd "$HOME"
(
  set -e
  helper="$(mktemp /tmp/otto-sync.XXXXXX)"
  trap 'rm -f -- "$helper"' EXIT
  aws s3 cp \
    "s3://otto-recsys-560403859723-us-west-2/manual/workspace-sync/af39235b19df/source/sync_otto_workspace.py" \
    "$helper" --region us-west-2 --only-show-errors
  printf '%s  %s\n' \
    '411b81fb2aaf604622efa14747a96c074afaa06e1592c9eadc1729950f3a068a' "$helper" | sha256sum --check
  python3 -u "$helper" restore --confirm-space otto-dev
)
```

Only the verified helper is temporarily downloaded; all recovered evidence is written
to persistent home storage. The temporary helper is removed after execution and remains
preserved in S3 and GitHub. No passwords or access keys need to be pasted anywhere.

The helper restores the three archives under:

```text
~/otto-artifacts/manual-sync/af39235b19df/
```

It locates an existing OTTO-named checkout in home, projects, SageMaker, or workspaces,
or clones the public repository into `~/otto-recommender-system` if none exists there.
It refuses multiple matching checkouts, unexpected origins, a non-main branch, untracked
or modified files, ahead/diverged branches, or a GitHub main that does not contain this
exact helper and the reviewed report. No stash, reset, clean, force push or deletion is
performed. This bounded search does not certify every possible custom checkout path.

Expected success: `AWS_GITHUB_WORKSPACE_SYNCED`. The saved receipt includes Git commit,
tree, checkout path, archive hashes and extraction checks, plus an S3 receipt read-back.
This is a verified snapshot, not a promise that later changes auto-synchronize.

## 5. Return the result and stop unused compute

Use the JupyterLab file browser to download:

```text
~/otto_workspace_sync/otto_workspace_sync_result.zip
```

Attach it to ChatGPT for review. The script prints the exact absolute path. On any
`STOPPED_REVIEW_REQUIRED`, return that same ZIP instead of repeating the failure. If no
ZIP exists, return the visible terminal error. Do not change IAM or install packages.

In the Studio tab choose Running instances, find `otto-dev` / JupyterLab, and choose
Stop. Stop the app, **do not delete the space**. Files persist when the app is stopped;
deleting the space destroys its storage. Retained storage/S3 can still incur charges.

## Bounds, tests and remaining work

Each archive/restore process has a 280-second work alarm, 15-second heartbeats,
64 AWS-call maximum and 128 MiB transfer maximum; reporting follows the work phase.
Git operations have individual timeouts. Restore requires at least 1 GiB free.
No training occurs and there are no background jobs started by this helper. These limits
do not shut down the interactive SageMaker application: stopping it is the owner's step.

Offline checks cover unsafe archive paths, checksum tampering, symlinks, no-overwrite
behavior, conditional S3 writes/read-back, partial recovery, unchanged replay, identity
and transfer gates, and actual local Git clone/fast-forward/preservation fixtures.
Full repository CI is a separate, required owner-triggered check.

After the synchronization receipt passes review, the scientific next step is a bounded
correct-cutoff graph-dependency plan followed by 256-query feature reconstruction with
measured memory/throughput. No full-data rebuild, ranker fit, feature rescreening, or
promotion is authorized by a successful file synchronization.

## Operational references

- AWS CloudShell upload/download: https://docs.aws.amazon.com/cloudshell/latest/userguide/getting-started.html
- GitHub browser upload and new branch: https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
- Studio launch: https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-launch.html
- JupyterLab space configuration: https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl-user-guide-configure-space.html
- Stop app versus delete space: https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-running-stop.html
- S3 conditional PutObject: https://docs.aws.amazon.com/AmazonS3/latest/API/API_PutObject.html

These document links explain service behavior. The user's AWS settings above were
observed through read-only connector calls during preparation, not inferred from docs.
