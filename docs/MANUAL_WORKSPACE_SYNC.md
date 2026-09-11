# OTTO workspace synchronization

## PR #42 correction and current execution boundary

The owner has explicitly requested that the assistant handle the GitHub correction
rather than requiring another manual package upload. Update the existing
`alvaromendizabal-patch-1` branch, preserve the pull request, and merge only after
its active quality, neural-contracts, notebooks and portfolio checks pass. Do not
weaken the workflow or type-checking configuration. This permission does not launch
AWS compute or imply that a stopped SageMaker workspace has already synchronized.

### Diagnosed failures

The first revision failed Ruff rules UP022 and UP036. The second revision corrected
those findings and passed Ruff, but failed mypy with three diagnostics at lines
509-511. In `Session.restore`, `receipt` first held the dictionary returned by
`extract_verified` and was then reused for JSON bytes. The correction keeps that
dictionary intact and uses a separately typed `receipt_bytes` value for hashing
and S3 publication. No feature formula, model, dataset, workflow, or quality gate
is changed.

The existing 38 offline tests cover full restore/replay, immutable source versions,
archive safety and preservation of local Git work. Passing those behavioral tests
alone is not proof that the complete repository checks pass. The real GitHub run
for the corrected commit remains the authority for Ruff, mypy and full-suite status.

The helper hash and all commands below are updated together. Old source objects
and the three reviewed bundle identities are preserved. Do not use a command from
an older chat message that pins a different helper. Successful GitHub publication
does not establish that the updated helper has been archived or executed in AWS.

The later archive/restore instructions below are retained for the AWS milestone;
they are not a request to repeat manual GitHub uploading. Execute no AWS stage
until the corrected revision is green, merged, and its prerequisites are checked.

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
  'd5a10235ec9fe8df01b26b8b016724a562d42bfa4b920b53cbca40d920670686' \
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
source/d5a10235ec9fe8df01b26b8b016724a562d42bfa4b920b53cbca40d920670686/sync_otto_workspace.py
source/d5a10235ec9fe8df01b26b8b016724a562d42bfa4b920b53cbca40d920670686/manifest.json
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
    "s3://otto-recsys-560403859723-us-west-2/manual/workspace-sync/af39235b19df/source/d5a10235ec9fe8df01b26b8b016724a562d42bfa4b920b53cbca40d920670686/sync_otto_workspace.py" \
    "$helper" --region us-west-2 --only-show-errors
  printf '%s  %s\n' \
    'd5a10235ec9fe8df01b26b8b016724a562d42bfa4b920b53cbca40d920670686' "$helper" | sha256sum --check
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

- Second-revision type-check failure: https://github.com/alvaromendizabal/otto-recommender-system/actions/runs/34547841699/job/103104192476
- First-revision PR job: https://github.com/alvaromendizabal/otto-recommender-system/actions/runs/34544773813/job/103094915619
- Failed push job: https://github.com/alvaromendizabal/otto-recommender-system/actions/runs/34544750792/job/103094841521
- GitHub browser upload: https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
- CloudShell files: https://docs.aws.amazon.com/cloudshell/latest/userguide/getting-started.html
- Studio launch: https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-launch.html
- Stop versus delete: https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-running-stop.html
