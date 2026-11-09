# GPU correctness checks on Modal

The optional `gpu-tests.yml` workflow runs short CUDA correctness checks on
Modal for relevant, trusted same-repository pull requests, including drafts.
Maintainers can also select **Run workflow**. The existing CPU workflow continues
to run on pushes and pull requests. No always-on GPU VM is needed.

This suite covers the prepared execution contract, planner, CUDA tensor routing,
and the stateless forecast result format. It runs `tests/test_cuda_contract.py`,
`tests/test_execution.py`, `tests/test_batching.py`, and
`tests/test_result_transport.py`. It does not download checkpoints, run the vLLM
server, measure forecast accuracy, or produce performance baselines. Passing it
does not verify the driver on the team's separate GPU VM; run that machine's
health check after boot and before experiments. Full-model serving/parity,
forecast accuracy evaluation, and performance profiles remain manual jobs on an
idle, suitably sized GPU (the reference baseline uses an A100).

Automatic runs are limited to changes in the planner, prepared execution
contract, adapter metadata used by that contract, its fake adapter example,
pooling/result transport, the selected test files, Modal runner/setup tests, or their workflow and
dependency configuration. The exact filters live in
[`gpu-tests.yml`](../.github/workflows/gpu-tests.yml). PRs changing only documentation
or unrelated model code do not trigger this suite. There is no GPU `push`
trigger, so a PR update does not also submit a duplicate push run.

Fork PRs are skipped before checkout and never receive Modal credentials. This
uses `pull_request`, never `pull_request_target`; only branches within this
repository are treated as trusted. Repository collaborators can change code that
runs with the Modal token, so restrict write access to people you trust. The job
has only `contents: read` GitHub permissions and does not persist checkout credentials.

## Account setup: apply billing limits first

1. Create a [Modal](https://modal.com/) account and choose a **Starter** workspace.
   Modal currently advertises $30/month in free credits. It requires a payment
   method; compute beyond applicable credits can be billed. Confirm the credit
   amount actually shown for your workspace before continuing.
2. As a workspace Owner or Manager, open **Settings → Usage & Billing**. Set the
   **Workspace usage budget** to **$25/month**, or a lower amount if your
   available monthly credits are below $30. This budget is total usage before
   credits, so it leaves some room below the advertised allowance.
3. Separately set the **Workspace spend limit** to **$0**. This is the limit on
   out-of-pocket charges after credits. Save and verify both settings in the
   dashboard; do not rely on the default spend limit. If your account cannot
   enforce a zero-dollar spend limit, leave GPU testing disabled until that is
   resolved.
4. Keep this workspace dedicated to CI, or account for its other usage. Credits
   and budgets apply to the whole workspace. Check the dashboard after the first
   run and when a new billing cycle starts.

The billing settings are the protection against paid overages. Repository flags,
timeouts, and estimated costs do not verify the remaining credits or guarantee
free execution. The workflow cannot inspect or change your billing settings.
See Modal's [pricing](https://modal.com/pricing),
[billing](https://modal.com/docs/guide/billing), and
[budget/spend-limit definitions](https://modal.com/docs/guide/budgets).
Environment-level budgets require a paid plan and are not needed for this setup.

Starter currently includes three Modal workspace seats. The five project members
do not all need a Modal seat to use the repository workflow: one workspace owner
can configure CI credentials, while teammates with the appropriate repository
permissions launch the workflow in GitHub.

## Credentials and GitHub configuration

Use a token belonging to the workspace whose billing limits you configured.
The [Modal setup guide](https://modal.com/docs/guide/getting-started) describes
browser authentication. For example, use a separate local environment:

```bash
python3 -m venv .venv-modal
.venv-modal/bin/python -m pip install modal==1.6.3
.venv-modal/bin/modal setup
```

Create or obtain a Modal API token pair for that workspace. In
**vTSFM → Settings → Secrets and variables → Actions**, add these repository
secrets:

| Secret | Value |
| --- | --- |
| `MODAL_TOKEN_ID` | The token ID |
| `MODAL_TOKEN_SECRET` | The matching token secret |

`modal token new` opens an authenticated browser flow and stores the token pair
in your local `~/.modal.toml` profile. Copy that profile's `token_id` and
`token_secret` directly into GitHub's secret forms; do not print or send them in
chat. Confirm that the selected workspace is the one with the billing caps.

After verifying the account's credit allowance, usage budget, and $0 spend limit,
add this repository **variable** in the same settings area:

| Variable | Value |
| --- | --- |
| `MODAL_GPU_ENABLED` | `true` |

This variable enables both automatic and manual GPU submissions. Setting it to
`true` is the owner's persistent acknowledgement that the credit allowance and
both billing limits are configured; automatic runs require no per-PR confirmation.
Leave it unset or set it to `false` to skip all GPU jobs. It is
an explicit setup acknowledgement, not a programmatic billing check. Never
commit token files, put secrets in workflow YAML, or paste token values into an
issue, PR, or chat. The workflow passes credentials to the Modal client; test
code does not need a GitHub token or access to the team's GCP VM.

This follows Modal's [GitHub Actions CI pattern](https://modal.com/docs/examples/ci-on-modal).
Creating an account and adding credentials are required before the first remote
run; local validation of the workflow does not establish Modal GPU execution.

## Run and inspect a check

With setup enabled, opening, reopening, or updating a same-repository PR with a
matching path runs the small CUDA suite automatically. Draft PRs are included.
All GPU runs share one concurrency group: an active run is allowed to finish,
and newer commits may replace an older pending run. GitHub does not guarantee
FIFO queue order. This avoids cancelling an active client while its GPU app is
still running.

For a manual override of the path filter, open **Actions → Modal GPU checks →
Run workflow**, select a trusted ref, and check the billing-limit confirmation.
The confirmation remains required for each manual run; it does not override a
disabled `MODAL_GPU_ENABLED` variable. The workflow file must be on the default
branch for GitHub to show this manual trigger. The prepared execution contract
from PR #26 is also required. Run only code you trust with the workspace credentials.

For a local invocation from a checkout, supply the same `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET` as shell environment variables through your local secret
manager. A saved Modal profile alone does not pass this workflow's explicit
credential gate. Then run:

```bash
MODAL_GPU_ENABLED=true .venv-modal/bin/modal run -m ci.modal_gpu
```

Use module mode (`-m`) so Modal uploads the complete `ci` Python package,
including the shared test runner. The first run builds a pinned Python/Torch test image; subsequent runs can reuse
that image. Inspect the GitHub job log for the GPU and package versions, CUDA
test output, and pytest summary. A missing GPU or a failed CUDA operation is a
failure, not a successful skipped check. Modal's dashboard shows the associated
execution and its usage.

Before submitting the GPU function, the local entrypoint writes its running app
ID to a fresh, private file unique to this GitHub run and attempt. An `always()`
cleanup step uses the same pinned Modal client to stop only that validated app
ID. An already stopped app is accepted only after checking the exact ID's stopped
lifecycle. If setup fails before the entrypoint records an ID, cleanup makes no
Modal call and no GPU function has been submitted. The ordinary ephemeral app
lifecycle also stops the app when its client disconnects; this workflow does not
deploy or detach an app. Forced runner termination can prevent cleanup, so inspect
Modal after a cancellation or timeout and stop only that run's app if necessary.

## Resource and usage bounds

The job requests one GPU, one physical CPU core, and 4 GiB of host memory.
Modal tries the L4 first and can schedule the same job on a T4 when L4
capacity is unavailable. Both run the CUDA tensor contract tests on
PyTorch 2.14.0 with CUDA 13.0, matching the numerical serving stack in
[PR #32](https://github.com/vinli0921/vTSFM/pull/32). The test output
reports the actual GPU and runtime versions.
Use the target GPU separately for tests of architecture-specific kernels.
See [Modal GPU fallbacks](https://modal.com/docs/guide/gpu#gpu-fallbacks).

It allows one container, keeps no warm pool, uses no persistent volumes, and has
no scheduled trigger or model downloads. The execution timeout is 300 seconds,
startup timeout 120 seconds, and test subprocess timeout 240 seconds. There are
no configured application retries. These bounds limit each submission; they are
not a monthly spending cap.

The GitHub job has a separate 30-minute wall-clock limit to allow for provisioning.
Modal's execution timeout excludes scheduling wait, so an app can wait longer
than five minutes before its tests start. Extending that wait does not increase
the five-minute GPU execution cap.

At the published November 2026 rates, five minutes of the requested GPU, CPU,
and memory is about $0.073 on an L4 or $0.056 on a T4 before startup and
image-building costs. Both use the same workspace usage budget and $0
out-of-pocket spend limit. Budgeting
roughly $0.10–$0.15 for a short run is an estimate, not a price guarantee. Recheck
[Modal's current rates](https://modal.com/pricing) and the workspace dashboard
before increasing test frequency or adding model inference.

Keep full-model parity checks, accuracy evaluation, and benchmarks manual until
their resource needs are measured. To stop future submissions, disable the workflow or set
`MODAL_GPU_ENABLED=false`; cancel any active job and verify its Modal execution
has ended. The provider's $0 spend limit remains the final credit-only safeguard.
