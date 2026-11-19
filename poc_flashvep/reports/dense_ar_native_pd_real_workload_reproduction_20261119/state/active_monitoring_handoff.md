# Active Native reproduction campaign

Checkpoint: 2026-11-20 00:31 KST. The campaign is not complete.

## Scope

Use LongBench v1 Mixed200 only for the Main comparison. Keep Layer/Token and Dynamic experiments stopped. Do not change the frozen Main grid or SLO after observing results.

All engines use B8192/S1024, KV42GiB, prefix caching disabled, FLASH_ATTN/HND, original BF16 and official batch invariance. Native uses the standard NIXL connector and continuous scheduler. It has no custom cohort guard.

Runtime source is frozen in `state/runtime_source_freeze.json`. Analysis and report code can change. Do not edit the nine runtime files during the campaign.

## Controllers

- Main driver: PID1661606.
- Follow-up driver: PID1743747. It waits for Main completion. It then runs Native clean/profile/clean controls at QPS1 and2.35. It next finishes B/S sensitivity.
- UCX log-only probe: PID1737403. It waits for follow-up completion.
- Clean C1 confirmation: PID1757350. It waits for the UCX probe. It adds two Standard and two Native starts to the existing C1 control.

The previous waiting follow-up PID1705934 was refreshed before it owned a GPU job. Its old imported analysis code had a corrected bug. The replacement waits before importing analysis modules. See `state/followup_CPU_controller_refresh.json`.

Read `state/GPU_lease_current.json` for the current GPU job. Do not infer activity from an old log tail. Check every hold marker before dispatching new work.

## Completed Main discovery

All14 first-seed jobs passed output200/200 and normal worker retirement. Rates are0.4/0.7/1/1.15/1.2/1.65/2.35. Main is now repeating QPS1 and1.65 with two new arrival seeds. It will finish22 starts.

At QPS2.35, first-start Native/Agg results were:

- TTFT p95: 46554.458/5066.959ms.
- ITL p99: 138.617/460.781ms.
- TPOT p95: 38.888/346.506ms.
- Generated output: 176.833/250.994tokens/s.
- Joint SLO: 3.0%/8.5%.

These are single-start finite-cohort results. They are not sustainable capacity. Native shows P backlog, but absolute Decode SLO protection and safe headroom are not established.

At QPS1.65, Native sampled P waiting mean was9.405, p95 was14. D ready waiting, including its first query, was0 in the sampled states. KV allocation refusals were0. These host observations do not establish GPU headroom.

## Figures and analysis

The four required draft PNG/PDF pairs are at the repository `poc_flashvep` root. Names start with `longbench200_`: `ttft_p95`, `itl_p99`, `tpot_p95`, and `output_tokens_s`, followed by `_vs_qps`.

Run `results_analysis`, `figures`, `figures_aux`, and `tail_analysis` in `poc_flashvep.dense_ar_native_reproduction` after all GPU work. Main also regenerates its figures after each full seed block.

N8 full KV/hidden/output and N32 continuous output gates passed. Main200 checks output and lifetime. Do not claim that all Main200 KV tensors were copied and compared.

The QPS0.4 pair has different read-only allocation-observer code in `standard_worker.py`. Preserve this limitation. QPS0.7 and later paired runtime files match. Every run stores its source ZIP and hashes.

## Important interpretation

Stock BI0 C1 Decode was approximately7-8ms. Official BI1 was approximately22-25ms. Main uses BI1 in both methods for strict output parity. Do not call this stock BI0 production performance.

The C1 BI1 kernel profile passed the5% observer point-estimate gate narrowly. Its Mean TBT change was+4.9747%. BI0 failed at+18.5181%. The C1 profile uses synchronous scheduling and Compile0. It is not Main kernel activity.

Native concurrent NIXL posting remains slow. First Main QPS0.4 had transfer-call wall mean105.21ms and calling-thread CPU mean0.115ms. NIXL1.4.1 thread-pool source contains `future.wait()`. This does not fully identify internal transfer latency or pure DMA time.

The new online profile code uses90-second prespecified captures. It checks clean/profile/clean point changes against5%. Failed observer gates remain diagnostic. Kernel inactivity can include DMA and control waits. It is not a safe executable window.

## Remaining closeout

1. Wait for Main, follow-up, UCX and C1 confirmation completion.
2. Inspect failures and observer gates. Do not hide failed results.
3. Regenerate all analysis and figures. Review the PNG files.
4. Write the final summary, QPS results, crossover and prefill-heavy feasibility reports. Update existing Native reports with final scoped verdicts.
5. Restore all eight approved burners. Verify ownership and positive utilization in three samples.
6. Verify `common.preserve()`. Preserve the user checkout, index and external runtime.
7. Write `CAMPAIGN_CLOSED.json` only after all GPU work ends.
8. Run the new `publication` module with `--archive`, then `--push`. Verify the remote SHA.

Publication is not complete. The new branch is `flashvep/dense-ar-native-pd-real-workload-reproduction-20261119`. Do not check it out. The private-index publisher preserves the user's current branch and staged changes.

## Later checkpoint: 2026-11-20 02:08 KST

Main22 starts are complete. All4400 outputs matched. Required four PNG/PDF pairs were refreshed and visually checked. Start-level CI is available only at QPS1/1.65, with3 starts per method. Other cells have1 start.

Native online profiles are complete and analyzed. QPS1 P/D kernel busy is69.1322%/89.1763%. QPS2.35 is99.3415%/85.2237%. Observer changes are within5%, with no dropped or invalid records. QPS2.35 P waiting mean is36.7721. D active decode mean is5.9778. D ready waiting is0. Large gaps overlap transfer Host enclosures. This is not a causal intervention or a safe window.

High-load P bottleneck is confirmed. Safe D headroom is not established. Keep offloading on hold.

Follow-up PID1743747 is completing sensitivity. Low8 and moderate8 are complete. High8 are now running. UCX probe PID1737403 and C1 confirmation PID1757350 still wait in order. Read the current lease and hold markers. The earlier filesystem-only monitor session54068 ended with143. This did not stop a controller or GPU job. The active assistant now checks the files directly.

All10 reports now exist, including the final-summary draft and prefill-heavy feasibility. C1, UCX, sensitivity and publication sections still require final updates. `historical_activity_audit`, `task_summary`, and `critical_path_summary` produced additional read-only evidence. Main QPS2.35 mean TTFT26904.91ms decomposes into P pre-prepare25299.17ms, P prepare-to-ready1313.26ms, ready-to-D229.34ms, D prepare-to-first63.14ms. These are Host boundaries, not exclusive GPU time.

`sensitivity_results.py` is ready to aggregate after24 cells. It records per-run runtime hashes. The moderate Native B8192/S1024 control predates the expanded queue observer. Other moderate controls include it. Record this limitation. Do not claim a globally best-tuned frontier.

`closeout.py` is ready. Run it only after all four completion markers exist. It validates output/config/source preservation, tests10 CPU checks, obtains the quiet lock and verifies all8 burners in3 positive-utilization samples. It writes `CAMPAIGN_CLOSED.json`. Then archive, push and run `closeout --post-publication`.

No publication occurred yet. Private-index publisher self-test passed. Publication helper dependencies match the branch parent. The five historical report hashes and nine frozen runtime files remain unchanged. Avoid an unrestricted `git status`: this user's existing index/worktree contains a very large unrelated deletion set. Use `common.preserve()` and scoped checks.

## Later checkpoint: 2026-11-20 02:35 KST

All GPU work ended. Sensitivity24/24 passed. Clean C1 has3 starts per arm and passed the scoped ratio-CI screen. Native/Standard ITL p99 ratio is0.96785[0.95846,0.97676]. TPOT p95 is0.96795[0.95602,0.98105]. This is a clean serial diagnostic, not concurrent isolation.

UCX probe selected CUDA IPC zero-copy for CUDA-to-CUDA remote reads. Main Native11 logs also contained36 context teardown errors each. All occurred after completion, by4.456–6.681seconds. Process exit codes and request KV retirement passed. SDK teardown remains unresolved. See `UCX_protocol_selection_and_teardown_audit.json`. Do not call SDK destruction error-free.

All10 reports were updated. CPU inventory is90 COMPLETE,2 FAILED,1 INCOMPLETE_PRESERVED. Main analysis and all figures were refreshed. Still run final clarity/link/tests, `closeout`, verified archive, private-index push and post-publication check. No push occurred yet. Keep all8 approved burners running during CPU closeout.
