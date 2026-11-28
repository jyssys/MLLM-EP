"""Reuse the immutable H200 launcher and add controlled vision or D observations."""

import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
import shutil

from ..mllm_epd_prefill_heavy_modality_discovery import runner as native


def main():
    launch = native.launch_models
    survey = native.survey
    profile = native.profile
    build_collector = native.build_collector
    enable_graph = os.environ.get("MLLM_IDLE_GRAPH") == "1"
    graph_source = Path(__file__).with_name("graph_activity.cc")
    window = os.environ.get("MLLM_IDLE_PROFILE_WINDOW_S")

    if window:
        delay, duration = map(float, window.split(","))
        if delay <= 0 or duration <= 0:
            raise ValueError("Profile window requires positive delay and duration")

        @asynccontextmanager
        async def profile_window(tag, enabled, copies, period, p_state_only):
            receipts = {}
            if not enabled:
                yield receipts
                return

            async def record():
                await asyncio.sleep(delay)
                async with profile(tag, True, copies, period, p_state_only) as result:
                    await asyncio.sleep(duration)
                receipts.update(result)

            task = asyncio.create_task(record())
            try:
                yield receipts
            except BaseException:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise
            else:
                await task

        native.profile = profile_window

    def launch_models(group, ec, devices, media, args):
        start = group.start

        def launch_worker(role, command, **env):
            if role in native.PORTS:
                command = list(command)
                command[2] = f"{__package__}.server_entry"
                command[command.index("--worker-extension-cls")+1] = f"{__package__}.vision_worker.IdleWorker"
                env["MLLM_IDLE_GRAPH"] = "1" if role == "decode" and enable_graph else "0"
                if "MLLM_IDLE_FIXTURES" in os.environ:
                    env["MLLM_IDLE_FIXTURES"] = os.environ["MLLM_IDLE_FIXTURES"]
            return start(role, command, **env)

        group.start = launch_worker
        try:
            launch(group, ec, devices, media, args)
        finally:
            group.start = start

    def build(asset_repo, run, runtime):
        receipt = build_collector(asset_repo, run, runtime)
        if enable_graph:
            from .graph_observer import build as build_graph

            frozen = run/"source"/Path(*__package__.split("."))/"graph_activity.cc"
            shutil.copyfile(graph_source,frozen)
            build_graph(run, runtime)
        return receipt

    async def run_survey(group, plan, manifests, routes, args):
        if plan.get("physical"):
            from .physical import campaign

            await campaign(group.run, plan, manifests, routes)
        else:
            if window:
                native.rt.write(group.run/"configs"/"observer_window.json", {
                    "delay_seconds": delay, "planned_duration_seconds": duration,
                    "anchor": "Async context immediately before native sampler and client launch",
                    "qualification": "Use actual common GPU ROI and compare the full clean/profile/clean cohort.",
                })
            await survey(group, plan, manifests, routes, args)

    native.launch_models, native.build_collector, native.survey = launch_models, build, run_survey
    native.main()


if __name__ == "__main__":
    main()
