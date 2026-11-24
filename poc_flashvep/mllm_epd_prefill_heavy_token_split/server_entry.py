"""Official API server with localhost-only, fixed observer RPC endpoints."""
import uvloop
from fastapi import Request
from vllm.entrypoints.openai import api_server


def api_host_policy():
    import os
    import sys
    import time
    from .common import ROOT,write
    before=sys.getswitchinterval()
    value=os.environ.get('MLLM_EPD_API_SWITCH_INTERVAL')
    if value is not None:
        value=float(value)
        assert .0001<=value<=.01
        sys.setswitchinterval(value)
    record=dict(time=time.time(),pid=os.getpid(),role=os.environ.get('MLLM_EPD_ROLE'),
        GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),before_seconds=before,
        actual_seconds=sys.getswitchinterval(),explicit=value is not None,
        scope='API process Python thread switch interval only. This function runs in main, not in EngineCore workers.')
    write(ROOT/'state'/f'api_host_policy_{os.getpid()}.json',record)
    return record


def main():
    api_host_policy()
    from . import host_timeline
    api_timeline=host_timeline.APITimeline();api_timeline.install()
    from .processor_handoff import install as install_processor_handoff
    install_processor_handoff()
    host_timeline.API_RECORDER=api_timeline
    original=api_server.build_app
    def build_app(*args,**kwargs):
        app=original(*args,**kwargs)
        @app.post('/campaign/timeline/begin')
        async def timeline_begin(request:Request):
            body=await request.json()
            workers=await app.state.engine_client.collective_rpc('campaign_timeline_begin',timeout=30,args=(body['name'],))
            return dict(workers=workers,api=await api_timeline.begin(body['name']))
        @app.post('/campaign/timeline/end')
        async def timeline_end():
            api=await api_timeline.close()
            workers=await app.state.engine_client.collective_rpc('campaign_timeline_end',timeout=60)
            return dict(workers=workers,api=api)
        @app.post('/campaign/profile/begin')
        async def begin(request:Request):
            body=await request.json()
            return await app.state.engine_client.collective_rpc('campaign_profile_begin',
                timeout=60,args=(body['name'],body.get('include_copies',True)))
        @app.post('/campaign/profile/end')
        async def end():
            return await app.state.engine_client.collective_rpc('campaign_profile_end',timeout=120)
        @app.get('/campaign/worker_info')
        async def info():
            return await app.state.engine_client.collective_rpc('campaign_worker_info',timeout=30)
        @app.post('/campaign/runtime_state')
        async def runtime_state():
            return await app.state.engine_client.collective_rpc('campaign_runtime_state',timeout=30)
        @app.post('/campaign/vision_fingerprint')
        async def vision_fingerprint():
            return await app.state.engine_client.collective_rpc('campaign_vision_fingerprint',timeout=120)
        @app.post('/campaign/llm_probe/begin')
        async def llm_probe_begin(request:Request):
            body=await request.json()
            return await app.state.engine_client.collective_rpc('campaign_llm_probe_begin',timeout=60,
                args=(body['name'],body.get('max_logits',16)))
        @app.post('/campaign/llm_probe/end')
        async def llm_probe_end():
            return await app.state.engine_client.collective_rpc('campaign_llm_probe_end',timeout=120)
        @app.post('/campaign/encoder_capture/begin')
        async def capture_begin(request:Request):
            body=await request.json()
            return await app.state.engine_client.collective_rpc('campaign_encoder_capture_begin',timeout=30,args=(body['name'],))
        @app.post('/campaign/encoder_capture/end')
        async def capture_end():
            return await app.state.engine_client.collective_rpc('campaign_encoder_capture_end',timeout=60)
        return app
    api_server.build_app=build_app
    api_server.cli_env_setup()
    parser=api_server.FlexibleArgumentParser(description='Official vLLM API with fixed campaign observation endpoints')
    parser=api_server.make_arg_parser(parser);args=parser.parse_args()
    api_server.validate_parsed_serve_args(args)
    uvloop.run(api_server.run_server(args))


if __name__=='__main__':main()
