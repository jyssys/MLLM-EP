"""Physical CUDA IPC vision and controlled native decode interference trials."""

import asyncio
import json
import os
from pathlib import Path
import time

import aiohttp

from ..mllm_epd_h200_prefill_heavy_batch_matched import capacity, runtime as rt
from ..mllm_epd_prefill_heavy_token_split.idle_admission import FirstToken
from ..mllm_specific_disaggregation_discovery.native_epd import prompt_messages

E, P, D = 18840,18841,18842


class Probe:
    def __init__(self, session, path):
        self.session = session
        self.path = path

    async def rpc(self, port, op, fixture=None, **more):
        start = time.monotonic()
        body = {"op":op, "fixture":fixture, **more}
        async with self.session.post(f"http://127.0.0.1:{port}/campaign/idle/vision",json=body) as reply:
            payload = await reply.text()
            if reply.status!=200:
                raise RuntimeError(f"worker={port}, operation={op}, error={payload[:900]}")
            result, = json.loads(payload)
        result["rpc_wall_ms"] = 1000*(time.monotonic()-start)
        return result

    def record(self, value):
        with self.path.open("a") as stream:
            stream.write(json.dumps(value) + "\n")

    async def retire(self, port, packet):
        return await self.rpc(port,"release",ticket=packet["ticket"])

    async def full(self,name):
        start = time.monotonic()
        encode = await self.rpc(D,"full_send",name)
        packet = encode.pop("packet")
        result = await self.rpc(E,"image_output",name,packet=packet)
        await self.retire(D,packet)
        return {"elapsed_ms":1000*(time.monotonic()-start),"sent_bytes":packet["bytes"],
                "D":encode,"E":result}

    async def split(self,name,cut):
        start = time.monotonic()
        patch = await self.rpc(E,"prepare",name,cut=cut)
        packet = patch.pop("packet")
        kept,remote = await asyncio.gather(
            self.rpc(E,"kept",name,cut=cut),
            self.rpc(D,"remote",name,cut=cut,packet=packet))
        back = remote.pop("packet")
        await self.retire(E,packet)
        final = await self.rpc(E,"join",name,cut=cut,packet=back)
        await self.retire(D,back)
        return {"elapsed_ms":1000*(time.monotonic()-start),
                "sent_bytes":packet["bytes"],"received_bytes": back["bytes"],
                "E_patch":patch,"E_kept":kept,"D":remote,"E_join":final}

    async def images(self):
        left = ["image_0","image_2","image_5"]
        right = ["image_1","image_3","image_4"]
        start = time.monotonic()
        e,d = await asyncio.gather(
            self.rpc(E,"balanced_group","group_all",members=left,wire=False),
            self.rpc(D,"balanced_group","group_all",members=right,wire=True))
        packet = d.pop("packet")
        c = await self.rpc(E,"join_images","group_all",remote_members=right,
                           original_order=[f"image_{i}" for i in range(6)],packet=packet)
        await self.retire(D,packet)
        return {"elapsed_ms":1000*(time.monotonic()-start),"remote_images":right,
                "sent_bytes":packet["bytes"],"E":e,"D":d,"join":c}

    async def native_post(self, port, path, payload, headers=None):
        return await capacity.json_reply(
            self.session.post(f"http://127.0.0.1:{port}/{path}",json=payload,headers=headers),path)

    async def with_decode(self,row,tag,mode,fixture):
        client = rt.module("client")
        body = client.payload(row,tag,128)
        await capacity.stage_request(self.session,E,row,tag,"E:full",client,output_count=128)
        original = await capacity.stage_request(self.session,P,row,tag,"P",client,export=True,output_count=128)
        body["kv_transfer_params"] = original["kv_transfer_params"]
        observed = []
        request_start = None
        first_event = asyncio.Event()
        vision_submit = vision_finish = None

        async def generation():
            nonlocal request_start
            request_start = time.monotonic()
            async with self.session.post(f"http://127.0.0.1:{D}/v1/chat/completions",
                                         json=body,headers=capacity.headers(tag,"D")) as response:
                response.raise_for_status()
                detector = FirstToken()
                carry = b""
                async for chunk in response.content.iter_any():
                    if detector.feed(chunk):
                        first_event.set()
                    carry += chunk
                    while b"\n" in carry:
                        line,carry = carry.split(b"\n",1)
                        if not line.startswith(b"data: ") or line[6:].strip()==b"[DONE]":
                            continue
                        value = json.loads(line[6:])
                        opts = value.get("choices") or []
                        if opts and (ids:=opts[0].get("token_ids") or opts[0].get("delta",{}).get("token_ids") or []):
                            observed.extend((i,time.monotonic()) for i in ids)

        async def work():
            nonlocal vision_submit,vision_finish
            if mode=="decoding":
                async with asyncio.timeout(60):
                    await first_event.wait()
            vision_submit = time.monotonic()
            result = await self.rpc(D,"full_send",fixture)
            vision_finish = time.monotonic()
            return result

        extra = None
        if mode=="clean":
            await generation()
        elif mode=="soon":
            task = asyncio.create_task(work())
            await asyncio.sleep(.003)
            await generation()
            extra = await task
        elif mode=="decoding":
            _,extra = await asyncio.gather(generation(),work())
        else:
            raise ValueError("Unknown decode interference mode")
        if len(observed)!=128:
            raise RuntimeError(f"Decode output did not satisfy fixed128: {len(observed)}")
        if extra is not None:
            packet=extra.pop("packet")
            extra["embedding"] = await self.rpc(E,"image_output",fixture,packet=packet)
            await self.retire(D,packet)
        key = capacity.headers(tag,"E")["x-campaign-metadata-key"]
        await self.native_post(E,"campaign/ec_ipc/release",{"key":key})
        times = [t for _,t in observed]
        return {"mode":mode,"fixture":fixture,"decode_submit":request_start,
                "first_token":times[0],"last_token":times[-1],
                "vision_submit":vision_submit,"vision_complete":vision_finish,
                "D_submit_first_ms":1000*(times[0]-request_start),
                "tpot_ms":1000*(times[-1]-times[0])/(len(times)-1),
                "itl_ms":[1000*(b-a) for a,b in zip(times,times[1:])],
                "token_ids":[i for i,_ in observed],"vision":extra}


async def campaign(run,plan,manifests,routes):
    client = rt.module("client")
    client.messages = prompt_messages
    row = next(iter(manifests.values()))[0]
    await client.execute([row],[routes["native"]],"vision_warmup",1,2026112809,16,serial=True,
                         unique_media_occurrence=True)
    names = plan["physical"]["fixtures"]
    count = plan["physical"].get("repetitions",4)
    expected = [entry for entry in json.loads(
        (Path(os.environ["MLLM_IDLE_FIXTURES"])/"manifest.json").read_text())["entries"]
        if entry["id"] in names]
    rt.write(run/"configs"/"physical_fixtures.json",expected)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180),trust_env=False) as session:
        probe = Probe(session,run/"raw"/"vision_activity.jsonl")
        layouts = {}
        for name in names if count else ():
            left,right = await asyncio.gather(probe.rpc(E,"load",name),probe.rpc(D,"load",name))
            for field in ("grid","backend","dtype","weights_bytes","cut_layer","window_rows"):
                if left[field]!=right[field]:
                    raise RuntimeError(f"Vision worker contract mismatch at {field}: {left[field]} vs {right[field]}")
            probe.record({"fixture":name,"E":left,"D":right,"kind":"layout"})
            layouts[name] = left["window_rows"]
            await probe.rpc(E,"native",name,reference=True)
            await probe.full(name)
            for repeat in range(count):
                a = await probe.rpc(E,"native",name)
                b = await probe.full(name)
                probe.record({"fixture":name,"repeat":repeat,"kind":"image","E_native":a,"D_full":b})
            print("IMAGE_COMPLETE",name,count,flush=True)
        if count and "group_all" in names and all(f"image_{i}" in names for i in range(6)):
            await probe.images()
            for repeat in range(count):
                reference = await probe.rpc(E,"native","group_all")
                measurement = await probe.images()
                probe.record({"kind":"image_balance","fixture":"group_all","repeat":repeat,
                              "E_native":reference,"E_D":measurement})
        for name in names if count and plan["physical"].get("window", True) else ():
            windows = layouts[name]
            cuts = set()
            for fraction in (.25,.5,.75):
                if len(windows)>1:
                    target = sum(windows)*(1-fraction)
                    cut = min(range(1,len(windows)),key=lambda k:abs(sum(windows[:k])-target))
                    cuts.add(cut)
            if cuts:
                initial_cut = min(cuts)
                await probe.rpc(E,"local_same_cut",name,cut=initial_cut)
                await probe.rpc(E,"local_frontier",name,cut=initial_cut)
                await probe.split(name,initial_cut)
            for repeat in range(count):
                a = await probe.rpc(E,"native",name)
                for cut in sorted(cuts):
                    local = await probe.rpc(E,"local_same_cut",name,cut=cut)
                    frontier = await probe.rpc(E,"local_frontier",name,cut=cut)
                    hybrid = await probe.split(name,cut)
                    probe.record({"fixture":name,"repeat":repeat,"kind":"window","cut":cut,
                                  "E_native":a,"E_same_cut":local,"E_frontier":frontier,"E_D":hybrid})
            print("WINDOW_COMPLETE",name,len(windows),count,flush=True)
        if not count:
            for name in names:
                await probe.rpc(E,"load",name)
                await probe.rpc(D,"load",name)
                await probe.rpc(E,"native",name,reference=True)
                await probe.full(name)
        for repeats in range(plan["physical"].get("interference_repetitions",0)):
            for name,mode in (("image_0","clean"),("image_0","soon"),("image_0","decoding"),
                              ("image_4","soon"),("image_4","decoding")):
                if name not in names:
                    continue
                witness = await probe.with_decode(row,f"vision_{repeats}_{name}_{mode}",mode,name)
                probe.record({"kind":"decode_interference", "repeat":repeats, **witness})
                print("INTERFERENCE_COMPLETE",name,mode,witness["tpot_ms"],flush=True)
    owner = rt.module("ipc_ec_control")
    owner.ownership_empty("physical", "cuda_ipc")
