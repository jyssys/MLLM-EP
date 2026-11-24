"""Bound observation by completed requests without changing serving arrivals."""
import asyncio
import time


def completed_requests(path):
    if not path.exists():return 0
    with path.open() as stream:return sum(line.endswith('\n') and bool(line.strip()) for line in stream)


async def collect(path,start,stop,urls,name,rpc,finished,state_interval_ms=0,poll_seconds=.05):
    assert 0<start<stop and urls and poll_seconds>0
    records={};opened=[];audit=dict(start_completed=start,stop_completed=stop)
    async def wait_for(count):
        while completed_requests(path)<count:
            assert not finished.is_set(),'Serving ended before the frozen observation boundary'
            await asyncio.sleep(poll_seconds)
    try:
        await wait_for(start)
        audit.update(begin_completed=completed_requests(path),begin_monotonic=time.monotonic())
        assert audit['begin_completed']<stop,'The frozen observation interval already ended'
        for url in urls:
            reply=await asyncio.to_thread(rpc,url,'profile/begin',dict(name=name,include_copies=True,
                state_interval_ms=state_interval_ms))
            records[url]=dict(begin=reply);opened.append(url)
        audit['all_collectors_open_completed']=completed_requests(path)
        assert audit['all_collectors_open_completed']<stop,'Collector setup consumed the frozen interval'
        await wait_for(stop)
        audit.update(end_completed=completed_requests(path),end_monotonic=time.monotonic())
    finally:
        for url in reversed(opened):
            try:records[url]['end']=await asyncio.to_thread(rpc,url,'profile/end',{})
            except Exception as error:records[url]['close_error']=type(error).__name__+': '+str(error)
    return records,audit
