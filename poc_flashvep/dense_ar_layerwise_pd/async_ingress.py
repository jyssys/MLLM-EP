"""Keep blocked input RPC writes off the token-delivery loop."""
import queue,threading,time


class AsyncIngress:
    def __init__(self,channels,record):
        self.queues={role:queue.Queue() for role in channels};self.errors=[];self.threads=[]
        def run(role,channel):
            try:
                while True:
                    message=self.queues[role].get()
                    if message is None:return
                    start=time.monotonic();channel.send(message);end=time.monotonic()
                    commands=message['commands'] if message['kind']=='add_batch' else [message]
                    for command in commands:
                        rid=command['request_id']
                        if rid.startswith('q') and rid[1:].isdigit():record(role,dict(sample_id=int(rid[1:]),role=role,
                            actual_send_start=start,actual_send_end=end,request_id=rid))
            except BaseException as e:self.errors.append((role,repr(e)))
        for role,channel in channels.items():
            thread=threading.Thread(target=run,args=(role,channel),daemon=True,name='dense-ingress-'+role)
            thread.start();self.threads.append(thread)
    def submit(self,role,message):self.check();self.queues[role].put(message)
    def check(self):assert not self.errors,('Input RPC writer failed',self.errors)
    def close(self):
        for q in self.queues.values():q.put(None)
        for thread in self.threads:thread.join(2)
        return not any(t.is_alive() for t in self.threads)
