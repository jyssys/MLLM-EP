"""Keep socket writes out of the open-loop arrival and token-receive loop."""
import queue
import threading


class Ingress:
    def __init__(self,channels):
        self.errors=[];self.queues={role:queue.Queue() for role in channels};self.threads=[]
        for role,channel in channels.items():
            def send(role=role,channel=channel):
                try:
                    while True:
                        value=self.queues[role].get()
                        if value is None:return
                        channel.send(value)
                except BaseException as error:self.errors.append(repr(error))
            thread=threading.Thread(target=send,daemon=True);thread.start();self.threads.append(thread)
    def send(self,role,value):
        self.check();self.queues[role].put(value)
    def check(self):assert not self.errors,self.errors
    def close(self):
        for q in self.queues.values():q.put(None)
        for t in self.threads:t.join(5)
        self.check()
