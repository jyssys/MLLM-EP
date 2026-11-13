import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from . import destination16 as module


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.patch=patch.object(module,'ROOT',self.root);self.patch.start()
    def tearDown(self):self.patch.stop();self.temp.cleanup()
    def row(self):
        sha=hashlib.sha256(b'exact test state').hexdigest()
        return dict(event='VALID_QUERY_STATE',external_request_id='q0',query_start=0,query_tokens=8000,
            KV=[dict(layer=i,SHA256=sha,bytes=32768000) for i in range(36)],final_hidden_SHA256=sha)
    def save(self,name,role,row):
        path=self.root/'raw'/name;path.mkdir(parents=True,exist_ok=True)
        (path/f'extension_{role}_test.jsonl').write_text(json.dumps(row)+'\n')
    def test_exact_independent_state(self):
        self.save('native','prefill',self.row());self.save('candidate','decode',self.row())
        self.assertTrue(module.reference_gate('candidate','native',['q0'])['passed'])
    def test_hidden_change_fails(self):
        r=self.row();r['final_hidden_SHA256']=hashlib.sha256(b'changed').hexdigest()
        self.save('native','prefill',self.row());self.save('candidate','decode',r)
        self.assertFalse(module.reference_gate('candidate','native',['q0'])['passed'])
    def test_incomplete_plane_set_cannot_pass(self):
        r=self.row();r['KV'].pop();self.save('native','prefill',r)
        with self.assertRaises(AssertionError):module.state_hashes('native','prefill')


if __name__=='__main__':unittest.main()
