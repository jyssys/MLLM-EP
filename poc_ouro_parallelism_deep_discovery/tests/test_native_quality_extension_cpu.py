import unittest
from poc_ouro_parallelism_deep_discovery.contract import MODELS
from poc_ouro_parallelism_deep_discovery.scripts.native_quality_extension import check_environment,finish_reason
from poc_ouro_parallelism_deep_discovery.scripts.validate_native_quality_extension import check_record


class Tokenizer:
    eos_token_id=0
    def encode(self,text,add_special_tokens=False):return [1,2]
    def decode(self,tokens,skip_special_tokens=False):return 'answer' if tokens==[7,0] else 'answer Q:'


class Tests(unittest.TestCase):
    def fixture(self):
        row=dict(id='x',dataset='gsm8k',prompt='frozen',stop_strings=['Q:'])
        value=dict(id='x',dataset='gsm8k',size='1.4B',mode='fixed',TP=4,
            revision=MODELS['1.4B']['revision'],physical_gpus=[4,5,6,7],
            source_protocol_sha256='p',extension_contract_sha256='c',cap=8192,temperature=0,EOS=0,
            output_ids=[7,0],generated_tokens=2,prompt_ids=[1,2],text='answer',
            native_api_text='answer',native_finish_reason='stop',native_stop_reason=None,
            finish_reason='eos',evidence_class='QUALITY_ONLY')
        return row,value
    def check(self,row,value):return check_record(row,value,Tokenizer(),'1.4B','p','c')
    def test_no_control_environment(self):check_environment({'CUDA_VISIBLE_DEVICES':'4,5,6,7'})
    def test_reject_inherited_controls(self):
        for key in ('POC4_FAMILY','POC4_R','OURO_CAMPAIGN_STORAGE_CONTROL'):
            with self.subTest(key=key),self.assertRaises(ValueError):check_environment({key:'tied'})
    def test_completion_classification(self):
        self.assertEqual(finish_reason('stop',None,0),'eos')
        self.assertEqual(finish_reason('stop','Q:',0),'stop')
        self.assertEqual(finish_reason('length',None,0),'length')
        with self.assertRaises(ValueError):finish_reason('abort',None,0)
    def test_valid_record(self):
        row,value=self.fixture()
        self.assertTrue(self.check(row,value)['finish_matches'])
    def test_same_length_wrong_prompt(self):
        row,value=self.fixture();value['prompt_ids']=[2,1]
        with self.assertRaises(ValueError):self.check(row,value)
    def test_unbound_text(self):
        row,value=self.fixture();value['text']='different'
        with self.assertRaises(ValueError):self.check(row,value)
    def test_unsupported_completion(self):
        row,value=self.fixture();value['native_finish_reason']='abort'
        with self.assertRaises(ValueError):self.check(row,value)
    def test_false_stop(self):
        row,value=self.fixture();value.update(output_ids=[7,8],native_finish_reason='stop',
            native_stop_reason='unfrozen',finish_reason='stop',text='answer Q:')
        with self.assertRaises(ValueError):self.check(row,value)
    def test_eos_at_cap_is_not_length(self):
        row,value=self.fixture();value.update(output_ids=[7]*8191+[0],generated_tokens=8192,
            native_finish_reason='length',finish_reason='length',text='answer Q:')
        with self.assertRaises(ValueError):self.check(row,value)


if __name__=='__main__':unittest.main()
