import unittest

from poc_ouro_parallelism_deep_discovery.scripts.validate_baseline_extension_integrity import check_record


class Tokenizer:
    def encode(self,text,add_special_tokens=False):
        return [ord(c) for c in text]
    def decode(self,tokens,skip_special_tokens=False):
        return ''.join(chr(t) for t in tokens)


class IntegrityTests(unittest.TestCase):
    def fixture(self):
        return dict(prompt='a',stop_strings=['!']), dict(prompt_ids=[97],output_ids=[49,33],
                                                       text='1!',finish_reason='stop')
    def test_valid_stop(self):
        row,value=self.fixture()
        self.assertEqual(check_record(row,value,Tokenizer())['matched_frozen_stop_count'],1)
    def test_changed_prompt_fails(self):
        row,value=self.fixture()
        value['prompt_ids']=[98]
        with self.assertRaises(ValueError):
            check_record(row,value,Tokenizer())
    def test_unbound_text_fails(self):
        row,value=self.fixture()
        value['text']='different'
        with self.assertRaises(ValueError):
            check_record(row,value,Tokenizer())
    def test_stop_without_configured_suffix_fails(self):
        row,value=self.fixture()
        row['stop_strings']=[]
        with self.assertRaises(ValueError):
            check_record(row,value,Tokenizer())
    def test_generation_after_terminal_fails(self):
        for tokens in ([49,33,50,33],[49,0,50,33]):
            row,value=self.fixture()
            value.update(output_ids=list(tokens),text=Tokenizer().decode(tokens))
            with self.assertRaises(ValueError):
                check_record(row,value,Tokenizer())
    def test_eos_precedes_cap(self):
        row,value=self.fixture()
        row['stop_strings']=[]
        value.update(output_ids=[49]*8191+[0],text='1'*8191+'\x00',finish_reason='eos')
        self.assertTrue(check_record(row,value,Tokenizer())['finish_matches'])


if __name__=='__main__':
    unittest.main()
