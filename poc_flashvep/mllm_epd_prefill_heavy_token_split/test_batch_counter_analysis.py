"""Do not convert missing query data or cleanup calls into actual LLM batches."""
import unittest

from .batch_counter_analysis import summarize_worker


class BatchSummaryTest(unittest.TestCase):
    def value(self):
        return dict(valid=True,original_methods_restored=True,errors={},prepared_query_histogram=[
            dict(requests=2,prompt_only_requests=1,decode_only_requests=1,mixed_requests=0,
                query_rows=5,prompt_query_rows=4,decode_query_rows=1,calls=3)],
            worker_schedule_histogram=[dict(scheduled_query_rows=5,calls=3),dict(scheduled_query_rows=0,calls=4)],
            encoder_output_histogram=[])

    def test_empty_cleanup_does_not_dilute_the_prepared_batch(self):
        result=summarize_worker(self.value(),'decode')
        self.assertEqual(result['mean_prepared_requests_per_call'],2.)
        self.assertEqual(result['empty_worker_control_calls'],4)
        self.assertEqual(result['total_prompt_query_rows'],12)

    def test_encoder_only_can_have_no_llm_input_preparation_but_decode_cannot(self):
        value=self.value();value['prepared_query_histogram']=[]
        self.assertTrue(summarize_worker(value,'encoder')['counter_valid'])
        self.assertFalse(summarize_worker(value,'decode')['counter_valid'])

    def test_nonconserving_query_counts_fail_without_a_fake_correction(self):
        value=self.value();value['prepared_query_histogram'][0]['decode_query_rows']=3
        with self.assertRaises(AssertionError):summarize_worker(value,'decode')


if __name__=='__main__':unittest.main()
