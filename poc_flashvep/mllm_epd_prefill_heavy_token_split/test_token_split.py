"""CPU contracts for exact multimodal prefix ownership and NIXL continuation."""
from types import SimpleNamespace
import unittest


def core(length=12, prefix=7, marked=True):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import EngineCoreRequest
    from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
    from .test_metadata_handoff import rendered
    transfer = dict(do_remote_decode=True, do_remote_prefill=False)
    if marked:
        transfer['campaign_token_split_v1'] = dict(prefix_tokens=prefix, full_prompt_tokens=length)
    params = SamplingParams(max_tokens=1, temperature=0,
                            extra_args=dict(kv_transfer_params=transfer))
    return EngineCoreRequest(request_id='real-core:P', prompt_token_ids=list(range(length)),
        mm_features=[MultiModalFeatureSpec(identifier='image', modality='image', mm_hash='image',
            data=rendered()[1][0]['mm_kwargs']['image'][0], mm_position=PlaceholderRange(2, 8))],
        sampling_params=params, pooling_params=None, arrival_time=1., lora_request=None,
        cache_salt=None, data_parallel_rank=None)


class TokenPrefixContractTest(unittest.TestCase):
    def test_native_unmarked_request_is_the_identical_object(self):
        from .token_split import truncate_producer_request
        value = core(marked=False)
        original_ids = value.prompt_token_ids
        self.assertIs(truncate_producer_request(value, 'prefill'), value)
        self.assertIs(value.prompt_token_ids, original_ids)

    def test_real_engine_request_keeps_all_multimodal_features_inside_image(self):
        from .token_split import truncate_producer_request
        value = core()
        original_features = value.mm_features
        original_ids = value.prompt_token_ids
        result = truncate_producer_request(value, 'prefill')
        self.assertEqual(result.prompt_token_ids, original_ids[:7])
        self.assertIs(result.mm_features, original_features)
        self.assertEqual(result.mm_features[0].mm_position.length, 8)
        self.assertEqual(original_ids, list(range(12)))

    def test_decoder_keeps_full_prompt_and_original_position_metadata(self):
        from .token_split import truncate_producer_request
        value = core()
        self.assertIs(truncate_producer_request(value, 'decode'), value)
        self.assertEqual(value.prompt_token_ids, list(range(12)))

    def test_real_engine_wire_roundtrip_keeps_partial_image_span(self):
        from vllm.v1.engine import EngineCoreRequest
        from vllm.v1.serial_utils import MsgpackEncoder, MsgpackDecoder
        from .token_split import truncate_producer_request
        value = truncate_producer_request(core(), 'prefill')
        buffers = [bytearray(memoryview(b)) for b in MsgpackEncoder().encode(value)]
        decoder = MsgpackDecoder(EngineCoreRequest, share_mem=False)
        decoder.pin_tensors = False
        decoded = decoder.decode(buffers)
        self.assertEqual(decoded.prompt_token_ids, list(range(7)))
        self.assertEqual(decoded.mm_features[0].mm_position.length, 8)
        self.assertEqual(decoded.sampling_params.extra_args, value.sampling_params.extra_args)

    def test_prefix_cannot_be_zero_full_prompt_or_beyond_prompt(self):
        from .token_split import truncate_producer_request
        for prefix in [0, 12, 13, -1, True, 3.5]:
            with self.subTest(prefix=prefix), self.assertRaises(AssertionError):
                truncate_producer_request(core(prefix=prefix), 'prefill')

    def test_full_length_mismatch_and_nonproducer_fail_closed(self):
        from .token_split import truncate_producer_request
        value = core()
        value.sampling_params.extra_args['kv_transfer_params']['campaign_token_split_v1']['full_prompt_tokens'] = 13
        with self.assertRaises(AssertionError):
            truncate_producer_request(value, 'prefill')
        value = core()
        value.sampling_params.extra_args['kv_transfer_params']['do_remote_decode'] = False
        with self.assertRaises(AssertionError):
            truncate_producer_request(value, 'prefill')


class RemotePrefixCountTest(unittest.TestCase):
    def request(self, prefix=7, remote=7, marked=True):
        params = dict(do_remote_prefill=True, remote_num_tokens=remote, remote_block_ids=[[1]])
        if marked:
            params['campaign_token_split_v1'] = dict(prefix_tokens=prefix, full_prompt_tokens=12)
        return SimpleNamespace(request_id='real-core:D', prompt_token_ids=list(range(12)),
                               num_prompt_tokens=12, kv_transfer_params=params)

    def test_marked_decode_matches_only_the_actual_remote_prefix(self):
        from .token_split import matched_remote_prefix
        self.assertEqual(matched_remote_prefix(self.request(), 0), (7, True))
        self.assertEqual(matched_remote_prefix(self.request(), 3), (4, True))
        self.assertEqual(matched_remote_prefix(self.request(), 7), (0, False))

    def test_native_and_already_imported_requests_delegate(self):
        from .token_split import matched_remote_prefix
        self.assertIsNone(matched_remote_prefix(self.request(marked=False), 0))
        value = self.request()
        value.kv_transfer_params['do_remote_prefill'] = False
        self.assertIsNone(matched_remote_prefix(value, 0))

    def test_overstated_or_missing_remote_prefix_fails(self):
        from .token_split import matched_remote_prefix
        for remote in [6, 8, None]:
            with self.subTest(remote=remote), self.assertRaises(AssertionError):
                matched_remote_prefix(self.request(remote=remote), 0)

    def test_real_connector_facade_does_not_call_full_prompt_matching(self):
        from .token_nixl import ExactTokenNixlConnector
        connector = ExactTokenNixlConnector.__new__(ExactTokenNixlConnector)
        connector.connector_scheduler = SimpleNamespace(
            get_num_new_matched_tokens=lambda *_: (11, True))
        self.assertEqual(connector.get_num_new_matched_tokens(self.request(), 0), (7, True))
        self.assertEqual(connector.get_num_new_matched_tokens(self.request(marked=False), 0), (11, True))


class ProxyPartitionTest(unittest.TestCase):
    def test_native_has_no_partition(self):
        from .proxy import token_prefix
        self.assertIsNone(token_prefix(8000))

    def test_static_fraction_uses_original_full_token_count(self):
        from .proxy import token_prefix
        self.assertEqual(token_prefix(8000, .25), 6000)
        self.assertEqual(token_prefix(8000, .1), 7200)
        self.assertEqual(token_prefix(3, .9), 1)

    def test_fixed_boundary_is_not_silently_aligned_or_replaced(self):
        from .proxy import token_prefix
        self.assertEqual(token_prefix(8000, fixed_prefix=1777), 1777)
        with self.assertRaises(AssertionError):
            token_prefix(8000, .2, 1777)


class BoundaryTeacherTest(unittest.TestCase):
    def test_only_an_explicit_teacher_changes_its_boundary_crossing_chunk(self):
        from .token_split import TEACHER_KEY, teacher_query_limit
        request = SimpleNamespace(kv_transfer_params=None, prompt_token_ids=list(range(12)), num_prompt_tokens=12)
        self.assertEqual(teacher_query_limit(request, 0, 8), 8)
        request.kv_transfer_params = {TEACHER_KEY: dict(prefix_tokens=7, full_prompt_tokens=12)}
        self.assertEqual(teacher_query_limit(request, 0, 8), 7)
        self.assertEqual(teacher_query_limit(request, 4, 8), 3)
        self.assertEqual(teacher_query_limit(request, 7, 5), 5)
        self.assertEqual(teacher_query_limit(request, 12, 1), 1)

    def test_teacher_cannot_silently_become_a_remote_or_partial_request(self):
        from .token_split import TEACHER_KEY, teacher_query_limit
        request = SimpleNamespace(kv_transfer_params={TEACHER_KEY: dict(prefix_tokens=7, full_prompt_tokens=12),
            'do_remote_prefill': True}, prompt_token_ids=list(range(12)), num_prompt_tokens=12)
        with self.assertRaises(AssertionError):
            teacher_query_limit(request, 0, 8)


if __name__ == '__main__':
    unittest.main()
