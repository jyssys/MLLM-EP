import unittest
from .inventory import memory_window_summary


class MemoryWindowTests(unittest.TestCase):
    def test_startup_is_not_a_measured_window_maximum(self):
        def row(time, allocated):
            return dict(time=time, device_total_bytes=1000, device_free_bytes=1000-allocated,
                allocator_allocated_bytes=allocated, allocator_reserved_bytes=allocated,
                allocator_peak_bytes=999, KV_used_blocks=2, KV_block_bytes=10)
        values = [row(0, 999), row(2, 100), row(3, 200), row(5, 998)]
        result = memory_window_summary(values, 1, 4)
        self.assertEqual(result['samples'], 2)
        self.assertEqual(result['sampled_CUDA_device_used_peak_bytes_HBM_percent'], 20)
        self.assertEqual(result['reported_lifetime_allocator_peak_bytes_HBM_percent'], 99.9)
        self.assertEqual(result['sampled_KV_request_allocation_peak_bytes_HBM_percent'], 2)
        self.assertEqual(memory_window_summary(values, 1, 4, 2000)[
            'sampled_CUDA_device_used_peak_bytes_HBM_percent'], 10)
        self.assertEqual(memory_window_summary(values, 10, 11)['status'], 'NOT_MEASURED')


if __name__ == '__main__':
    unittest.main()
