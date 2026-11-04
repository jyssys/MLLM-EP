import ctypes
import unittest
from .collector_buffers import set_size_attribute


class FakeSetter:
    def __call__(self, attribute, size, value):
        self.attribute = attribute
        self.size = ctypes.cast(size, ctypes.POINTER(ctypes.c_size_t)).contents.value
        self.value = ctypes.cast(value, ctypes.POINTER(ctypes.c_size_t)).contents.value
        return 0


class CollectorBufferTests(unittest.TestCase):
    def test_size_is_a_pointer(self):
        setter = FakeSetter()
        set_size_attribute(setter, 6, 16)
        self.assertEqual(setter.argtypes[1], ctypes.POINTER(ctypes.c_size_t))
        self.assertEqual((setter.attribute, setter.size, setter.value), (6, ctypes.sizeof(ctypes.c_size_t), 16))


if __name__ == '__main__':
    unittest.main()
