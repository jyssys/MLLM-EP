"""Do not confuse the Colocated name with a numeric Layer cut."""
import re,unittest


class SelectionNames(unittest.TestCase):
    def test_only_numeric_cuts_enter_layer_selection(self):
        names=['primed_graph_serving_qwen8_L8000_O16_N64_s3_'+label+'_v32'
            for label in ('nativeA','nativeB','colocated','c35','c32','c28')]
        self.assertEqual([n.split('_s3_')[1].split('_')[0] for n in names if re.search(r'_c\d+_',n)],
            ['c35','c32','c28'])


if __name__=='__main__':unittest.main()
