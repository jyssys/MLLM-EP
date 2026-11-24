"""Context leaks must fail even when original request IDs differ."""
import unittest

from .dataset_integrity_audit import assert_disjoint


class ContextAuditTest(unittest.TestCase):
    def row(self,name,sha,pixel,document=None):
        return dict(request_id=name,images=[dict(SHA256=sha,pixel_SHA256=pixel)],document_id=document)

    def test_distinct_ids_and_contexts_pass(self):
        result=assert_disjoint([('discovery',[self.row('a','1','p')]),('heldout',[self.row('b','2','q')])])
        self.assertEqual([r['requests'] for r in result],[1,1])

    def test_reencoded_identical_pixels_are_still_a_context_leak(self):
        with self.assertRaises(AssertionError):
            assert_disjoint([('discovery',[self.row('a','1','p')]),('heldout',[self.row('b','2','p')])])

    def test_source_independent_document_id_cannot_cross_splits(self):
        with self.assertRaises(AssertionError):
            assert_disjoint([('discovery',[self.row('a','1','p','doc')]),('heldout',[self.row('b','2','q','doc')])])

    def test_duplicate_requests_inside_a_split_are_not_hidden_by_sets(self):
        row=self.row('a','1','p')
        with self.assertRaises(AssertionError):assert_disjoint([('discovery',[row,row])])


if __name__=='__main__':unittest.main()
