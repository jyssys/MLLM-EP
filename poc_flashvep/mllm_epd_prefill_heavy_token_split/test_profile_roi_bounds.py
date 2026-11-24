"""Unobserved time must not become physical idle in a shortened capture."""
import unittest

from .profile_analysis import observation_rois


def activity(left,right):
    anchor=lambda second:dict(host_lower_ns=second*1e9,host_upper_ns=second*1e9)
    return (None,None,None,dict(left=anchor(left),right=anchor(right)))


class ProfileROIBoundsTest(unittest.TestCase):
    def test_short_capture_uses_only_intersection_of_all_three_observers(self):
        values=dict(E=activity(25,85),P=activity(26,84),D=activity(27,83))
        rois=observation_rois(0,100,list(range(101)),values,True)
        self.assertNotIn('full_cohort',rois)
        self.assertEqual(rois['observed_common_ROI'],(27,83))
        self.assertEqual(rois['central_arrival_window'],(27,83))

    def test_normal_capture_keeps_the_original_full_cohort_contract(self):
        rois=observation_rois(0,100,list(range(101)),{},False)
        self.assertEqual(rois,dict(full_cohort=(0,100),central_arrival_window=(10,90)))

    def test_late_capture_does_not_fabricate_a_central_arrival_window(self):
        rois=observation_rois(0,200,list(range(101)),dict(E=activity(120,180),D=activity(121,181)),True)
        self.assertEqual(rois,dict(observed_common_ROI=(121,180)))

    def test_no_common_capture_is_rejected(self):
        with self.assertRaisesRegex(AssertionError,'No common'):
            observation_rois(0,100,list(range(101)),dict(E=activity(10,20),D=activity(30,40)),True)


if __name__=='__main__':unittest.main()
