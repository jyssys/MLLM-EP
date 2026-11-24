"""A second owned E-side process must not stop or adopt another GPU owner."""
import unittest


class CoResidencyGuardTest(unittest.TestCase):
    def lease(self,**kwargs):
        return dict(dict(gpu=4,role='encoder',ready=True,finished=False,pid=100,sid=100),**kwargs)

    def test_only_the_verified_encoder_session_is_accepted(self):
        from .servers import validate_co_resident_facts
        validate_co_resident_facts(self.lease(),4,[dict(gpu=4,pid=101,uid=1000,sid=100)],1000)

    def test_foreign_uid_or_session_is_rejected(self):
        from .servers import validate_co_resident_facts
        for row in [dict(gpu=4,pid=101,uid=999,sid=100),dict(gpu=4,pid=101,uid=1000,sid=102)]:
            with self.subTest(row=row),self.assertRaises(AssertionError):
                validate_co_resident_facts(self.lease(),4,[row],1000)

    def test_absent_finished_unready_and_wrong_role_are_rejected(self):
        from .servers import validate_co_resident_facts
        with self.assertRaises(AssertionError):
            validate_co_resident_facts(self.lease(),4,[],1000)
        for values in [dict(finished=True),dict(ready=False),dict(role='decode'),dict(gpu=5)]:
            with self.subTest(values=values),self.assertRaises(AssertionError):
                validate_co_resident_facts(self.lease(**values),4,[dict(gpu=4,pid=101,uid=1000,sid=100)],1000)


if __name__=='__main__':
    unittest.main()
