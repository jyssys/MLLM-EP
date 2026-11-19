"""Bind the new evidence root before installing unchanged inference adapters."""
import os
if os.environ.get('DENSE_MATCHED_AXIS_BOOTSTRAP')=='1':
    from poc_flashvep.dense_ar_axis_sweetspot.common import bind_legacy
    bind_legacy()
    from poc_flashvep.dense_ar_matched_axis_pd import site_bootstrap
