"""Exercise publication with a metadata fixture at the observed raw-file scale."""
import hashlib
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
from .common import ROOT, REPO, write, preserve
from .evidence_archive import files
from . import publication


def main():
    preserve()
    paths, _ = files()
    # Only path lengths and record counts reproduce this metadata-size failure.
    # These synthetic digest values are NOT hashes of the raw evidence.
    rows = [dict(path=str(p.relative_to(REPO)), bytes=p.stat().st_size,
        SHA256=hashlib.sha256(str(p.relative_to(REPO)).encode()).hexdigest()) for p in paths]
    result = dict(time=time.time(), fixture_records=len(rows),
        scope='CPU publication fixture only. Paths reflect the observed file scale. '
              'Digest values hash path strings,not file contents. No real archive or scientific proof is created.')
    with tempfile.TemporaryDirectory(prefix='publication_scale_fixture_', dir=ROOT/'tools') as directory:
        base=Path(directory)
        target=base/'campaign'
        (target/'state').mkdir(parents=True)
        source=base/'source'
        source.mkdir()
        archive=target/'archives/evidence_archive_manifest_v1.json'
        write(archive,dict(verified_lossless=True,local_members=rows,public_archive=dict(parts=[]),
            fixture_only=True,scope=result['scope']))
        result['expanded_metadata_bytes']=archive.stat().st_size
        try:
            with patch.multiple(publication,ROOT=target,REPO=base,SOURCE=source,DOCUMENTS=()):
                selected=publication.prepare()
            result.update(passed=True,maximum_selected_file_bytes=max(p.stat().st_size for p in selected),
                selected_expanded_manifest=archive in selected,
                selected_files=[str(p.relative_to(base)) for p in selected])
        except AssertionError as error:
            result.update(passed=False,error=repr(error))
    preserve()
    suffix='green' if result['passed'] else 'red'
    write(ROOT/'analysis'/f'publication_manifest_scale_{suffix}_v1.json',result)
    print('PUBLICATION_METADATA_SCALE',json.dumps(result),flush=True)
    assert result['passed'],'The actual publication selector cannot publish the observed metadata scale'


if __name__=='__main__':
    main()
