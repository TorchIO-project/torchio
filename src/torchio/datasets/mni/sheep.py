"""MNI Sheep atlas."""

from __future__ import annotations

import shutil
import urllib.parse

from ...data import ScalarImage
from ...download import compress
from ...download import download_and_extract_archive
from .mni import SubjectMNI


class Sheep(SubjectMNI):
    """Ovine brain atlas at 0.5 mm resolution.

    See `the MNI website
    <https://nist.mni.mcgill.ca/?page_id=714>`_ for more information.
    """

    def __init__(self) -> None:
        self.name = "NIFTI_ovine_05mm"
        self.url_dir = urllib.parse.urljoin(self.url_base, "sheep/")
        self.filename = f"{self.name}.zip"
        self.url = urllib.parse.urljoin(self.url_dir, self.filename)
        t1_nii_path = self.download_root / "ovine_model_05.nii"
        t1_niigz_path = self.download_root / "ovine_model_05.nii.gz"
        if not self.download_root.is_dir():
            download_and_extract_archive(
                self.url,
                download_root=self.download_root,
                filename=self.filename,
            )
            shutil.rmtree(self.download_root / "masks")
            for path in self.download_root.iterdir():
                if path == t1_nii_path:
                    compress(t1_nii_path, t1_niigz_path)
                path.unlink()
        super().__init__(t1=ScalarImage(t1_niigz_path))
