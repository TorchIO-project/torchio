"""MNI Pediatric atlases."""

from __future__ import annotations

import urllib.parse

from ...data import LabelMap
from ...data import ScalarImage
from ...download import compress
from ...download import download_and_extract_archive
from .mni import SubjectMNI

SUPPORTED_YEARS = (
    (4.5, 18.5),
    (4.5, 8.5),
    (7, 11),
    (7.5, 13.5),
    (10, 14),
    (13, 18.5),
)


def _format_age(n: float) -> str:
    integer = int(n)
    decimal = int(10 * (n - integer))
    return f"{integer:02d}.{decimal}"


class Pediatric(SubjectMNI):
    """MNI pediatric atlases.

    See `the MNI website
    <https://nist.mni.mcgill.ca/pediatric-atlases-4-5-18-5y/>`_
    for more information.

    Args:
        years: Tuple of 2 ages. Possible values are: ``(4.5, 18.5)``,
            ``(4.5, 8.5)``, ``(7, 11)``, ``(7.5, 13.5)``,
            ``(10, 14)`` and ``(13, 18.5)``.
        symmetric: If ``True``, the left-right symmetric templates will be
            used. Otherwise, the asymmetric (natural) templates will be used.
    """

    def __init__(
        self,
        years: tuple[float, float],
        symmetric: bool = False,
    ) -> None:
        self.url_dir = "http://www.bic.mni.mcgill.ca/~vfonov/nihpd/obj1/"
        sym_string = "sym" if symmetric else "asym"
        if not isinstance(years, tuple) or years not in SUPPORTED_YEARS:
            message = f"Years must be a tuple in {SUPPORTED_YEARS}"
            raise ValueError(message)
        a, b = years
        self.file_id = f"{sym_string}_{_format_age(a)}-{_format_age(b)}"
        self.name = f"nihpd_{self.file_id}_nifti"
        self.filename = f"{self.name}.zip"
        self.url = urllib.parse.urljoin(self.url_dir, self.filename)
        if not self.download_root.is_dir():
            download_and_extract_archive(
                self.url,
                download_root=self.download_root,
                filename=self.filename,
            )
            (self.download_root / self.filename).unlink()
            for path in self.download_root.glob("*.nii"):
                compress(path)
                path.unlink()

        subject_kwargs = self._get_subject_kwargs(".nii.gz")
        super().__init__(**subject_kwargs)

    def _get_subject_kwargs(self, extension: str) -> dict:
        root = self.download_root
        return {
            "t1": ScalarImage(root / f"nihpd_{self.file_id}_t1w{extension}"),
            "t2": ScalarImage(root / f"nihpd_{self.file_id}_t2w{extension}"),
            "pd": ScalarImage(root / f"nihpd_{self.file_id}_pdw{extension}"),
            "mask": LabelMap(root / f"nihpd_{self.file_id}_mask{extension}"),
        }
