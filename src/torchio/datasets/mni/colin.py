"""Colin27 MNI template."""

from __future__ import annotations

import urllib.parse
from typing import ClassVar

from ...data import LabelMap
from ...data import ScalarImage
from ...download import compress
from ...download import download_and_extract_archive
from .mni import SubjectMNI

TISSUES_2008 = {
    1: "Cerebro-spinal fluid",
    2: "Gray Matter",
    3: "White Matter",
    4: "Fat",
    5: "Muscles",
    6: "Skin and Muscles",
    7: "Skull",
    9: "Fat 2",
    10: "Dura",
    11: "Marrow",
    12: "Vessels",
}


class Colin27(SubjectMNI):
    """Colin27 MNI template.

    More information can be found in the website of the
    `1998 <https://nist.mni.mcgill.ca/colin-27-average-brain/>`_ and
    `2008 <http://www.bic.mni.mcgill.ca/ServicesAtlases/Colin27Highres>`_
    versions.

    Args:
        version: Template year. It can be ``1998`` or ``2008``.

    Warning:
        The resolution of the ``2008`` version is quite high. The
        subject instance will contain four images of size
        362 x 434 x 362, therefore applying a transform to
        it might take longer than expected.
    """

    NAME_TO_LABEL: ClassVar[dict[str, int]] = {
        name: label for label, name in TISSUES_2008.items()
    }

    def __init__(self, version: int = 1998) -> None:
        if version not in (1998, 2008):
            msg = f'Version must be 1998 or 2008, not "{version}"'
            raise ValueError(msg)
        self.version = version
        self.name = f"mni_colin27_{version}_nifti"
        self.url_dir = urllib.parse.urljoin(self.url_base, "colin27/")
        self.filename = f"{self.name}.zip"
        self.url = urllib.parse.urljoin(self.url_dir, self.filename)
        if not self.download_root.is_dir():
            download_and_extract_archive(
                self.url,
                download_root=self.download_root,
                filename=self.filename,
            )
            # Fix label map
            # https://github.com/TorchIO-project/torchio/issues/220
            if version == 2008:
                path = self.download_root / "colin27_cls_tal_hires.nii"
                cls_image = LabelMap(path)
                cls_image.set_data(cls_image.data.round().byte())
                cls_image.save(path)

            (self.download_root / self.filename).unlink()
            for path in self.download_root.glob("*.nii"):
                compress(path)
                path.unlink()

        subject_kwargs = self._get_subject_kwargs(
            self.download_root,
            extension=".nii.gz",
        )
        super().__init__(**subject_kwargs)

    def _get_subject_kwargs(self, download_root, extension):
        if self.version == 1998:
            return _get_colin1998_kwargs(download_root, extension)
        return _get_colin2008_kwargs(download_root, extension)


def _get_colin1998_kwargs(download_root, extension):
    t1, head, mask = (
        download_root / f"colin27_t1_tal_lin{suffix}{extension}"
        for suffix in ("", "_headmask", "_mask")
    )
    return {
        "t1": ScalarImage(t1),
        "head": LabelMap(head),
        "brain": LabelMap(mask),
    }


def _get_colin2008_kwargs(download_root, extension):
    t1, t2, pd, label = (
        download_root / f"colin27_{name}_tal_hires{extension}"
        for name in ("t1", "t2", "pd", "cls")
    )
    return {
        "t1": ScalarImage(t1),
        "t2": ScalarImage(t2),
        "pd": ScalarImage(pd),
        "cls": LabelMap(label),
    }
