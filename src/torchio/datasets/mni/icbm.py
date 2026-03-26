"""ICBM 2009c Nonlinear Symmetric template."""

from __future__ import annotations

import urllib.parse
from pathlib import Path

import torch

from ...data import LabelMap
from ...data import ScalarImage
from ...download import compress
from ...download import download_and_extract_archive
from ...download import get_torchio_cache_dir
from .mni import SubjectMNI


class ICBM2009CNonlinearSymmetric(SubjectMNI):
    r"""ICBM template.

    More information can be found in the
    `website <http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009>`_.

    Args:
        load_4d_tissues: If ``True``, the tissue probability maps will be
            loaded together into a 4D image. Otherwise, they will be loaded
            into independent images.
    """

    def __init__(self, load_4d_tissues: bool = False) -> None:
        self.name = "mni_icbm152_nlin_sym_09c_nifti"
        self.url_base = "http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/"
        self.filename = f"{self.name}.zip"
        self.url = urllib.parse.urljoin(self.url_base, self.filename)
        download_root = get_torchio_cache_dir() / self.name
        if not download_root.is_dir():
            download_and_extract_archive(
                self.url,
                download_root=download_root,
                filename=self.filename,
                remove_finished=True,
            )

        files_dir = download_root / "mni_icbm152_nlin_sym_09c"

        p = str(files_dir / "mni_icbm152")
        m = "tal_nlin_sym_09c"
        s = ".nii.gz"

        tissues_path = f"{p}_tissues_{m}.nii.gz"
        if not Path(tissues_path).is_file():
            gm = LabelMap(f"{p}_gm_{m}.nii")
            wm = LabelMap(f"{p}_wm_{m}.nii")
            csf = LabelMap(f"{p}_csf_{m}.nii")
            gm.load()
            wm.load()
            csf.load()
            gm.set_data(torch.cat((gm.data, wm.data, csf.data)))
            gm.save(tissues_path)

        for fp in files_dir.glob("*.nii"):
            compress(fp, fp.with_suffix(".nii.gz"))
            fp.unlink()

        subject_kwargs: dict[str, ScalarImage | LabelMap] = {
            "t1": ScalarImage(f"{p}_t1_{m}{s}"),
            "eyes": LabelMap(f"{p}_t1_{m}_eye_mask{s}"),
            "face": LabelMap(f"{p}_t1_{m}_face_mask{s}"),
            "brain": LabelMap(f"{p}_t1_{m}_mask{s}"),
            "t2": ScalarImage(f"{p}_t2_{m}{s}"),
            "pd": ScalarImage(f"{p}_csf_{m}{s}"),
        }
        if load_4d_tissues:
            subject_kwargs["tissues"] = LabelMap(
                tissues_path,
                channels_last=True,
            )
        else:
            subject_kwargs["gm"] = LabelMap(f"{p}_gm_{m}{s}")
            subject_kwargs["wm"] = LabelMap(f"{p}_wm_{m}{s}")
            subject_kwargs["csf"] = LabelMap(f"{p}_csf_{m}{s}")

        super().__init__(**subject_kwargs)
