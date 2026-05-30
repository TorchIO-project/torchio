import shutil
import urllib.parse

from ...data import ScalarImage
from ...download import download_and_extract_archive
from ...utils import compress
from .mni import SubjectMNI


class Sheep(SubjectMNI):
    """Population-averaged ovine brain MRI template from the MNI.

    A stereotaxic, population-averaged T1-weighted brain template constructed
    from 14 normal adult sheep (*Ovis orientalis aries*) at 0.5 mm isotropic
    resolution. Provides a common stereotaxic reference frame for localizing
    anatomical and functional information across individual sheep and studies.

    More information can be found on the
    [MNI Ovine Brain Atlas page](https://www.mcgill.ca/bic/neuroinformatics/brain-atlases-ovine-brain-atlas)`_.

    Keys: `t1`.

    Examples:

        >>> import torchio as tio
        >>> subject = tio.datasets.Sheep()
        >>> subject
        Sheep(Keys: ('t1',); images: 1)
    """

    def __init__(self):
        self.name = 'NIFTI_ovine_05mm'
        self.url_dir = urllib.parse.urljoin(self.url_base, 'sheep/')
        self.filename = f'{self.name}.zip'
        self.url = urllib.parse.urljoin(self.url_dir, self.filename)
        t1_nii_path = self.download_root / 'ovine_model_05.nii'
        t1_niigz_path = self.download_root / 'ovine_model_05.nii.gz'
        if not self.download_root.is_dir():
            download_and_extract_archive(
                self.url,
                download_root=self.download_root,
                filename=self.filename,
            )
            shutil.rmtree(self.download_root / 'masks')
            for path in self.download_root.iterdir():
                if path == t1_nii_path:
                    compress(t1_nii_path, t1_niigz_path)
                path.unlink()
        try:
            subject_dict = {'t1': ScalarImage(t1_niigz_path)}
        except FileNotFoundError:  # for backward compatibility
            subject_dict = {'t1': ScalarImage(t1_nii_path)}
        super().__init__(subject_dict)
