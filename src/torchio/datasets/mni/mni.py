"""Base class for MNI atlases."""

from __future__ import annotations

from ...data.subject import Subject
from ...download import get_torchio_cache_dir


class SubjectMNI(Subject):
    """Atlases from the Montreal Neurological Institute (MNI).

    See `the website <https://nist.mni.mcgill.ca/?page_id=714>`_ for more
    information.
    """

    url_base = "http://packages.bic.mni.mcgill.ca/mni-models/"
    name: str

    @property
    def download_root(self):
        """Return the download root directory for this atlas."""
        return get_torchio_cache_dir() / self.name
