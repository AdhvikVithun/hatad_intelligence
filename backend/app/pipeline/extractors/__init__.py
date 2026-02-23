"""Document extractors for all 16 supported document types."""

from .base import BaseExtractor
from .ec import ECExtractor
from .sale_deed import SaleDeedExtractor
from .patta import PattaExtractor
from .generic import GenericExtractor
from .fmb import FMBExtractor
from .adangal import AdangalExtractor
from .layout_approval import LayoutApprovalExtractor
from .legal_heir import LegalHeirExtractor
from .poa import POAExtractor
from .court_order import CourtOrderExtractor
from .will_extractor import WillExtractor
from .partition_deed import PartitionDeedExtractor
from .gift_deed import GiftDeedExtractor
from .release_deed import ReleaseDeedExtractor
from .a_register import ARegisterExtractor
from .chitta import ChittaExtractor

__all__ = [
    "BaseExtractor",
    "ECExtractor",
    "SaleDeedExtractor",
    "PattaExtractor",
    "GenericExtractor",
    "FMBExtractor",
    "AdangalExtractor",
    "LayoutApprovalExtractor",
    "LegalHeirExtractor",
    "POAExtractor",
    "CourtOrderExtractor",
    "WillExtractor",
    "PartitionDeedExtractor",
    "GiftDeedExtractor",
    "ReleaseDeedExtractor",
    "ARegisterExtractor",
    "ChittaExtractor",
]
