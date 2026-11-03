"""Public names for classes that already train. Nothing here is a new algorithm.

Top-level ``baselines/<name>.py`` would shadow the existing packages
(``baselines/inscl``, ``baselines/citb``, and the other method directories),
so the wrappers live in this package.
"""
from baselines.wrappers.citb_agem import CITBAGEMMethod
from baselines.wrappers.citb_ewc import CITBEWCMethod
from baselines.wrappers.citb_l2 import CITBL2Method
from baselines.wrappers.citb_replay import CITBReplayMethod
from baselines.wrappers.clora import CLoRAMethod
from baselines.wrappers.gainlora import GainLoRAMethod
from baselines.wrappers.gorp import GORPMethod
from baselines.wrappers.inscl import InsCLMethod
from baselines.wrappers.kpig import KPIGMethod
from baselines.wrappers.saber import SABERMethod
from baselines.wrappers.sapt import SAPTMethod
from baselines.wrappers.sdft import SDFTMethod
from baselines.wrappers.sequential import SequentialLoRAMethod
from baselines.wrappers.treelora import TreeLoRAMethod

REGISTRY = {
    "sequential_lora": SequentialLoRAMethod,
    "citb_replay": CITBReplayMethod,
    "citb_ewc": CITBEWCMethod,
    "citb_agem": CITBAGEMMethod,
    "citb_l2": CITBL2Method,
    "inscl": InsCLMethod,
    "kpig": KPIGMethod,
    "clora": CLoRAMethod,
    "gorp": GORPMethod,
    "sdft": SDFTMethod,
    "gainlora": GainLoRAMethod,
    "treelora": TreeLoRAMethod,
    "sapt": SAPTMethod,
    "saber": SABERMethod,
}

__all__ = [
    "REGISTRY",
    "SequentialLoRAMethod",
    "CITBReplayMethod",
    "CITBEWCMethod",
    "CITBAGEMMethod",
    "CITBL2Method",
    "InsCLMethod",
    "KPIGMethod",
    "CLoRAMethod",
    "GORPMethod",
    "SDFTMethod",
    "GainLoRAMethod",
    "TreeLoRAMethod",
    "SAPTMethod",
    "SABERMethod",
]
