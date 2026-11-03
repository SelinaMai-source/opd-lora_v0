"""The public wrappers must be the same classes the trainer already builds."""
from __future__ import annotations

import unittest

import baselines.wrappers as wrappers
from baselines.basic_baselines.sequential_lora.method import SequentialLoRAMethod
from baselines.citb.agem.method import CITBAGEMMethod
from baselines.citb.ewc.method import CITBEWCMethod
from baselines.citb.l2.method import CITBL2Method
from baselines.citb.replay.method import CITBReplayMethod
from baselines.clora.method import CLoRAMethod
from baselines.gainlora.method import GainLoRAMethod
from baselines.gorp.method import GORPMethod
from baselines.inscl.method import InsCLMethod
from baselines.kpig.method import KPIGMethod
from baselines.saber.method import SABERMethod
from baselines.sapt.method import SAPTMethod
from baselines.sdft.method import SDFTMethod
from baselines.treelora.method import TreeLoRAMethod
from benchmarks.instrdialog import get_metadata as id_meta
from benchmarks.instrdialogpp import get_metadata as pp_meta
from methods.legacy_compat_replay import legacy_counts
from methods.selectors import budget_for


REAL = {
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


class PublicWrapperTests(unittest.TestCase):
    def test_wrappers_are_the_existing_classes(self) -> None:
        self.assertEqual(set(wrappers.REGISTRY), set(REAL))
        for name, cls in REAL.items():
            self.assertIs(wrappers.REGISTRY[name], cls)

    def test_metadata_does_not_require_a_private_disk(self) -> None:
        for meta in (id_meta(), pp_meta()):
            self.assertNotIn("/root/autodl-tmp", str(meta))
            self.assertTrue(meta["filename"].endswith("_multiref.json"))

    def test_public_budget_helpers_keep_the_two_definitions(self) -> None:
        old = legacy_counts(45, strategy="uniform")
        proto = legacy_counts(45, strategy="proto")
        new = budget_for(45, 0.20)
        self.assertEqual(old["n_replay"], 9)
        self.assertEqual(proto["n_replay"], 8)
        self.assertEqual(new.n_replay_target, 11)
        self.assertEqual(new.n_related, 6)
        self.assertEqual(new.n_degraded, 5)


if __name__ == "__main__":
    unittest.main()
