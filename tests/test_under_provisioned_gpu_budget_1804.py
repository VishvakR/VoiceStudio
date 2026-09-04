"""#1804: a GPU below the engine's VRAM floor was budgeted as fast hardware.

Third report of one shape. #1226 (GTX 1650 Ti), #1222 (Quadro P2000) and now
#1804 (GTX 1650) are all 4 GB cards running the `omnivoice` engine, which
declares `min_vram_gb = 6.0` precisely because "below this the driver pages to
system RAM and a render that should take seconds runs for minutes".

The reporter's own breadcrumbs show the budget, not the hardware, ending the
job::

    11:53:02 generate:start → 11:59:14 stream-retryable-abort  = 372 s
    12:00:15 generate:start → 12:05:16 stream-retryable-abort  = 301 s

which is `generate_timeout_s()` exactly: 300 + max(0, len - 1200) / 40.

Everything downstream of the routing verdict already acted on it — the caveat
(`resolve_routing`), the preflight toast, the timeout message naming the card
(`test_low_vram_advisory.py`). The budget alone did not, so the slowest
supported configuration got **half** the 600 s a plain CPU host gets. These
tests pin the corrected floor, and the boundaries it must not cross.
"""
from __future__ import annotations

import importlib
import sys

import pytest

from core.device_caps import HostCaps
from services.engine_routing import _caveat, under_provisioned_vram
from services.tts_backend import OmniVoiceBackend


@pytest.fixture
def model_manager(monkeypatch):
    for mod_name in ("core.config", "services.model_manager"):
        if getattr(sys.modules.get(mod_name), "__file__", None) is None:
            sys.modules.pop(mod_name, None)
    import services.model_manager as mm
    return mm


def _gpu(vram_gb: float, *, family: str = "cuda",
         name: str = "NVIDIA GeForce GTX 1650") -> HostCaps:
    return HostCaps(
        family=family,
        available_families=(family, "cpu"),
        device_name=name,
        vram_gb=vram_gb,
    )


@pytest.fixture
def on_host(monkeypatch, model_manager):
    """Pin the host probe and the two budgets to their shipped defaults."""
    import core.device_caps as caps

    def _pin(host):
        monkeypatch.setattr(caps, "detect_host_caps", lambda: host)
        monkeypatch.setattr(model_manager, "GPU_JOB_TIMEOUT_S", 300.0)
        monkeypatch.setattr(model_manager, "CPU_JOB_TIMEOUT_S", 600.0)
        return model_manager

    return _pin


FLOOR = OmniVoiceBackend.min_vram_gb  # 6.0


# ── the reported bug ─────────────────────────────────────────────────────


def test_a_4gb_card_gets_the_cpu_budget_not_half_of_it(on_host):
    """The regression. A card that pages to system RAM performs like a CPU, so
    it must not be budgeted like a 24 GB one."""
    mm = on_host(_gpu(4.0))
    assert mm.generate_timeout_s(
        "A short render", execution_device="cuda", min_vram_gb=FLOOR,
    ) == 600.0


def test_the_engine_alone_is_enough_to_derive_the_floor(on_host):
    """Callers that pass `engine=` (tts_stream, batch, dub, openai_compat,
    archetypes) need no change — the floor is read off the engine."""
    mm = on_host(_gpu(4.0))
    assert mm.generate_timeout_s(
        "A short render", execution_device="cuda", engine=OmniVoiceBackend,
    ) == 600.0


def test_length_scaling_still_applies_on_top_of_the_raised_floor(on_host):
    """The reporter's longer take (4080 chars ⇒ 372 s before) keeps its bonus;
    the floor moved, the slope did not."""
    mm = on_host(_gpu(4.0))
    assert mm.generate_timeout_s(
        "x" * 4080, execution_device="cuda", min_vram_gb=FLOOR,
    ) == 600.0 + (4080 - 1200) / 40.0


def test_the_whole_class_not_just_cuda(on_host):
    """ROCm is the other dedicated-VRAM family; the same paging happens there."""
    mm = on_host(_gpu(4.0, family="rocm", name="AMD Radeon RX 6500 XT"))
    assert mm.generate_timeout_s(
        "A short render", execution_device="rocm", min_vram_gb=FLOOR,
    ) == 600.0


# ── the boundaries it must not cross ─────────────────────────────────────


def test_a_large_card_keeps_the_accelerated_budget(on_host):
    mm = on_host(_gpu(24.0, name="NVIDIA RTX 4090"))
    assert mm.generate_timeout_s(
        "A short render", execution_device="cuda", min_vram_gb=FLOOR,
    ) == 300.0


def test_an_engine_with_no_declared_floor_is_never_judged(on_host):
    """Only engines with a measured figure opt in; inventing a floor for the
    rest would silently double the watchdog for every other engine."""
    mm = on_host(_gpu(4.0))
    assert mm.generate_timeout_s("A short render", execution_device="cuda") == 300.0


def test_mps_is_not_judged_by_a_cuda_measured_floor(on_host):
    """`HostCaps.vram_gb` on MPS is system RAM / 2 for a UNIFIED pool — an 8 GB
    Mac reports 4.0 and runs this engine fine."""
    mm = on_host(_gpu(4.0, family="mps", name="Apple Silicon (MPS)"))
    assert mm.generate_timeout_s(
        "A short render", execution_device="mps", min_vram_gb=FLOOR,
    ) == 300.0


def test_a_failed_vram_probe_does_not_guess(on_host):
    """vram_gb == 0 means the probe failed, not that the card has no memory."""
    mm = on_host(_gpu(0.0))
    assert mm.generate_timeout_s(
        "A short render", execution_device="cuda", min_vram_gb=FLOOR,
    ) == 300.0


def test_a_cpu_fallback_render_is_unaffected(on_host):
    """Routing already sent this one to the CPU; it gets the CPU budget by the
    device branch, and the VRAM branch must not double-apply."""
    mm = on_host(_gpu(4.0))
    assert mm.generate_timeout_s(
        "A short render", execution_device="cpu", min_vram_gb=FLOOR,
    ) == 600.0


# ── an operator's explicit setting still wins ────────────────────────────


def test_an_explicit_universal_budget_is_honoured_verbatim(monkeypatch):
    """Same contract the CPU branch has kept since #1787: someone who lowered
    the watchdog to fail fast keeps failing fast, small card or not."""
    import core.device_caps as caps
    import services.model_manager as mm_mod

    monkeypatch.setenv("OMNIVOICE_GENERATE_TIMEOUT_S", "123.5")
    mm = importlib.reload(mm_mod)
    try:
        monkeypatch.setattr(caps, "detect_host_caps", lambda: _gpu(4.0))
        assert mm.generate_timeout_s(
            "A short render", execution_device="cuda", min_vram_gb=FLOOR,
        ) == 123.5
    finally:
        monkeypatch.delenv("OMNIVOICE_GENERATE_TIMEOUT_S", raising=False)
        importlib.reload(mm_mod)


def test_a_raised_accelerated_budget_is_never_cut_down_to_the_cpu_one(
    monkeypatch, model_manager,
):
    """The floor is a `max`, not an assignment."""
    import core.device_caps as caps

    monkeypatch.setattr(caps, "detect_host_caps", lambda: _gpu(4.0))
    monkeypatch.setattr(model_manager, "GPU_JOB_TIMEOUT_S", 900.0)
    monkeypatch.setattr(model_manager, "CPU_JOB_TIMEOUT_S", 600.0)
    assert model_manager.generate_timeout_s(
        "A short render", execution_device="cuda", min_vram_gb=FLOOR,
    ) == 900.0


# ── the verdict cannot drift from the warning built on it ────────────────


@pytest.mark.parametrize(
    "caps, expected",
    [
        (_gpu(4.0), True),
        (_gpu(24.0, name="NVIDIA RTX 4090"), False),
        (_gpu(0.0), False),
        (_gpu(4.0, family="mps", name="Apple Silicon (MPS)"), False),
        (_gpu(4.0, family="rocm", name="AMD Radeon RX 6500 XT"), True),
    ],
)
def test_the_budget_and_the_caveat_read_the_same_verdict(caps, expected):
    """One predicate, three consumers (caveat, timeout message, budget). Three
    inline copies is how the budget came to disagree with the warning printed
    next to it."""
    assert under_provisioned_vram(caps, FLOOR) is expected
    assert bool(_caveat(caps, FLOOR)) is expected


def test_an_undeclared_floor_and_a_missing_probe_are_both_silent():
    assert under_provisioned_vram(_gpu(4.0), 0.0) is False
    assert under_provisioned_vram(None, FLOOR) is False
