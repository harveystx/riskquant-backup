# src/riskquant/credit/jtd/ingest/validate.py
"""
Structural validation of the JTD ingest bundle.

This is the contract-driven layer: it enforces the shape of each per-model
master and of the shared multi-factor loadings relation BEFORE any block is
built — required columns present, no nulls in required columns, dtype families
correct, obligor key unique within a frame and a subset of the obligor master,
and (critically) an obligor appears in at most one per-model frame. Semantic /
wiring validation (factor kinds, distinct uniforms, sum-beta^2 <= 1, chi-nu
homogeneity) is model-specific and lands separately; this layer answers only
"is the master shaped the way its contract demands".

Every check fails fast with a specific, actionable message. After this passes,
the builders may treat their input as structurally valid.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

from riskquant.credit.jtd.ingest.bundle import JtdIngestBundle
from riskquant.credit.jtd.ingest.contracts import (
    JTD_MODEL_CONTRACTS,
    MF_LOADINGS_CONTRACT,
    OBLIGOR_KEY,
    JtdModelContract,
    MfLoadingsContract,
)
from riskquant.credit.masters.enums.credit_factor_kind import CreditFactorKind

if TYPE_CHECKING:
    from riskquant.credit.masters.obligors.compiled import CompiledObligorCredit

_PREVIEW = 10


# -------------------------------------------------------------------------
# Per-frame structural predicates
# -------------------------------------------------------------------------


def validate_required_columns(
    df: pd.DataFrame, contract: JtdModelContract, *, ctx: str
) -> None:
    missing = sorted(contract.required_columns - set(df.columns))
    if missing:
        raise ValueError(
            f"{ctx}: master for {contract.model.name} is missing required "
            f"column(s) {missing}. Required: {sorted(contract.required_columns)}."
        )


def validate_no_required_nulls(
    df: pd.DataFrame, contract: JtdModelContract, *, ctx: str
) -> None:
    for col in sorted(contract.required_columns):
        if df[col].isna().any():
            n = int(df[col].isna().sum())
            raise ValueError(
                f"{ctx}: {contract.model.name} column '{col}' has {n} null "
                "value(s); per-model masters are dense by contract."
            )


def validate_dtypes(df: pd.DataFrame, contract: JtdModelContract, *, ctx: str) -> None:
    if not pd.api.types.is_integer_dtype(df[OBLIGOR_KEY]):
        raise ValueError(
            f"{ctx}: {contract.model.name} '{OBLIGOR_KEY}' must be an integer "
            f"dtype, got {df[OBLIGOR_KEY].dtype}."
        )
    for col in sorted(contract.float_columns):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(
                f"{ctx}: {contract.model.name} '{col}' must be numeric, got "
                f"{df[col].dtype}."
            )
    for col in sorted(contract.factor_name_columns):
        if not (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
        ):
            raise ValueError(
                f"{ctx}: {contract.model.name} factor-name column '{col}' must be "
                f"string-like, got {df[col].dtype}."
            )


def validate_uid_unique_within_frame(
    df: pd.DataFrame, contract: JtdModelContract, *, ctx: str
) -> None:
    uids = df[OBLIGOR_KEY].to_numpy()
    vals, counts = np.unique(uids, return_counts=True)
    dups = vals[counts > 1][:_PREVIEW].tolist()
    if dups:
        raise ValueError(
            f"{ctx}: {contract.model.name} has duplicate obligor uid(s) {dups}; "
            "each obligor must appear at most once per model frame."
        )


def validate_uids_in_master(
    df: pd.DataFrame,
    obligor_master: CompiledObligorCredit,
    contract: JtdModelContract,
    *,
    ctx: str,
) -> None:
    uids = df[OBLIGOR_KEY].to_numpy()
    known = np.isin(uids, obligor_master.uid)
    if not known.all():
        unknown = np.unique(uids[~known])[:_PREVIEW].tolist()
        raise ValueError(
            f"{ctx}: {contract.model.name} references obligor uid(s) {unknown} "
            "absent from the obligor master (the single source of truth for "
            "PD/LGD)."
        )


# -------------------------------------------------------------------------
# Cross-frame structural predicates
# -------------------------------------------------------------------------


def validate_uid_disjoint_across_bundle(bundle: JtdIngestBundle, *, ctx: str) -> None:
    """An obligor carries exactly one JTD model, so its uid may appear in at most
    one per-model frame. The wide union frame enforced this implicitly (one row
    per uid); per-model frames must enforce it explicitly."""
    seen: dict[int, str] = {}
    for model, frame in bundle.frames_by_model().items():
        for uid in frame[OBLIGOR_KEY].tolist():
            uid_i = int(uid)
            if uid_i in seen:
                raise ValueError(
                    f"{ctx}: obligor uid {uid_i} appears in both "
                    f"{seen[uid_i]} and {model.name}. Each obligor must be routed "
                    "to exactly one JTD model."
                )
            seen[uid_i] = model.name


# -------------------------------------------------------------------------
# Multi-factor loadings relation
# -------------------------------------------------------------------------


def validate_mf_loadings_structural(
    mf_loadings_df: pd.DataFrame, contract: MfLoadingsContract, *, ctx: str
) -> None:
    missing = sorted(contract.required_columns - set(mf_loadings_df.columns))
    if missing:
        raise ValueError(f"{ctx}: mf_loadings missing required column(s) {missing}.")

    key = contract.obligor_key
    if mf_loadings_df[key].isna().any():
        raise ValueError(f"{ctx}: mf_loadings '{key}' has null value(s).")
    if not pd.api.types.is_integer_dtype(mf_loadings_df[key]):
        raise ValueError(
            f"{ctx}: mf_loadings '{key}' must be an integer dtype, got "
            f"{mf_loadings_df[key].dtype}."
        )

    name_col = contract.factor_name_column
    if mf_loadings_df[name_col].isna().any():
        raise ValueError(f"{ctx}: mf_loadings '{name_col}' has null value(s).")
    if not (
        pd.api.types.is_object_dtype(mf_loadings_df[name_col])
        or pd.api.types.is_string_dtype(mf_loadings_df[name_col])
    ):
        raise ValueError(
            f"{ctx}: mf_loadings '{name_col}' must be string-like, got "
            f"{mf_loadings_df[name_col].dtype}."
        )

    load_col = contract.loading_column
    if not pd.api.types.is_numeric_dtype(mf_loadings_df[load_col]):
        raise ValueError(
            f"{ctx}: mf_loadings '{load_col}' must be numeric, got "
            f"{mf_loadings_df[load_col].dtype}."
        )
    loadings = mf_loadings_df[load_col].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(loadings)):
        raise ValueError(f"{ctx}: mf_loadings '{load_col}' contains NaN/inf.")


def validate_mf_loadings_routable(
    bundle: JtdIngestBundle, contract: MfLoadingsContract, *, ctx: str
) -> None:
    """Every loading row must attach to an obligor that actually appears in a
    multi-factor model frame. Guards the silent-drop class (a typo'd or
    mis-routed obligor_uid would otherwise vanish in the per-block filter)."""
    if bundle.mf_loadings_df is None:
        return

    mf_uids: set[int] = set()
    for model, frame in bundle.frames_by_model().items():
        if JTD_MODEL_CONTRACTS[model].uses_mf_loadings:
            mf_uids.update(int(u) for u in frame[OBLIGOR_KEY].tolist())

    if not mf_uids:
        raise ValueError(
            f"{ctx}: mf_loadings supplied but the bundle has no multi-factor "
            "obligors; the loadings are unroutable."
        )

    key = contract.obligor_key
    stray = sorted({int(u) for u in bundle.mf_loadings_df[key].tolist()} - mf_uids)
    if stray:
        raise ValueError(
            f"{ctx}: mf_loadings references obligor_uid(s) {stray[:_PREVIEW]} that "
            "are not multi-factor obligors in any bundle frame (silent-drop "
            "guard)."
        )


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------


def validate_ingest_bundle(
    bundle: JtdIngestBundle,
    obligor_master: CompiledObligorCredit,
) -> None:
    """Run the full structural validation pass over a JTD ingest bundle.

    Fails fast on the first violated invariant. After this returns, every
    populated frame satisfies its column contract, uids are unique-per-frame,
    master-resident, and disjoint across frames, and the loadings relation is
    well-shaped and routable.
    """
    if bundle.is_empty():
        raise ValueError(
            "JtdIngestBundle is empty: no per-model master frame is populated."
        )

    for model, frame in bundle.frames_by_model().items():
        contract = JTD_MODEL_CONTRACTS[model]
        ctx = f"jtd_ingest[{model.name}]"
        validate_required_columns(frame, contract, ctx=ctx)
        validate_no_required_nulls(frame, contract, ctx=ctx)
        validate_dtypes(frame, contract, ctx=ctx)
        validate_uid_unique_within_frame(frame, contract, ctx=ctx)
        validate_uids_in_master(frame, obligor_master, contract, ctx=ctx)

    validate_uid_disjoint_across_bundle(bundle, ctx="jtd_ingest")

    # Multi-factor loadings: required iff any MF model frame is present.
    mf_present = any(
        JTD_MODEL_CONTRACTS[m].uses_mf_loadings for m in bundle.frames_by_model()
    )
    if mf_present and bundle.mf_loadings_df is None:
        raise ValueError(
            "jtd_ingest: a multi-factor model frame is present but mf_loadings "
            "is missing."
        )
    if bundle.mf_loadings_df is not None:
        validate_mf_loadings_structural(
            bundle.mf_loadings_df, MF_LOADINGS_CONTRACT, ctx="jtd_ingest[mf_loadings]"
        )
        validate_mf_loadings_routable(
            bundle, MF_LOADINGS_CONTRACT, ctx="jtd_ingest[mf_loadings]"
        )


# -------------------------------------------------------------------------
# Semantic / wiring validators (model-specific correctness)
# -------------------------------------------------------------------------
# Beyond shape: factor-role kinds, conditional independence of uniforms, loading
# ranges, and the Student-t chi/nu reconciliation. They operate on RESOLVED uix
# (the builders resolve names -> uix via _shared._resolve_factor_uix, then call
# these). chi reconciliation additionally reads the universe's chi_nu_by_uix so
# the obligor frame cannot disagree with the generator about degrees of freedom.


def assert_factor_kind(
    uix: npt.NDArray[np.int32],
    expected: CreditFactorKind,
    factor_kind: npt.NDArray[np.int8],
    *,
    ctx: str,
) -> None:
    """Every resolved UIX must be a factor of the kind its role requires.

    Catches mis-wiring the loose name→UIX dict cannot: e.g. a beta-binomial
    obligor pointing ``beta_factor`` at a UNIFORM column, or a Bernoulli obligor
    pointing ``uniform_factor`` at a systemic normal. The kind comes from the
    universe master (``factor_kind`` is UIX-indexed)."""
    actual = factor_kind[uix]
    bad = actual != np.int8(expected)
    if bad.any():
        offending = np.unique(uix[bad])[:_PREVIEW].tolist()
        got = {int(k): CreditFactorKind(int(k)).name for k in np.unique(actual[bad])}
        raise ValueError(
            f"{ctx}: factor(s) at UIX {offending} must be kind "
            f"{expected.name}, but resolve to kind(s) {got}."
        )


def assert_uniforms_distinct(
    uix: npt.NDArray[np.int32],
    *,
    ctx: str,
    role: str = "uniform",
) -> None:
    """Each obligor needs its OWN private row for `role`.

    For the Bernoulli family that row is the uniform: sharing it makes the
    conditionally-independent default draws comonotonic (rank-identical).

    For the copula family it is the idiosyncratic normal: the latent correlation
    contract `Corr(A_i, A_j) = sum_k beta_ik beta_jk` assumes `eps_i` is private,
    and a shared epsilon silently adds `w_i * w_j` to every affected pair (C-07).

    `role` only names the offender in the message; the invariant is identical.
    """
    vals, counts = np.unique(uix, return_counts=True)
    shared = vals[counts > 1][:_PREVIEW].tolist()
    if shared:
        raise ValueError(
            f"{ctx}: {role} factor UIX {shared} are shared across obligors. "
            f"Each obligor requires a distinct {role} factor; sharing one "
            "introduces dependence the model's correlation contract does not "
            "account for."
        )


def validate_factor_loading_range(
    obligor_uids: npt.NDArray[np.int64], factor_loading: npt.NDArray[np.float64]
) -> None:
    bad = ~((factor_loading >= -1.0) & (factor_loading <= 1.0))
    if bad.any():
        raise ValueError(
            f"compile_jtd_portfolio: factor_loading outside [-1, 1] for obligor(s) "
            f"{obligor_uids[bad][:_PREVIEW].tolist()}."
        )


def validate_nu_positive(
    obligor_uids: npt.NDArray[np.int64], nu: npt.NDArray[np.float64]
) -> None:
    bad = ~(nu > 0.0)  # also catches NaN
    if bad.any():
        raise ValueError(
            f"compile_jtd_portfolio: nu must be > 0 (finite) for obligor(s) "
            f"{obligor_uids[bad][:_PREVIEW].tolist()}."
        )


def validate_chi_group_nu_homogeneity(
    obligor_uids: npt.NDArray[np.int64],
    chi_squared_uix: npt.NDArray[np.int32],
    nu: npt.NDArray[np.float64],
    factor_name_to_uix: dict[str, int],
) -> None:
    """Within a chi group, nu must be constant.

    Obligors sharing a ``chi_squared_uix`` share one mixing draw
    ``W ~ chi^2(nu)`` per scenario — the grouped t-copula construction (Daul,
    De Giorgi, Lindskog & McNeil, 2003). A single W cannot simultaneously be
    chi-square of two different dof, so every obligor on a given chi column
    must declare the same nu. The kernel's ``sqrt(nu_i / W)`` transform assumes
    this silently: a mismatch produces a scaled-t with the wrong dof and a
    miscalibrated marginal PD, with no runtime error. Fail fast here.

    The check is the group invariant only; it does not (cannot, from the frame
    alone) verify that the shared nu equals the dof the chi column was actually
    drawn with in the generator — that cross-layer guarantee comes from
    deriving the generator's ``nu_by_chi`` from these same blocks.
    """
    if chi_squared_uix.shape[0] == 0:
        return

    # Group by chi column via a stable sort; group nu-range == 0 iff constant.
    order = np.argsort(chi_squared_uix, kind="stable")
    cu = chi_squared_uix[order]
    nv = nu[order]

    starts = np.concatenate(([0], np.flatnonzero(np.diff(cu)) + 1))
    bad_groups = np.flatnonzero(
        np.minimum.reduceat(nv, starts) != np.maximum.reduceat(nv, starts)
    )
    if bad_groups.size == 0:
        return

    uix_to_name = {uix: name for name, uix in factor_name_to_uix.items()}
    uu = obligor_uids[order]
    ends = np.concatenate((starts[1:], [cu.shape[0]]))

    details = []
    for g in bad_groups[:_PREVIEW]:
        s, e = int(starts[g]), int(ends[g])
        col = int(cu[s])
        name = uix_to_name.get(col, f"<uix {col}>")
        nus = np.unique(nv[s:e]).tolist()
        uids = uu[s:e][:_PREVIEW].tolist()
        details.append(f"{name!r}: nu={nus} across obligor(s) {uids}")

    raise ValueError(
        "COPULA_SF_T: obligors sharing a chi_squared_factor must declare the "
        "same nu (one W ~ chi^2(nu) per scenario cannot match multiple dof). "
        "Offending chi group(s): " + "; ".join(details) + "."
    )


def validate_chi_nu_against_reference(
    obligor_uids: npt.NDArray[np.int64],
    chi_squared_uix: npt.NDArray[np.int32],
    nu: npt.NDArray[np.float64],
    chi_nu_by_uix: Mapping[int, float],
    factor_name_to_uix: dict[str, int],
) -> None:
    """Reconcile obligor nu against the dof its chi column is actually drawn with.

    The kernel forms ``T_i = sqrt(nu_i / W) * A_i``; the marginal is Student-t
    with ``nu_i`` dof ONLY if the chi column obligor i references is drawn as
    ``W ~ chi^2(nu_i)``. ``nu_i`` lives on the obligor frame; the draw dof lives
    on the generator (``nu_by_chi``). They are independent inputs and nothing
    else reconciles them, so two silent failure modes exist:

    - **mismatch** — obligor declares ``nu_i`` but the generator draws that chi
      column at a different dof ``m``: ``sqrt(nu_i / W)`` with ``W ~ chi^2(m)``
      is a scaled-t with ``m`` dof, and ``F_t^{-1}(PD; nu_i)`` no longer gives
      ``P(T_i < t_i) = PD_i``.
    - **absent** — obligor references a column the generator does not draw as
      chi at all: that column carries a standard normal, not ``1/sqrt(W)``, so
      the latent variable is nonsense.

    Both produce plausible-looking losses with no runtime error. Fail fast.

    ``chi_nu_by_uix`` maps chi factor UIX -> dof and must come from the SAME
    generator that produces the factor matrix (``generator.chi_nu_by_uix()``),
    never a parallel registry that could itself have drifted.
    """
    if chi_squared_uix.shape[0] == 0:
        return

    # Vectorized membership + lookup against the (small) chi map via sorted
    # searchsorted — same primitive shape as the package's uid joins, no
    # per-obligor dict loop.
    keys = np.fromiter(chi_nu_by_uix.keys(), dtype=np.int64, count=len(chi_nu_by_uix))
    vals = np.fromiter(
        chi_nu_by_uix.values(), dtype=np.float64, count=len(chi_nu_by_uix)
    )
    sorter = np.argsort(keys, kind="stable")
    skeys = keys[sorter]
    svals = vals[sorter]

    chi64 = chi_squared_uix.astype(np.int64)
    n_keys = skeys.shape[0]
    if n_keys == 0:
        # No chi columns declared by the generator at all: every t obligor
        # references a column that will be drawn as a standard normal.
        present = np.zeros(chi64.shape[0], dtype=np.bool_)
        ref_nu = np.full(chi64.shape[0], np.nan, dtype=np.float64)
    else:
        pos = np.searchsorted(skeys, chi64)
        pos_clipped = np.clip(pos, 0, n_keys - 1)
        present = (pos < n_keys) & (skeys[pos_clipped] == chi64)
        ref_nu = np.where(present, svals[pos_clipped], np.nan)

    uix_to_name = {uix: name for name, uix in factor_name_to_uix.items()}

    absent = ~present
    if absent.any():
        bad_cols = sorted({int(c) for c in chi_squared_uix[absent].tolist()})
        names = [uix_to_name.get(c, f"<uix {c}>") for c in bad_cols]
        uids = obligor_uids[absent][:_PREVIEW].tolist()
        raise ValueError(
            "COPULA_SF_T: chi_squared_factor(s) "
            f"{names} are referenced by obligor(s) {uids} but the generator "
            "does not draw them as chi columns (chi_nu_by_uix has no entry). "
            "The generator would emit a standard normal there, not 1/sqrt(W)."
        )

    mismatch = ref_nu != nu  # present everywhere here; NaN already excluded
    if mismatch.any():
        idx = np.flatnonzero(mismatch)[:_PREVIEW]
        details = [
            f"uid {int(obligor_uids[i])} "
            f"({uix_to_name.get(int(chi_squared_uix[i]), '<uix>')}): "
            f"declared nu={float(nu[i])}, generator draws nu={float(ref_nu[i])}"
            for i in idx
        ]
        raise ValueError(
            "COPULA_SF_T: obligor nu disagrees with the dof its chi column is "
            "drawn with (the t-marginal would be miscalibrated). "
            + "; ".join(details)
            + "."
        )


# -------------------------------------------------------------------------
# Semantic pre-build pass (the orchestration over the predicates above)
# -------------------------------------------------------------------------
# Runs AFTER structural validation and BEFORE any builder. Resolves each frame's
# factor-name columns to UIX and asserts wiring correctness, so the per-model
# builders can construct from input they may treat as fully valid. Resolution
# here is ingest-time and vectorized; the builders re-resolve for construction
# (cheap, and keeps validation and construction decoupled).

from riskquant.credit.jtd.ingest._shared import _resolve_factor_uix  # noqa: E402
from riskquant.credit.jtd.ingest.contracts import FACTOR_COLUMN_KIND  # noqa: E402
from riskquant.credit.jtd.registry import JTD_MODEL_SPECS  # noqa: E402


def _resolve(frame: pd.DataFrame, col: str, factor_name_to_uix, *, ctx: str):
    return _resolve_factor_uix(
        frame[col].to_numpy(copy=False), factor_name_to_uix, ctx=f"{ctx}.{col}"
    )


def _assert_column_kinds(frame, contract, factor_name_to_uix, factor_kind, *, ctx):
    """Every factor-name column must resolve to the kind its role requires."""
    for col in sorted(contract.factor_name_columns):
        uix = _resolve(frame, col, factor_name_to_uix, ctx=ctx)
        assert_factor_kind(
            uix, FACTOR_COLUMN_KIND[col], factor_kind, ctx=f"{ctx}.{col}"
        )


def _assert_distinct_columns(frame, contract, factor_name_to_uix, *, ctx):
    for col in sorted(contract.distinct_columns):
        uix = _resolve(frame, col, factor_name_to_uix, ctx=ctx)
        role = col.removesuffix("_factor")
        assert_uniforms_distinct(uix, ctx=f"{ctx}.{col}", role=role)


def _semantic_sf_gaussian(
    frame,
    *,
    factor_name_to_uix,
    factor_kind,
    chi_nu_by_uix,
    bundle,
    ctx,
):
    validate_factor_loading_range(
        frame["uid"].to_numpy(dtype=np.int64),
        frame["factor_loading"].to_numpy(dtype=np.float64),
    )


def _semantic_sf_students_t(
    frame, *, factor_name_to_uix, factor_kind, chi_nu_by_uix, bundle, ctx
):
    uids = frame["uid"].to_numpy(dtype=np.int64)
    validate_factor_loading_range(
        uids, frame["factor_loading"].to_numpy(dtype=np.float64)
    )
    nu = frame["nu"].to_numpy(dtype=np.float64)
    validate_nu_positive(uids, nu)
    chi_uix = _resolve(frame, "chi_squared_factor", factor_name_to_uix, ctx=ctx)
    validate_chi_group_nu_homogeneity(uids, chi_uix, nu, factor_name_to_uix)
    validate_chi_nu_against_reference(
        uids, chi_uix, nu, chi_nu_by_uix, factor_name_to_uix
    )


def _semantic_mf_gaussian(
    frame,
    *,
    factor_name_to_uix,
    factor_kind,
    chi_nu_by_uix,
    bundle,
    ctx,
):
    """Systemic-kind, coverage, no-duplicate, and sum-beta^2 <= 1 — all checked
    on the long-format loadings relation (the builder then packs CSR)."""
    loadings = bundle.mf_loadings_df  # structural validation guarantees presence
    sub = loadings[loadings["obligor_uid"].isin(set(frame["uid"].tolist()))]

    sys_uix = _resolve_factor_uix(
        sub["factor_name"].to_numpy(copy=False),
        factor_name_to_uix,
        ctx=f"{ctx}.systemic_factor",
    )
    assert_factor_kind(
        sys_uix,
        FACTOR_COLUMN_KIND["systemic_factor"],
        factor_kind,
        ctx=f"{ctx}.systemic_factor",
    )

    # Coverage: every MF obligor needs >= 1 loading row.
    have = set(int(u) for u in sub["obligor_uid"].tolist())
    missing = sorted(set(int(u) for u in frame["uid"].tolist()) - have)
    if missing:
        raise ValueError(
            f"{ctx}: obligor(s) {missing[:_PREVIEW]} have no systemic loadings."
        )

    # No factor referenced twice for one obligor.
    if sub.duplicated(subset=["obligor_uid", "factor_name"]).any():
        dups = (
            sub
            .loc[sub.duplicated(subset=["obligor_uid", "factor_name"]), "obligor_uid"]
            .unique()
            .tolist()[:_PREVIEW]
        )
        raise ValueError(
            f"{ctx}: obligor(s) {dups} load on the same systemic factor twice."
        )

    # sum beta^2 <= 1 (room for a real idiosyncratic component).
    sum_sq = sub["factor_loading"].to_numpy(dtype=np.float64) ** 2
    by = pd.Series(sum_sq).groupby(sub["obligor_uid"].to_numpy()).sum()
    over = by[by > 1.0 + 1e-9].index.tolist()[:_PREVIEW]
    if over:
        raise ValueError(
            f"{ctx}: sum of squared loadings exceeds 1 for obligor(s) {over}."
        )


def _semantic_mf_students_t(
    frame,
    *,
    factor_name_to_uix,
    factor_kind,
    chi_nu_by_uix,
    bundle,
    ctx,
):
    """MF systemic loadings checks (coverage / no-dup / sum-beta^2 / systemic
    kind) plus the Student-t chi/nu reconciliation. The systemic side lives on
    the shared loadings relation; nu and the chi factor live on this frame."""
    _semantic_mf_gaussian(
        frame,
        factor_name_to_uix=factor_name_to_uix,
        factor_kind=factor_kind,
        chi_nu_by_uix=chi_nu_by_uix,
        bundle=bundle,
        ctx=ctx,
    )
    uids = frame["uid"].to_numpy(dtype=np.int64)
    nu = frame["nu"].to_numpy(dtype=np.float64)
    validate_nu_positive(uids, nu)
    chi_uix = _resolve(frame, "chi_squared_factor", factor_name_to_uix, ctx=ctx)
    validate_chi_group_nu_homogeneity(uids, chi_uix, nu, factor_name_to_uix)
    validate_chi_nu_against_reference(
        uids, chi_uix, nu, chi_nu_by_uix, factor_name_to_uix
    )


def _semantic_marshall_olkin(
    frame,
    *,
    factor_name_to_uix,
    factor_kind,
    chi_nu_by_uix,
    bundle,
    ctx,
):
    """
    Semantic validation for the Marshall-Olkin Common Shock model.
    """
    sys_pd = frame["systemic_pd"].to_numpy(dtype=np.float64)

    # systemic_pd must be bounded [0.0, 1.0).
    # It must be strictly less than 1.0 to prevent division by zero during the
    # idiosyncratic decoupling step in the builder: (1 - pd_total) / (1 - systemic_pd)
    bad = ~((sys_pd >= 0.0) & (sys_pd < 1.0))
    if bad.any():
        uids = frame["uid"].to_numpy(dtype=np.int64)
        raise ValueError(
            f"{ctx}: 'systemic_pd' must be in [0.0, 1.0) for obligor(s) "
            f"{uids[bad][:_PREVIEW].tolist()} to prevent infinite survival probabilities."
        )


# Per-model semantic-validator dispatch, DERIVED from the model registry: each
# spec names its validator (`semantic_validator_attr`); the function itself lives
# here, co-located with the predicates it calls. Adding a model means declaring
# the name in its spec — no edit to a parallel dict here. A declared-but-missing
# name fails fast at import.
def _resolve_semantic_extra() -> dict:
    table: dict = {}
    for spec in JTD_MODEL_SPECS:
        attr = spec.semantic_validator_attr
        if attr is None:
            continue
        fn = globals().get(attr)
        if fn is None:
            raise RuntimeError(
                f"{spec.model.name}: semantic_validator_attr '{attr}' is not "
                f"defined in validate.py."
            )
        table[spec.model] = fn
    return table


_SEMANTIC_EXTRA = _resolve_semantic_extra()


def validate_ingest_bundle_semantics(bundle, factor_universe) -> None:
    """Semantic / wiring validation over a structurally-valid bundle.

    Resolves factor-name columns to UIX and asserts: every role resolves to its
    required kind, Bernoulli-family uniforms are distinct, loadings lie in
    range, Student-t nu is positive and chi-consistent with the generator, and
    multi-factor loadings are covered, unique,
    and satisfy sum-beta^2 <= 1. Fails fast on the first violation.
    """
    fmap = factor_universe.factor_name_to_uix
    fkind = factor_universe.factor_kind
    chi_nu = factor_universe.chi_nu_by_uix

    for model, frame in bundle.frames_by_model().items():
        contract = JTD_MODEL_CONTRACTS[model]
        ctx = f"jtd_semantic[{model.name}]"
        _assert_column_kinds(frame, contract, fmap, fkind, ctx=ctx)
        _assert_distinct_columns(frame, contract, fmap, ctx=ctx)
        extra = _SEMANTIC_EXTRA.get(model)
        if extra is not None:
            extra(
                frame,
                factor_name_to_uix=fmap,
                factor_kind=fkind,
                chi_nu_by_uix=chi_nu,
                bundle=bundle,
                ctx=ctx,
            )
