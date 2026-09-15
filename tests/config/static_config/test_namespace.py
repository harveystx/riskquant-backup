# tests/config/static_config/test_namespace.py
"""
Regression tests for `riskquant.config.namespace.enforce_id_namespace`
(RQ-AUD-022 remediation).

This is the foundational ID-space safety check every namespace-bounded id
(node/curve/surface/cube) is validated against; previously it had no
dedicated unit test (only indirect coverage via a passing end-to-end
compile).
"""
from __future__ import annotations

import numpy as np
import pytest

from riskquant.config.namespace import GlobalIdNamespace, enforce_id_namespace
from riskquant.exceptions import ConfigurationError


def test_ids_within_bounds_do_not_raise():
    ids = np.array(
        [
            GlobalIdNamespace.NODES.min_id,
            GlobalIdNamespace.NODES.min_id + 1,
            GlobalIdNamespace.NODES.max_id,
        ],
        dtype=np.int32,
    )
    enforce_id_namespace(ids=ids, bounds=GlobalIdNamespace.NODES)  # must not raise


def test_id_below_min_raises():
    ids = np.array([GlobalIdNamespace.NODES.min_id - 1], dtype=np.int64)
    with pytest.raises(ConfigurationError, match="ID Namespace Violation"):
        enforce_id_namespace(ids=ids, bounds=GlobalIdNamespace.NODES)


def test_id_above_max_raises():
    ids = np.array([GlobalIdNamespace.NODES.max_id + 1], dtype=np.int64)
    with pytest.raises(ConfigurationError, match="ID Namespace Violation"):
        enforce_id_namespace(ids=ids, bounds=GlobalIdNamespace.NODES)


def test_id_from_a_different_namespace_raises():
    # A perfectly valid CURVES id must be rejected against the NODES bounds.
    ids = np.array([GlobalIdNamespace.CURVES.min_id], dtype=np.int64)
    with pytest.raises(ConfigurationError, match="ID Namespace Violation"):
        enforce_id_namespace(ids=ids, bounds=GlobalIdNamespace.NODES)


def test_empty_id_array_does_not_raise():
    ids = np.array([], dtype=np.int64)
    enforce_id_namespace(ids=ids, bounds=GlobalIdNamespace.NODES)  # must not raise


def test_error_message_names_the_namespace_and_examples():
    ids = np.array([GlobalIdNamespace.NODES.max_id + 100], dtype=np.int64)
    with pytest.raises(ConfigurationError) as exc:
        enforce_id_namespace(ids=ids, bounds=GlobalIdNamespace.NODES)
    message = str(exc.value)
    assert GlobalIdNamespace.NODES.name in message
    assert str(GlobalIdNamespace.NODES.max_id + 100) in message


@pytest.mark.parametrize(
    "namespace",
    [
        GlobalIdNamespace.NODES,
        GlobalIdNamespace.CURVES,
        GlobalIdNamespace.SURFACES,
        GlobalIdNamespace.CUBES,
    ],
)
def test_every_namespace_block_is_internally_consistent(namespace):
    """Every declared namespace block must have min_id <= max_id and fit
    within int32 (the declared dtype for rf_id/container ids throughout the
    numerical core)."""
    assert namespace.min_id <= namespace.max_id
    assert namespace.max_id <= np.iinfo(np.int32).max
    assert namespace.min_id >= 0
