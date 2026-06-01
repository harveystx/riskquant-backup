# src/riskquant/config/enums/container_topology.py


from enum import IntEnum


class ContainerTopologyType(IntEnum):
    NODE = 0
    CURVE = 1
    SURFACE = 2
    CUBE = 3


VALID_CONTAINER_TOPOLOGY_IDS: frozenset[int] = frozenset(
    item.value for item in ContainerTopologyType
)
VALID_CONTAINER_TOPOLOGY_NAMES: frozenset[str] = frozenset(
    item.name for item in ContainerTopologyType
)

VALID_CONTAINER_TOPOLOGY_ID_TO_NAME: dict[int, str] = {
    item.value: item.name for item in ContainerTopologyType
}
VALID_CONTAINER_TOPOLOGY_NAME_TO_ID: dict[str, int] = {
    item.name: item.value for item in ContainerTopologyType
}
