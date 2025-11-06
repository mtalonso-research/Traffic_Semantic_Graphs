from nuplan.common.maps.abstract_map import AbstractMap, MapObject
from nuplan.common.maps.abstract_map_factory import AbstractMapFactory
from nuplan.common.actor_state.state_representation import Point2D
from typing import List, Optional, Tuple, Dict, Deque
from nuplan.common.maps.maps_datatypes import RasterLayer, RasterMap, SemanticMapLayer
import numpy as np
import numpy.typing as npt

class MockMapApi(AbstractMap):
    def __init__(self):
        pass

    def get_available_map_objects(self) -> List[SemanticMapLayer]:
        return []

    def get_available_raster_layers(self) -> List[SemanticMapLayer]:
        return []

    def get_raster_map_layer(self, layer: SemanticMapLayer) -> RasterLayer:
        return RasterLayer()

    def get_raster_map(self, layers: List[SemanticMapLayer]) -> RasterMap:
        return RasterMap()

    @property
    def map_name(self) -> str:
        return "mock_map"

    def get_all_map_objects(self, point: Point2D, layer: SemanticMapLayer) -> List[MapObject]:
        return []

    def get_one_map_object(self, point: Point2D, layer: SemanticMapLayer) -> Optional[MapObject]:
        return None

    def is_in_layer(self, point: Point2D, layer: SemanticMapLayer) -> bool:
        return False

    def get_proximal_map_objects(
        self, point: Point2D, radius: float, layers: List[SemanticMapLayer]
    ) -> Dict[SemanticMapLayer, List[MapObject]]:
        return {layer: [] for layer in layers}

    def get_map_object(self, object_id: str, layer: SemanticMapLayer) -> Optional[MapObject]:
        return None

    def get_distance_to_nearest_map_object(
        self, point: Point2D, layer: SemanticMapLayer
    ) -> Tuple[Optional[str], Optional[float]]:
        return None, None

    def get_distance_to_nearest_raster_layer(self, point: Point2D, layer: SemanticMapLayer) -> float:
        return float('inf')

    def get_distances_matrix_to_nearest_map_object(
        self, points: List[Point2D], layer: SemanticMapLayer
    ) -> Optional[npt.NDArray[np.float64]]:
        return None

    def initialize_all_layers(self) -> None:
        pass

class MockMapFactory(AbstractMapFactory):
    def __init__(self):
        pass

    def build_map_from_name(self, map_name: str) -> AbstractMap:
        return MockMapApi()