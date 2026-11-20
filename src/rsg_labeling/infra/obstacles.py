from enum import IntEnum


class ObstacleClass(IntEnum):
    """Obstacle class IDs consumed by geometry/table adapters."""

    OBSTACLE_CLASS_INVALID = 0  #  invalid class
    # Vehicles
    OBSTACLE_CLASS_VEHICLE_UNKNOWN = 1280  # DW_OBSTACLE_CLASS_VEHICLE_BASE + 0
    OBSTACLE_CLASS_VEHICLE_CAR = 1281  # DW_OBSTACLE_CLASS_VEHICLE_BASE + 1
    OBSTACLE_CLASS_VEHICLE_TRUCK = 1282  # DW_OBSTACLE_CLASS_VEHICLE_BASE + 2
    OBSTACLE_CLASS_VEHICLE_BUS = 1283  # DW_OBSTACLE_CLASS_VEHICLE_BASE + 3
    OBSTACLE_CLASS_VEHICLE_EMERGENCY = 1284  # DW_OBSTACLE_CLASS_VEHICLE_BASE + 4
    OBSTACLE_CLASS_VEHICLE_CONSTRUCTION = 1285  # DW_OBSTACLE_CLASS_VEHICLE_BASE + 5
    OBSTACLE_CLASS_VEHICLE_POLICE = 1286  # DW_OBSTACLE_CLASS_VEHICLE_BASE + 6
    OBSTACLE_CLASS_VEHICLE_SCHOOL_BUS = 1287  # DW_OBSTACLE_CLASS_VEHICLE_BASE + 7
    # Two wheeled vehicles
    OBSTACLE_CLASS_BIKE_UNKNOWN = 2304  # DW_OBSTACLE_CLASS_BIKE_BASE + 0
    OBSTACLE_CLASS_BIKE = 2305  # DW_OBSTACLE_CLASS_BIKE_BASE + 1
    OBSTACLE_CLASS_BIKE_MOTOR = 2306  # DW_OBSTACLE_CLASS_BIKE_BASE + 2
    # Two wheeled vehicles with rider
    OBSTACLE_CLASS_BIKE_UNKNOWN_WITH_RIDER = 2307  # DW_OBSTACLE_CLASS_BIKE_BASE + 3
    OBSTACLE_CLASS_BIKE_WITH_RIDER = 2308  # DW_OBSTACLE_CLASS_BIKE_BASE + 4
    OBSTACLE_CLASS_BIKE_MOTOR_WITH_RIDER = 2309  # DW_OBSTACLE_CLASS_BIKE_BASE + 5
    OBSTACLE_CLASS_TRICYCLE = 2310  # DW_OBSTACLE_CLASS_BIKE_BASE + 6
    # Pedestrians
    OBSTACLE_CLASS_PEDESTRIAN_UNKNOWN = 4352  # DW_OBSTACLE_CLASS_PEDESTRIAN_BASE + 0
    OBSTACLE_CLASS_PEDESTRIAN_ADULT = 4353  # DW_OBSTACLE_CLASS_PEDESTRIAN_BASE + 1
    OBSTACLE_CLASS_PEDESTRIAN_CHILD = 4354  # DW_OBSTACLE_CLASS_PEDESTRIAN_BASE + 2
    OBSTACLE_CLASS_PEDESTRIAN_CONSTRUCTION_WORKER = 4355  # DW_OBSTACLE_CLASS_PEDESTRIAN_BASE + 3
    OBSTACLE_CLASS_PEDESTRIAN_OFFICIAL = 4356  # DW_OBSTACLE_CLASS_PEDESTRIAN_BASE + 4
    # Animals
    OBSTACLE_CLASS_ANIMAL_UNKNOWN = 8448  # DW_OBSTACLE_CLASS_ANIMAL_BASE + 0
    # small animal (bird,squirrel,cat)
    OBSTACLE_CLASS_ANIMAL_SMALL = 8449  # DW_OBSTACLE_CLASS_ANIMAL_BASE + 1
    # medium animal (like a large dog)
    OBSTACLE_CLASS_ANIMAL_MEDIUM = 8450  # DW_OBSTACLE_CLASS_ANIMAL_BASE + 2
    # large animal (horse,cow,deer,moose)
    OBSTACLE_CLASS_ANIMAL_LARGE = 8451  # DW_OBSTACLE_CLASS_ANIMAL_BASE + 3
    # Objects (in-animate). Official objects are obstacles on the road put there by a
    # road authority, either temporarily or permanently
    OBSTACLE_CLASS_OBJECT_UNKNOWN = 16384  # DW_OBSTACLE_CLASS_OBJECT_BASE + 0
    OBSTACLE_CLASS_OBJECT_HAZARD = 16385  # DW_OBSTACLE_CLASS_OBJECT_BASE + 1
    OBSTACLE_CLASS_OBJECT_OFFICIAL = 16386  # DW_OBSTACLE_CLASS_OBJECT_BASE + 2
    OBSTACLE_CLASS_OBJECT_CONE = 16387  # DW_OBSTACLE_CLASS_OBJECT_BASE + 3
    # Physical barriers:
    # - below bumper obstacle
    OBSTACLE_CLASS_OBJECT_CURB = 16388  # DW_OBSTACLE_CLASS_OBJECT_BASE + 4
    # - taller obstacle
    OBSTACLE_CLASS_OBJECT_BARRIER = 16389  # DW_OBSTACLE_CLASS_OBJECT_BASE + 5
    # - low barrier splitting lanes
    OBSTACLE_CLASS_OBJECT_PHYSICAL_LANE_LINE = 16390  # DW_OBSTACLE_CLASS_OBJECT_BASE + 6
    # Special obstacles:
    # - Overpass, parking garage ceiling
    OBSTACLE_CLASS_OBJECT_HARD_OVERHANG = 16391  # DW_OBSTACLE_CLASS_OBJECT_BASE + 7
    # - Tree branches, rope
    OBSTACLE_CLASS_OBJECT_SOFT_OVERHANG = 16392  # DW_OBSTACLE_CLASS_OBJECT_BASE + 8
    # - Obstacle drivable at some speed
    OBSTACLE_CLASS_OBJECT_SPEED_BUMP = 16393  # DW_OBSTACLE_CLASS_OBJECT_BASE + 9
    # - Cavity region
    OBSTACLE_CLASS_OBJECT_POT_HOLE = 16394  # DW_OBSTACLE_CLASS_OBJECT_BASE + 10
    # - Other negative obstacle (drop off)
    OBSTACLE_CLASS_OBJECT_NEGATIVE = 16395  # DW_OBSTACLE_CLASS_OBJECT_BASE + 11
    # - non-physical treated as obstacle
    OBSTACLE_CLASS_OBJECT_FORBIDDEN = 16396  # DW_OBSTACLE_CLASS_OBJECT_BASE + 12

    # Other
    OBSTACLE_CLASS_UNDEFINED_STATIC = 32768  # DW_OBSTACLE_CLASS_UNDEFINED_STATIC_BASE + 0
    OBSTACLE_CLASS_UNDEFINED_ANIMATE = 33024  # DW_OBSTACLE_CLASS_UNDEFINED_ANIMATE_BASE + 0
    # Virtual obstacles
    # - Paint lanes, gore areas, boundaries
    OBSTACLE_CLASS_FORBIDDEN_LANE_LINE = 513  # DW_OBSTACLE_CLASS_IS_VIRTUAL + 1
    OBSTACLE_CLASS_CROSSABLE_LANE_LINE = 514  # DW_OBSTACLE_CLASS_IS_VIRTUAL + 2
    OBSTACLE_CLASS_GORE_AREA = 515  # DW_OBSTACLE_CLASS_IS_VIRTUAL + 3
    OBSTACLE_CLASS_OCCLUSION = 516  # DW_OBSTACLE_CLASS_IS_VIRTUAL + 4


NATURAL_LANGUAGE_NAME_MAP = {
    # buf:lint:ignore ENUM_ZERO_VALUE_SUFFIX
    "OBSTACLE_CLASS_INVALID": "Invalid",
    # Vehicles
    "OBSTACLE_CLASS_VEHICLE_UNKNOWN": "Vehicle",
    "OBSTACLE_CLASS_VEHICLE_CAR": "Car",
    "OBSTACLE_CLASS_VEHICLE_TRUCK": "Truck",
    "OBSTACLE_CLASS_VEHICLE_BUS": "Bus",
    "OBSTACLE_CLASS_VEHICLE_EMERGENCY": "Emergency vehicle",
    "OBSTACLE_CLASS_VEHICLE_CONSTRUCTION": "Construction vehicle",
    "OBSTACLE_CLASS_VEHICLE_POLICE": "Police car",
    "OBSTACLE_CLASS_VEHICLE_SCHOOL_BUS": "School bus",
    # Two wheeled vehicles
    "OBSTACLE_CLASS_BIKE_UNKNOWN": "Bike",
    "OBSTACLE_CLASS_BIKE": "Bike",
    "OBSTACLE_CLASS_BIKE_MOTOR": "Motorcycle",
    # Two wheeled vehicles with rider
    "OBSTACLE_CLASS_BIKE_UNKNOWN_WITH_RIDER": "Bike with rider",
    "OBSTACLE_CLASS_BIKE_WITH_RIDER": "Bike with rider",
    "OBSTACLE_CLASS_BIKE_MOTOR_WITH_RIDER": "Motorcycle with rider",
    "OBSTACLE_CLASS_TRICYCLE": "Tricycle",
    # Pedestrians
    "OBSTACLE_CLASS_PEDESTRIAN_UNKNOWN": "Pedestrian",
    "OBSTACLE_CLASS_PEDESTRIAN_ADULT": "Pedestrian",
    "OBSTACLE_CLASS_PEDESTRIAN_CHILD": "Child pedestrian",
    "OBSTACLE_CLASS_PEDESTRIAN_CONSTRUCTION_WORKER": "Construction worker",
    "OBSTACLE_CLASS_PEDESTRIAN_OFFICIAL": "Law enforcement officer",
    # Animals
    "OBSTACLE_CLASS_ANIMAL_UNKNOWN": "Animal",
    # small animal (bird,squirrel,cat)
    "OBSTACLE_CLASS_ANIMAL_SMALL": "Small animal",
    # medium animal (like a large dog)
    "OBSTACLE_CLASS_ANIMAL_MEDIUM": "Medium animal",
    # large animal (horse,cow,deer,moose)
    "OBSTACLE_CLASS_ANIMAL_LARGE": "Large animal",
    # Objects (in-animate). Official objects are obstacles on the road put there by a
    # road authority, either temporarily or permanently
    "OBSTACLE_CLASS_OBJECT_UNKNOWN": "Object",
    "OBSTACLE_CLASS_OBJECT_HAZARD": "Hazard",
    "OBSTACLE_CLASS_OBJECT_OFFICIAL": "Official object",
    "OBSTACLE_CLASS_OBJECT_CONE": "Cone",
    # Physical barriers:
    # - below bumper obstacle
    "OBSTACLE_CLASS_OBJECT_CURB": "Curb",
    # - taller obstacle
    "OBSTACLE_CLASS_OBJECT_BARRIER": "Barrier",
    # - low barrier splitting lanes
    "OBSTACLE_CLASS_OBJECT_PHYSICAL_LANE_LINE": "Physical lane line",
    # Special obstacles:
    # - Overpass, parking garage ceiling
    "OBSTACLE_CLASS_OBJECT_HARD_OVERHANG": "Hard overhang",
    # - Tree branches, rope
    "OBSTACLE_CLASS_OBJECT_SOFT_OVERHANG": "Soft overhang",
    # - Obstacle drivable at some speed
    "OBSTACLE_CLASS_OBJECT_SPEED_BUMP": "Speed bump",
    # - Cavity region
    "OBSTACLE_CLASS_OBJECT_POT_HOLE": "Pot hole",
    # - Other negative obstacle (drop off)
    "OBSTACLE_CLASS_OBJECT_NEGATIVE": "Negative obstacle",
    # - non-physical treated as obstacle
    "OBSTACLE_CLASS_OBJECT_FORBIDDEN": "Forbidden obstacle",
    # Other
    "OBSTACLE_CLASS_UNDEFINED_STATIC": "Undefined static",
    "OBSTACLE_CLASS_UNDEFINED_ANIMATE": "Undefined animate",
    # Virtual obstacles
    # - Paint lanes, gore areas, boundaries
    "OBSTACLE_CLASS_FORBIDDEN_LANE_LINE": "Forbidden lane line",
    "OBSTACLE_CLASS_CROSSABLE_LANE_LINE": "Crossable lane line",
    "OBSTACLE_CLASS_GORE_AREA": "Gore area",
    "OBSTACLE_CLASS_OCCLUSION": "Occlusion",
}
