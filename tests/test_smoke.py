import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from preprocessor import SOURCES, VALID_DESTINATIONS, get_validation_errors


def test_valid_destinations_exclude_source_city():
    for source in SOURCES:
        assert source not in VALID_DESTINATIONS[source]


def test_valid_route_has_no_validation_errors():
    source = SOURCES[0]
    destination = VALID_DESTINATIONS[source][0]

    errors, _warnings = get_validation_errors(
        source=source,
        destination=destination,
        airline="Indigo",
        stops="zero",
        passengers=1,
        dep_hour=10,
    )

    assert errors == []
