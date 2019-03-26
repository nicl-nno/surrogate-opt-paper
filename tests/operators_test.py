from src.evolution.operators import (
    default_operators
)


def test_default_operators_correct():
    operators = default_operators()

    population = operators.init_population(size=10)

    assert len(population) == 10
