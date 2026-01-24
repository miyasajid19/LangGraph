from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
import math


class QuadraticState(TypedDict, total=False):
    a: float
    b: float
    c: float
    expression: str
    discriminant: float
    root1: Annotated[float | None, "First root of the equation"]
    root2: Annotated[float | None, "Second root of the equation"]
    error: str


# --------------------
# Graph Definition
# --------------------

graph = StateGraph(QuadraticState)


def formulate_expression(state: QuadraticState) -> QuadraticState:
    a = state["a"]
    b = state["b"]
    c = state["c"]

    b_sign = "+" if b >= 0 else "-"
    c_sign = "+" if c >= 0 else "-"

    state["expression"] = f"{a}x^2 {b_sign} {abs(b)}x {c_sign} {abs(c)} = 0"
    return state


def calculate_discriminant(state: QuadraticState) -> QuadraticState:
    a = state["a"]

    if a == 0:
        state["error"] = "Coefficient 'a' cannot be zero. This is not a quadratic equation."
        return state

    b = state["b"]
    c = state["c"]

    state["discriminant"] = b**2 - 4 * a * c
    return state


def check_discriminant(state: QuadraticState) -> Literal[
    "unique_roots", "same_roots", "no_real_root"
]:
    if "error" in state:
        return "no_real_root"

    d = state["discriminant"]
    eps = 1e-9

    if d > eps:
        return "unique_roots"
    elif abs(d) <= eps:
        return "same_roots"
    else:
        return "no_real_root"


def unique_roots(state: QuadraticState) -> QuadraticState:
    a = state["a"]
    b = state["b"]
    d = state["discriminant"]

    sqrt_d = math.sqrt(d)
    state["root1"] = (-b + sqrt_d) / (2 * a)
    state["root2"] = (-b - sqrt_d) / (2 * a)

    return state


def same_roots(state: QuadraticState) -> QuadraticState:
    a = state["a"]
    b = state["b"]

    root = -b / (2 * a)
    state["root1"] = root
    state["root2"] = root

    return state


def no_real_root(state: QuadraticState) -> QuadraticState:
    state["root1"] = None
    state["root2"] = None
    return state


# --------------------
# Nodes
# --------------------

graph.add_node(
    "formulate_expression",
    formulate_expression,
    description="Formulate quadratic expression from coefficients a, b, c",
)

graph.add_node(
    "calculate_discriminant",
    calculate_discriminant,
    description="Calculate discriminant of the quadratic equation",
)

graph.add_node(
    "unique_roots",
    unique_roots,
    description="Calculate two unique real roots",
)

graph.add_node(
    "same_roots",
    same_roots,
    description="Calculate one real root (double root)",
)

graph.add_node(
    "no_real_root",
    no_real_root,
    description="Handle case with no real roots or errors",
)

# --------------------
# Edges
# --------------------

graph.add_edge(START, "formulate_expression")
graph.add_edge("formulate_expression", "calculate_discriminant")
graph.add_conditional_edges("calculate_discriminant", check_discriminant)

graph.add_edge("unique_roots", END)
graph.add_edge("same_roots", END)
graph.add_edge("no_real_root", END)

# --------------------
# Run
# --------------------

if __name__ == "__main__":
    workflow = graph.compile()

    test_cases = [
        {"a": 1, "b": -3, "c": 2},  # Two roots
        {"a": 1, "b": 2, "c": 1},   # Same roots
        {"a": 1, "b": 2, "c": 5},   # No real roots
        {"a": 0, "b": 2, "c": 1},   # Invalid
    ]

    for case in test_cases:
        result = workflow.invoke(case)
        print("\nInput:", case)
        print("Output:", result)
