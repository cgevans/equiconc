import ast
import inspect
from pathlib import Path

import equiconc


STUB_PATH = Path(__file__).resolve().parents[1] / "equiconc" / "__init__.pyi"


def _stub_module() -> ast.Module:
    return ast.parse(STUB_PATH.read_text())


def _class_def(name: str) -> ast.ClassDef:
    for node in _stub_module().body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is missing from {STUB_PATH}")


def _function_def(class_name: str, method_name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in _class_def(class_name).body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    ]
    if not matches:
        raise AssertionError(f"{class_name}.{method_name} is missing from {STUB_PATH}")
    # Overloaded methods end with the concrete implementation signature.
    return matches[-1]


def _stub_positional_args(class_name: str, method_name: str) -> list[str]:
    fn = _function_def(class_name, method_name)
    return [arg.arg for arg in fn.args.posonlyargs + fn.args.args if arg.arg != "self"]


def _stub_kwonly_args(class_name: str, method_name: str) -> list[str]:
    return [arg.arg for arg in _function_def(class_name, method_name).args.kwonlyargs]


def _runtime_positional_args(obj) -> list[str]:
    return [
        name
        for name, param in inspect.signature(obj).parameters.items()
        if name != "self"
        and param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]


def _runtime_kwonly_args(obj) -> list[str]:
    return [
        name
        for name, param in inspect.signature(obj).parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    ]


def _stub_properties(class_name: str) -> set[str]:
    return {
        node.name
        for node in _class_def(class_name).body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(decorator, ast.Name) and decorator.id == "property"
            for decorator in node.decorator_list
        )
    }


def _stub_public_methods(class_name: str) -> set[str]:
    return {
        node.name
        for node in _class_def(class_name).body
        if isinstance(node, ast.FunctionDef)
        and not node.name.startswith("_")
        and node.name not in _stub_properties(class_name)
    }


def test_solver_options_stub_matches_native_constructor_and_getters():
    assert _stub_kwonly_args("SolverOptions", "__init__") == _runtime_kwonly_args(
        equiconc.SolverOptions
    )

    native_getters = {
        name for name in dir(equiconc.SolverOptions()) if not name.startswith("_")
    }
    assert _stub_properties("SolverOptions") == native_getters


def test_system_stub_matches_native_public_surface():
    assert _stub_kwonly_args("System", "__init__") == _runtime_kwonly_args(equiconc.System)

    system = equiconc.System()
    native_methods = {name for name in dir(system) if not name.startswith("_")}
    assert _stub_public_methods("System") == native_methods

    assert _stub_positional_args("System", "monomer") == _runtime_positional_args(
        system.monomer
    )
    assert _stub_positional_args("System", "complex") == _runtime_positional_args(
        system.complex
    )
    assert _stub_kwonly_args("System", "complex") == _runtime_kwonly_args(system.complex)


def test_equilibrium_stub_matches_native_public_surface():
    eq = equiconc.System().monomer("A", 1e-9).equilibrium()
    native_members = {name for name in dir(eq) if not name.startswith("_")}
    stub_members = _stub_public_methods("Equilibrium") | _stub_properties("Equilibrium")
    assert stub_members == native_members
