"""
AST metadata extraction using tree-sitter.

Parses code chunks with tree-sitter and extracts structural metadata:
function/class signatures, imports, complexity, and nesting depth.

This module provides a **reliable baseline** of code understanding that
works for every supported language without external tools.  It
complements SCIP: when SCIP data is available both are used; when SCIP
is missing the AST metadata ensures chunks still carry structural context.
"""

from __future__ import annotations

import logging
from typing import Optional

from lumen.config import ASTMetadata

logger = logging.getLogger(__name__)


# ── Per-language node-type mappings ──────────────────────────────────
#
# Tree-sitter grammars use different node type names per language.
# Each mapping tells us which AST node types correspond to functions,
# classes, imports, and branching constructs.

_LANG_CONFIG: dict[str, dict] = {
    "python": {
        "function_types": {"function_definition"},
        "class_types": {"class_definition"},
        "import_types": {"import_statement", "import_from_statement"},
        "branch_types": {
            "if_statement", "for_statement", "while_statement",
            "try_statement", "match_statement",
        },
        "name_field": "name",
        "params_field": "parameters",
        "return_field": "return_type",
        "method_types": {"function_definition"},  # methods are functions inside classes
    },
    "typescript": {
        "function_types": {"function_declaration", "arrow_function", "generator_function_declaration"},
        "class_types": {"class_declaration", "abstract_class_declaration"},
        "import_types": {"import_statement"},
        "branch_types": {
            "if_statement", "for_statement", "for_in_statement",
            "while_statement", "do_statement", "try_statement",
            "switch_statement", "ternary_expression",
        },
        "name_field": "name",
        "params_field": "parameters",
        "return_field": "return_type",
        "method_types": {"method_definition"},
        "extra_types": {"interface_declaration", "type_alias_declaration", "enum_declaration"},
    },
    "javascript": {
        "function_types": {"function_declaration", "arrow_function", "generator_function_declaration"},
        "class_types": {"class_declaration"},
        "import_types": {"import_statement"},
        "branch_types": {
            "if_statement", "for_statement", "for_in_statement",
            "while_statement", "do_statement", "try_statement",
            "switch_statement", "ternary_expression",
        },
        "name_field": "name",
        "params_field": "parameters",
        "return_field": None,
        "method_types": {"method_definition"},
    },
    "go": {
        "function_types": {"function_declaration"},
        "class_types": set(),  # Go uses type_declaration for structs
        "import_types": {"import_declaration"},
        "branch_types": {
            "if_statement", "for_statement", "switch_statement",
            "select_statement", "type_switch_statement",
        },
        "name_field": "name",
        "params_field": "parameters",
        "return_field": "result",
        "method_types": {"method_declaration"},
        "extra_types": {"type_declaration"},
    },
    "rust": {
        "function_types": {"function_item"},
        "class_types": {"struct_item", "enum_item", "trait_item"},
        "import_types": {"use_declaration"},
        "branch_types": {
            "if_expression", "for_expression", "while_expression",
            "loop_expression", "match_expression",
        },
        "name_field": "name",
        "params_field": "parameters",
        "return_field": "return_type",
        "method_types": {"function_item"},  # methods inside impl blocks
        "extra_types": {"impl_item"},
    },
    "java": {
        "function_types": set(),  # Java only has methods inside classes
        "class_types": {"class_declaration", "interface_declaration", "enum_declaration"},
        "import_types": {"import_declaration"},
        "branch_types": {
            "if_statement", "for_statement", "enhanced_for_statement",
            "while_statement", "do_statement", "try_statement",
            "switch_expression",
        },
        "name_field": "name",
        "params_field": "parameters",
        "return_field": "type",
        "method_types": {"method_declaration", "constructor_declaration"},
    },
    "ruby": {
        "function_types": {"method"},
        "class_types": {"class", "module"},
        "import_types": set(),  # Ruby uses `require` calls, not import nodes
        "branch_types": {
            "if", "unless", "while", "until", "for",
            "case", "begin",
        },
        "name_field": "name",
        "params_field": "parameters",
        "return_field": None,
        "method_types": {"method", "singleton_method"},
    },
    "cpp": {
        "function_types": {"function_definition"},
        "class_types": {"class_specifier", "struct_specifier"},
        "import_types": {"preproc_include"},
        "branch_types": {
            "if_statement", "for_statement", "while_statement",
            "do_statement", "switch_statement", "try_statement",
            "for_range_loop",
        },
        "name_field": "declarator",
        "params_field": "parameters",
        "return_field": "type",
        "method_types": {"function_definition"},
    },
    "c": {
        "function_types": {"function_definition"},
        "class_types": {"struct_specifier"},
        "import_types": {"preproc_include"},
        "branch_types": {
            "if_statement", "for_statement", "while_statement",
            "do_statement", "switch_statement",
        },
        "name_field": "declarator",
        "params_field": "parameters",
        "return_field": "type",
        "method_types": set(),
    },
    "csharp": {
        "function_types": set(),
        "class_types": {"class_declaration", "interface_declaration", "struct_declaration", "enum_declaration"},
        "import_types": {"using_directive"},
        "branch_types": {
            "if_statement", "for_statement", "foreach_statement",
            "while_statement", "do_statement", "try_statement",
            "switch_statement",
        },
        "name_field": "name",
        "params_field": "parameters",
        "return_field": "type",
        "method_types": {"method_declaration", "constructor_declaration"},
    },
}


# ── Tree-sitter parser cache ────────────────────────────────────────

_parser_cache: dict[str, object] = {}


def _get_parser(language: str):
    """Return a cached tree-sitter parser for *language*, or ``None``."""
    if language in _parser_cache:
        return _parser_cache[language]

    try:
        from tree_sitter_language_pack import get_parser  # noqa: WPS433
        parser = get_parser(language)
        _parser_cache[language] = parser
        return parser
    except Exception:  # noqa: BLE001
        _parser_cache[language] = None
        return None


# ── Core extraction ─────────────────────────────────────────────────


def _extract_name(node, lang_cfg: dict) -> str:
    """
    Extract the human-readable name from a definition node.

    Handles quirks like C/C++ where the name lives inside a
    nested ``declarator`` node.
    """
    name_field = lang_cfg.get("name_field", "name")
    name_node = node.child_by_field_name(name_field)

    if name_node is None:
        return ""

    # C/C++ function_definition: the "declarator" field contains a
    # function_declarator which in turn has an "declarator" for the name.
    if name_node.type in ("function_declarator", "pointer_declarator"):
        inner = name_node.child_by_field_name("declarator")
        if inner:
            return inner.text.decode(errors="replace")
    return name_node.text.decode(errors="replace")


def _build_signature(node, lang_cfg: dict, keyword: str = "def") -> str:
    """Build a human-readable signature from a function/class AST node."""
    name = _extract_name(node, lang_cfg)
    if not name:
        return ""

    params_field = lang_cfg.get("params_field", "parameters")
    params_node = node.child_by_field_name(params_field)
    params = params_node.text.decode(errors="replace") if params_node else ""

    return_field = lang_cfg.get("return_field")
    ret_node = node.child_by_field_name(return_field) if return_field else None
    ret = f" -> {ret_node.text.decode(errors='replace')}" if ret_node else ""

    return f"{keyword} {name}{params}{ret}"


def _count_branches(node, branch_types: set[str]) -> int:
    """Recursively count branching constructs in an AST subtree."""
    count = 1 if node.type in branch_types else 0
    for child in node.children:
        count += _count_branches(child, branch_types)
    return count


def _max_depth(node, current: int = 0) -> int:
    """Compute the maximum nesting depth of an AST subtree."""
    # These node types increase nesting depth
    scope_types = {
        "block", "statement_block", "compound_statement",
        "function_body", "class_body", "impl_item",
        "function_definition", "method_definition",
        "class_definition", "class_declaration",
    }
    depth = current + 1 if node.type in scope_types else current
    if not node.children:
        return depth
    return max(_max_depth(child, depth) for child in node.children)


def extract_ast_metadata(
    code: str,
    language: str,
) -> Optional[ASTMetadata]:
    """
    Parse *code* with tree-sitter and extract structural metadata.

    Parameters
    ----------
    code:
        Source code text (the chunk content).
    language:
        Tree-sitter language identifier (e.g. ``"python"``,
        ``"typescript"``, ``"go"``).

    Returns
    -------
    ``ASTMetadata`` on success, ``None`` if tree-sitter is unavailable
    or the language is not supported.
    """
    lang_cfg = _LANG_CONFIG.get(language)
    if lang_cfg is None:
        return None

    parser = _get_parser(language)
    if parser is None:
        return None

    try:
        tree = parser.parse(code.encode("utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("tree-sitter parse failed for %s: %s", language, exc)
        return None

    root = tree.root_node

    function_types = lang_cfg.get("function_types", set())
    class_types = lang_cfg.get("class_types", set())
    method_types = lang_cfg.get("method_types", set())
    import_types = lang_cfg.get("import_types", set())
    extra_types = lang_cfg.get("extra_types", set())
    branch_types = lang_cfg.get("branch_types", set())

    node_types: list[str] = []
    signatures: list[str] = []
    imports: list[str] = []

    for child in root.children:
        ntype = child.type

        # ── Imports ───────────────────────────────────────────
        if ntype in import_types:
            imports.append(child.text.decode(errors="replace").strip())
            node_types.append(ntype)
            continue

        # ── Top-level functions ───────────────────────────────
        if ntype in function_types:
            sig = _build_signature(child, lang_cfg, keyword="def")
            if sig:
                signatures.append(sig)
            node_types.append(ntype)
            continue

        # ── Classes / structs / interfaces ────────────────────
        if ntype in class_types:
            name = _extract_name(child, lang_cfg)
            if name:
                signatures.append(f"class {name}")
            # Extract methods inside the class body
            _extract_class_members(child, lang_cfg, method_types, signatures)
            node_types.append(ntype)
            continue

        # ── Extra types (Go type_declaration, TS interfaces, etc.)
        if ntype in extra_types:
            name = _extract_name(child, lang_cfg)
            if name:
                signatures.append(f"type {name}")
            node_types.append(ntype)
            continue

        # ── Go method declarations (receiver-based) ──────────
        if ntype in method_types and ntype not in function_types:
            sig = _build_signature(child, lang_cfg, keyword="method")
            if sig:
                signatures.append(sig)
            node_types.append(ntype)
            continue

        # ── Export wrappers (TypeScript: export class ...) ────
        if ntype == "export_statement":
            for inner in child.children:
                if inner.type in class_types:
                    name = _extract_name(inner, lang_cfg)
                    if name:
                        signatures.append(f"class {name}")
                    _extract_class_members(inner, lang_cfg, method_types, signatures)
                    node_types.append(inner.type)
                elif inner.type in function_types:
                    sig = _build_signature(inner, lang_cfg, keyword="def")
                    if sig:
                        signatures.append(sig)
                    node_types.append(inner.type)
                elif inner.type in extra_types:
                    name = _extract_name(inner, lang_cfg)
                    if name:
                        signatures.append(f"type {name}")
                    node_types.append(inner.type)
            continue

        # ── Decorated definitions (Python: @decorator\ndef ...) ─
        if ntype == "decorated_definition":
            for inner in child.children:
                if inner.type in function_types:
                    sig = _build_signature(inner, lang_cfg, keyword="def")
                    if sig:
                        signatures.append(sig)
                    node_types.append(inner.type)
                elif inner.type in class_types:
                    name = _extract_name(inner, lang_cfg)
                    if name:
                        signatures.append(f"class {name}")
                    _extract_class_members(inner, lang_cfg, method_types, signatures)
                    node_types.append(inner.type)
            continue

    # ── Complexity + depth (computed over the full tree) ──────
    complexity = _count_branches(root, branch_types)
    nesting = _max_depth(root)

    return ASTMetadata(
        node_types=node_types,
        signatures=signatures,
        imports=imports,
        complexity=complexity,
        nesting_depth=nesting,
    )


def _extract_class_members(
    class_node,
    lang_cfg: dict,
    method_types: set[str],
    signatures: list[str],
) -> None:
    """Walk a class body and append method signatures."""
    body = class_node.child_by_field_name("body")
    if body is None:
        return

    for member in body.children:
        if member.type in method_types:
            sig = _build_signature(member, lang_cfg, keyword="  method")
            if sig:
                signatures.append(sig)
        # Handle Python: methods inside classes are function_definitions
        if member.type in lang_cfg.get("function_types", set()) and member.type not in method_types:
            sig = _build_signature(member, lang_cfg, keyword="  method")
            if sig:
                signatures.append(sig)


def ast_metadata_text(metadata: Optional[ASTMetadata]) -> str:
    """
    Render AST metadata into a human-readable header for embedding.

    Similar to ``_symbol_metadata_text`` for SCIP — this text is
    prepended to the chunk content so the embedding model captures
    structural context.
    """
    if metadata is None:
        return ""

    if not metadata.signatures and not metadata.imports:
        return ""

    parts = ["[AST Context]"]

    if metadata.imports:
        for imp in metadata.imports[:10]:  # cap at 10 to stay concise
            parts.append(f"  import: {imp[:120]}")

    for sig in metadata.signatures:
        parts.append(f"  {sig[:150]}")

    if metadata.complexity > 0:
        parts.append(f"  complexity: {metadata.complexity} branches")

    return "\n".join(parts) + "\n\n"
