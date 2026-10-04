import re
import sympy as sp
from latex2sympy2 import latex2sympy

def tokens_to_latex(tokens):
    """
    Convert MathInk.ai predicted token list into a LaTeX string.
    Exactly preserves project implementation: ''.join(tokens).
    """
    if not tokens:
        return ""
    latex = "".join(tokens)
    return latex

def clean_latex_string(latex):
    """
    Clean minor LaTeX formatting quirks (e.g. trailing braces, extra dollars).
    """
    if not latex:
        return ""
    cleaned = latex.replace("$", "").strip()
    return cleaned

def classify_expression(latex):
    """
    Classify mathematical LaTeX expression into:
    - linear equation
    - quadratic equation
    - cubic equation
    - polynomial equation
    - constant equation
    - arithmetic expression
    - polynomial expression
    - mathematical expression
    """
    try:
        clean = latex.replace(" ", "")

        if "=" in clean:
            lhs_text, rhs_text = clean.split("=", 1)

            lhs = latex2sympy(lhs_text)
            rhs = latex2sympy(rhs_text)

            if isinstance(lhs, list):
                lhs = lhs[0]
            if isinstance(rhs, list):
                rhs = rhs[0]

            equation = sp.expand(lhs - rhs)
            variables = list(equation.free_symbols)

            if len(variables) == 0:
                return "constant equation"

            var = variables[0]
            degree = sp.degree(equation, var)

            if degree == 1:
                return "linear equation"
            elif degree == 2:
                return "quadratic equation"
            elif degree == 3:
                return "cubic equation"
            elif degree is not None and degree > 3:
                return "polynomial equation"
            else:
                return "non-polynomial equation"

        expr = latex2sympy(clean)
        if isinstance(expr, list):
            expr = expr[0]

        variables = list(expr.free_symbols)
        if len(variables) == 0:
            return "arithmetic expression"

        if expr.is_polynomial(*variables):
            return "polynomial expression"

        return "mathematical expression"

    except Exception as e:
        return "mathematical expression"

def validate_math_expression(latex):
    """
    Validate predicted LaTeX expression before solving.
    Returns status dict with parsed SymPy expression.
    """
    try:
        expr = latex2sympy(latex)

        if isinstance(expr, list):
            expr = expr[0]

        if isinstance(expr, sp.Equality):
            functions = expr.atoms(sp.Function)
            if functions:
                return {
                    "status": "invalid_for_solving",
                    "reason": "Expression contains undefined functions",
                    "expression": expr
                }
            return {
                "status": "valid_equation",
                "reason": "Valid mathematical equation",
                "expression": expr
            }

        return {
            "status": "valid_expression",
            "reason": "Valid mathematical expression",
            "expression": expr
        }

    except Exception as e:
        return {
            "status": "invalid",
            "reason": str(e),
            "expression": None
        }
