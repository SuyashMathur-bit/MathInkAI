import re
import sympy as sp
from latex2sympy2 import latex2sympy
from .latex import classify_expression, validate_math_expression

def solve_latex(latex):
    """
    Direct equation solver from notebook Cell 61.
    """
    try:
        result = latex2sympy(latex)

        if isinstance(result, list):
            solutions = []
            for item in result:
                if isinstance(item, sp.Equality):
                    solutions.append(item.rhs)
                else:
                    solutions.append(item)
            return solutions

        if isinstance(result, sp.Equality):
            symbols = list(result.free_symbols)
            if symbols:
                return sp.solve(result, symbols[0])

        return result
    except Exception as e:
        return f"Error: {e}"

def process_mathink_expression(latex):
    """
    Complete Math Processing from notebook Cell 74.
    Handles equations, simplifications, and constant evaluations.
    """
    if not isinstance(latex, str) or not latex.strip():
        return {
            "latex": latex,
            "type": "invalid",
            "status": "invalid",
            "solution": None,
            "message": "No equation or expression provided."
        }

    expression_type = classify_expression(latex)

    try:
        if "=" in latex:
            lhs_text, rhs_text = latex.split("=", 1)
            lhs = latex2sympy(lhs_text)
            rhs = latex2sympy(rhs_text)

            if isinstance(lhs, list):
                lhs = lhs[0]
            if isinstance(rhs, list):
                rhs = rhs[0]

            equation = sp.Eq(lhs, rhs)
            variables = list(equation.free_symbols)

            if len(variables) == 0:
                is_true = sp.simplify(lhs - rhs) == 0
                return {
                    "latex": latex,
                    "type": expression_type,
                    "status": "valid",
                    "solution": "True (Identity)" if is_true else "False (Contradiction)",
                    "message": "Constant statement."
                }

            # Sort variables alphabetically for deterministic solving
            var = sorted(variables, key=lambda s: s.name)[0]
            solutions = sp.solve(equation, var)

            sol_strings = [f"{var} = {sp.latex(sol)}" for sol in solutions]
            sol_display = ", ".join(sol_strings) if sol_strings else "No solution found"

            return {
                "latex": latex,
                "type": expression_type,
                "status": "solved",
                "solution": sol_display,
                "raw_solutions": [str(s) for s in solutions],
                "message": "Equation solved successfully."
            }

        # Mathematical expression simplification
        expr = latex2sympy(latex)
        if isinstance(expr, list):
            expr = expr[0]

        simplified = sp.simplify(expr)
        return {
            "latex": latex,
            "type": expression_type,
            "status": "expression",
            "solution": sp.latex(simplified),
            "raw_solutions": [str(simplified)],
            "message": "Expression simplified successfully."
        }

    except Exception as e:
        return {
            "latex": latex,
            "type": expression_type,
            "status": "not_solved",
            "solution": None,
            "message": str(e)
        }

def step_by_step_solution(latex):
    """
    Step-by-step mathematical reasoning engine from notebook Cell 78.
    Generates educational step cards for display in the UI.
    """
    try:
        clean = latex.replace(" ", "")

        if "=" not in clean:
            expr = latex2sympy(clean)
            if isinstance(expr, list):
                expr = expr[0]
            simplified = sp.simplify(expr)
            return {
                "type": "expression",
                "steps": [
                    f"Original expression: $${latex}$$",
                    f"Combine like terms and apply algebraic identities.",
                    f"Simplified result: $${sp.latex(simplified)}$$"
                ],
                "answer": sp.latex(simplified)
            }

        lhs_text, rhs_text = clean.split("=", 1)
        lhs = latex2sympy(lhs_text)
        rhs = latex2sympy(rhs_text)

        if isinstance(lhs, list):
            lhs = lhs[0]
        if isinstance(rhs, list):
            rhs = rhs[0]

        equation = sp.Eq(lhs, rhs)
        variables = list(equation.free_symbols)

        if not variables:
            val = sp.simplify(lhs - rhs)
            return {
                "type": "constant equation",
                "steps": [
                    f"Given identity: $${sp.latex(equation)}$$",
                    f"Evaluation: $${sp.latex(lhs)} - {sp.latex(rhs)} = {sp.latex(val)}$$"
                ],
                "answer": "True" if val == 0 else "False"
            }

        var = sorted(variables, key=lambda s: s.name)[0]
        standard_form = sp.expand(lhs - rhs)
        degree = sp.degree(standard_form, var)

        if degree == 1:
            solution = sp.solve(equation, var)
            steps = [
                f"Given linear equation: $${sp.latex(equation)}$$",
                f"Rearrange terms by moving all terms containing $${var}$$ to the left and constants to the right:",
                f"$${sp.latex(standard_form)} = 0$$",
                f"Isolate the variable $${var}$$ by dividing by its coefficient:",
                f"$${var} = {sp.latex(solution[0])}$$"
            ]
            final_ans = f"{var} = {sp.latex(solution[0])}"
            return {
                "type": "linear equation",
                "steps": steps,
                "answer": final_ans
            }

        elif degree == 2:
            solution = sp.solve(equation, var)
            factorized = sp.factor(standard_form)
            steps = [
                f"Given quadratic equation: $${sp.latex(equation)}$$",
                f"Rewrite equation in standard form $$ax^2 + bx + c = 0$$:",
                f"$${sp.latex(standard_form)} = 0$$",
                f"Factorize the polynomial or apply the quadratic formula:",
                f"$${sp.latex(factorized)} = 0$$",
                f"Set each factor equal to zero to obtain the roots:"
            ]
            for sol in solution:
                steps.append(f"$${var} = {sp.latex(sol)}$$")

            final_ans = ", ".join(f"{var} = {sp.latex(sol)}" for sol in solution)
            return {
                "type": "quadratic equation",
                "steps": steps,
                "answer": final_ans
            }

        else:
            solution = sp.solve(equation, var)
            steps = [
                f"Given polynomial equation: $${sp.latex(equation)}$$",
                f"Rewrite equation in standard polynomial form:",
                f"$${sp.latex(standard_form)} = 0$$",
                f"Solve for roots of $${var}$$:",
                f"Roots: $${sp.latex(solution)}$$"
            ]
            final_ans = ", ".join(f"{var} = {sp.latex(sol)}" for sol in solution)
            return {
                "type": "polynomial equation",
                "steps": steps,
                "answer": final_ans
            }

    except Exception as e:
        return {
            "type": "error",
            "steps": [f"Could not compute symbolic steps: {str(e)}"],
            "answer": None
        }

def create_solution_output(latex):
    """
    Format complete output payload for API and UI rendering.
    """
    step_res = step_by_step_solution(latex)
    proc_res = process_mathink_expression(latex)

    return {
        "input_latex": latex,
        "expression_type": step_res.get("type", proc_res.get("type", "unknown")),
        "status": proc_res.get("status", "unknown"),
        "steps": step_res.get("steps", []),
        "final_answer": step_res.get("answer") or proc_res.get("solution") or "No closed-form solution",
        "message": proc_res.get("message", "")
    }
