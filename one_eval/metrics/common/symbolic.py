from typing import List, Any, Dict
try:
    from sympy import parse_expr, simplify
    from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

def compute_symbolic_match(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    if not HAS_SYMPY:
        return {"score": 0.0, "error": "sympy not installed. Please pip install sympy."}
        
    scores = []
    # Allow implicit multiplication (e.g. "2x" -> "2*x")
    transformations = (standard_transformations + (implicit_multiplication_application,))
    
    for p, r in zip(preds, refs):
        try:
            # Handle list refs (multi-gold)
            r_list = r if isinstance(r, list) else [r]
            is_match = False
            
            # Parse prediction
            # Note: evaluate=False might be safer, but we need simplify() which evaluates.
            # We catch exceptions for invalid syntax.
            p_str = str(p).strip()
            # Basic cleanup for LaTeX
            p_str = p_str.replace("\\", "") 
            
            p_expr = parse_expr(p_str, transformations=transformations)
            
            for r_item in r_list:
                try:
                    r_str = str(r_item).strip().replace("\\", "")
                    r_expr = parse_expr(r_str, transformations=transformations)
                    
                    # Check equivalence: simplify(p - r) == 0
                    diff = simplify(p_expr - r_expr)
                    if diff == 0:
                        is_match = True
                        break
                except Exception:
                    continue
            
            scores.append(1.0 if is_match else 0.0)
            
        except Exception:
            # Parse error or simplify error -> fail
            scores.append(0.0)
            
    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "details": scores
    }