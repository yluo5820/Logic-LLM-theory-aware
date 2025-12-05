import re
from itertools import product

class TheoryAwareTranslator:
    """
    Translates CLOVER DSL into Theory-Aware Z3 code (QF_LIA).
    
    Compatible with:
    1. AR-LSAT (Sorting, Ordering, Counting)
    2. ZebraLogic (Bijective mappings, Distinct constraints)
    
    Key Features:
    - Unrolls EnumSorts into Integer constants.
    - Bit-blasts Functions into Integer variables with domain bounds.
    - Recursively unrolls ForAll/Exists/Count/Distinct.
    - Injects Python helper functions to handle complex option nesting naturally.
    """

    def __init__(self, declared_enums, declared_ints, declared_lists, declared_functions, dataset_name="AR-LSAT"):
        self.enums = declared_enums
        self.ints = declared_ints
        self.lists = declared_lists
        self.funcs = declared_functions
        self.dataset_name = dataset_name
        
        # Consolidate all finite domains
        self.all_sorts = {}
        self.all_sorts.update(self.enums)
        self.all_sorts.update(self.ints)
        self.all_sorts.update(self.lists)

    def generate_preamble(self):
        lines = [
            "from z3 import *",
            "",
            "# --- Helper Functions (DSL Emulation) ---",
            "def BoolToInt(b): return If(b, 1, 0)",
            # is_exception handles AR-LSAT's "Find the option that is FALSE" logic
            "def is_exception(x): return not x",
            "",
            "# Standard validity checks that reuse the global pre_conditions list",
            "# FIX: Use *pre_conditions to unpack the list into arguments",
            "def is_valid(phi):",
            "    s = Solver()",
            "    s.add(*pre_conditions)",
            "    s.add(Not(phi))",
            "    return s.check() == unsat",
            "",
            "def is_sat(phi):",
            "    s = Solver()",
            "    s.add(*pre_conditions)",
            "    s.add(phi)",
            "    return s.check() == sat",
            "",
            "def is_unsat(phi):",
            "    return not is_sat(phi)",
            "",
            "pre_conditions = []",
            "",
            "# --- Constants & Sorts ---"
        ]
        
        # 1. Define Constants for Enum Members (e.g. Fred=1, Juan=2)
        for name, members in self.all_sorts.items():
            # Skip if list is already integers
            if all(str(m).isdigit() for m in members): continue
            
            lines.append(f"# Sort: {name}")
            for i, member in enumerate(members):
                lines.append(f"{member} = {i + 1}")
            lines.append("")

        # 2. Define Ground Variables (Bit-blasting)
        lines.append("# --- Ground Variables (QF_LIA) ---")
        for f_name, args in self.funcs.items():
            domain_sorts = args[:-1]
            codomain_sort = args[-1]
            
            # Resolve domain values
            domain_values_list = []
            for sort in domain_sorts:
                if sort in self.all_sorts:
                    domain_values_list.append(self.all_sorts[sort])
                elif sort == 'int' or sort == 'bool':
                     raise ValueError(f"Cannot unroll infinite sort '{sort}' in function '{f_name}'.")

            # Resolve Codomain Bounds
            min_val, max_val = 1, 10
            if codomain_sort in self.all_sorts:
                vals = self.all_sorts[codomain_sort]
                if all(str(v).isdigit() for v in vals):
                    nums = [int(v) for v in vals]
                    min_val, max_val = min(nums), max(nums)
                else:
                    min_val, max_val = 1, len(vals)
            elif codomain_sort == "bool":
                min_val, max_val = 0, 1
            
            # Generate Cartesian Product
            for combo in product(*domain_values_list):
                clean_combo = [str(c) for c in combo]
                var_name = f"{f_name}_{'_'.join(clean_combo)}"
                
                lines.append(f"{var_name} = Int('{var_name}')")
                # Add bounds to pre_conditions
                lines.append(f"pre_conditions.append(And({var_name} >= {min_val}, {var_name} <= {max_val}))")
            lines.append("")
            
        return lines

    def extract_paired_token_index(self, statement, start_index, left_token, right_token):
        """Standard CLOVER token parser to handle nesting correctly."""
        if start_index < 0 or start_index >= len(statement) or statement[start_index] != left_token:
            return -1
        level = 1
        for i in range(start_index + 1, len(statement)):
            if statement[i] == left_token:
                level += 1
            elif statement[i] == right_token:
                level -= 1
                if level == 0:
                    return i
        return -1

    def substitute_bound_vars(self, expr, scope):
        """Safe substitution using word boundaries."""
        for var_name, val in scope.items():
            pattern = r"\b" + re.escape(var_name) + r"\b"
            expr = re.sub(pattern, str(val), expr)
        return expr

    def resolve_functions(self, expr):
        """
        Replaces 'assigned(Fred)' with 'assigned_Fred'.
        Only targets declared user functions.
        """
        if not self.funcs: return expr
        func_names = [re.escape(k) for k in self.funcs.keys()]
        pattern = r"\b(" + "|".join(func_names) + r")\s*\(([^)]+)\)"
        
        def replacer(match):
            f_name = match.group(1)
            args_str = match.group(2)
            args = [x.strip() for x in args_str.split(",")]
            return f"{f_name}_{'_'.join(args)}"
        
        return re.sub(pattern, replacer, expr)

    def recursive_translate(self, stmt, scope):
        stmt = stmt.strip()
        
        # --- 1. DETECT QUANTIFIERS (ForAll, Exists, Count, Distinct) ---
        keywords = ["ForAll", "Exists", "Count", "Distinct"]
        first_kw = None
        min_idx = len(stmt)
        
        # Find the *first* keyword to respect nesting order
        for kw in keywords:
            match = re.search(r"\b" + kw + r"\s*\(", stmt)
            if match and match.start() < min_idx:
                min_idx = match.start()
                first_kw = kw

        if first_kw:
            # Parse the Quantifier Block
            open_paren = stmt.find("(", min_idx)
            scope_start = stmt.find("[", open_paren)
            scope_end = self.extract_paired_token_index(stmt, scope_start, '[', ']')
            close_paren = self.extract_paired_token_index(stmt, open_paren, '(', ')')
            
            # Check if we successfully parsed the [vars] block.
            # If not (e.g. Distinct(a,b) with no brackets), fall through to Base Case.
            if scope_start != -1 and scope_end != -1 and close_paren != -1:
                prefix = stmt[:min_idx]
                suffix = stmt[close_paren+1:]
                
                var_decl_str = stmt[scope_start+1 : scope_end] # "c:children"
                body_str = stmt[scope_end+1 : close_paren].strip()
                if body_str.startswith(","): body_str = body_str[1:].strip()
                
                # Parse Vars
                vars_def = []
                for v in var_decl_str.split(','):
                    v = v.strip()
                    if ':' in v:
                        v_name, v_sort = v.split(':')
                        vars_def.append((v_name.strip(), v_sort.strip()))
                    else:
                        # Fallback for implicit sorts
                        pass

                # Get Domains
                iterables = []
                for v_name, v_sort in vars_def:
                    if v_sort in self.all_sorts:
                        iterables.append(self.all_sorts[v_sort])
                    else:
                        raise ValueError(f"Unknown sort '{v_sort}' in: {stmt}")

                # Unroll
                unrolled_parts = []
                for combo in product(*iterables):
                    new_scope = scope.copy()
                    for i, val in enumerate(combo):
                        new_scope[vars_def[i][0]] = val
                    
                    # Recurse
                    trans_body = self.recursive_translate(body_str, new_scope)
                    
                    if first_kw == "Count":
                        unrolled_parts.append(f"BoolToInt({trans_body})")
                    else:
                        unrolled_parts.append(trans_body)

                # Reassemble
                if first_kw == "ForAll":
                    replacement = f"And({', '.join(unrolled_parts)})"
                elif first_kw == "Exists":
                    replacement = f"Or({', '.join(unrolled_parts)})"
                elif first_kw == "Count":
                    replacement = f"Sum({', '.join(unrolled_parts)})"
                elif first_kw == "Distinct":
                    # Z3 Distinct takes variable args
                    replacement = f"Distinct({', '.join(unrolled_parts)})"

                return (self.recursive_translate(prefix, scope) + 
                        replacement + 
                        self.recursive_translate(suffix, scope))

        # --- 2. BASE CASE ---
        # Substitute variables -> Resolve Functions
        stmt = self.substitute_bound_vars(stmt, scope)
        stmt = self.resolve_functions(stmt)
        
        return stmt

    def translate(self, constraints, options):
        code_lines = self.generate_preamble()
        
        code_lines.append("# --- Problem Constraints ---")
        for c in constraints:
            # We translate the constraint and append it to pre_conditions
            t_c = self.recursive_translate(c, {})
            code_lines.append(f"pre_conditions.append({t_c})")
        
        code_lines.append("")
        code_lines.append("# --- Options / Query ---")
        
        if isinstance(options, str):
            options = [options]

        for i, opt in enumerate(options):
            t_opt = self.recursive_translate(opt, {})
            
            # Print format adaptation
            if self.dataset_name == "ZebraLogic":
                # ZebraLogic often expects a simple "True" or validation check
                # Note: LogicInferenceEngine for ZebraLogic often checks: if "A" == predicted
                # But sometimes it parses specific output.
                # Safest generic fallback for valid/sat checks:
                code_lines.append(f"if {t_opt}: print('True')")
            else:
                # AR-LSAT expects labeled options
                code_lines.append(f"if {t_opt}: print('({chr(65+i)})')")
            
        return "\n".join(code_lines)