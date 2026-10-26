# -*- coding: utf-8 -*-
"""Retail EWS: preview/apply the approved inclusive TEK_END_DTE validity rule.
No recipes are executed. Writes only if APPLY_CHANGES=True and project key is main.
"""
import re
import ast
import dataiku
from collections import Counter

APPLY_CHANGES = False
MAIN_PROJECT_KEY = "CREDITRISKEARLYWARNINGSYSTEMRETAILPORTFOLIO"

project_key = dataiku.default_project_key()
project = dataiku.api_client().get_default_project()

# Case A: DATE '{reference_date}' < TRUNC(NVL(e.TEK_END_DTE, DATE '9999-12-31'))
# Also handles d.HISTORIC_DATE, md.snapshot_date, t.TRANSACTION_DTE, etc.
RULE_A = re.compile(
    r"(?P<ref>\bDATE\s*'[^']+'|\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)"
    r"(?P<before>\s*)<(?P<after>\s*)"
    r"(?P<end>TRUNC\s*\(\s*NVL\s*\(\s*"
    r"(?:[A-Za-z_]\w*\.)?TEK_END_DTE\s*,\s*"
    r"DATE\s*'9999-12-31'\s*\)\s*\))",
    re.IGNORECASE,
)

# Case B: NVL(e.TEK_END_DTE, DATE '9999-12-31') >= (DATE '{reference_date}' + 1)
# Change ONLY the end-date lower bound: >= DATE '{reference_date}'
# Keep the TEK_STRT_DTE < (date + 1) bound unchanged.
RULE_B = re.compile(
    r"(?P<end>\bNVL\s*\(\s*(?:[A-Za-z_]\w*\.)?"
    r"TEK_END_DTE\s*,\s*DATE\s*'9999-12-31'\s*\))"
    r"(?P<operator>\s*>=\s*)\(\s*"
    r"(?P<ref>DATE\s*'[^']+'|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)"
    r"\s*\+\s*1\s*\)",
    re.IGNORECASE,
)


def make_patch(code):
    # Restrict changes to the exact two identified validity patterns.
    changed_a, n_a = RULE_A.subn(
        lambda m: m.group("ref") + m.group("before") + "<=" +
                  m.group("after") + m.group("end"),
        code,
    )
    changed_b, n_b = RULE_B.subn(
        lambda m: m.group("end") + m.group("operator") + m.group("ref"),
        changed_a,
    )
    return changed_b, n_a, n_b


print("=" * 105)
print("RETAIL EWS - TEK_END_DTE INCLUSIVE VALIDITY PATCH")
print("=" * 105)
print("PROJECT:", project_key)
print("MODE:", "APPLY" if APPLY_CHANGES else "PREVIEW ONLY")
print("Rule A: date < TRUNC(NVL(TEK_END_DTE,...))  =>  date <= TRUNC(NVL(...))")
print("Rule B: NVL(TEK_END_DTE,...) >= (date + 1)  =>  >= date")

# SAFETY: changes can only be saved in the specified main project.
if APPLY_CHANGES and project_key != MAIN_PROJECT_KEY:
    raise ValueError(
        "Write blocked: current project is not the expected MAIN Retail EWS project. "
        "Open MAIN and rerun the preview before enabling APPLY_CHANGES. "
        "Current project: " + project_key
    )

proposals = []
scan_errors = []
for item in project.list_recipes():
    name = item.name if hasattr(item, "name") else item["name"]
    try:
        settings = project.get_recipe(name).get_settings()
        if not hasattr(settings, "get_code") or not hasattr(settings, "set_code"):
            continue
        old_code = settings.get_code() or ""
        new_code, n_a, n_b = make_patch(old_code)
        if old_code == new_code:
            continue
        # Validate Python syntax without executing the recipe.
        ast.parse(new_code, filename=name)
        old_lines = old_code.splitlines()
        new_lines = new_code.splitlines()
        if len(old_lines) != len(new_lines):
            raise ValueError("Unexpected line-count change")
        modified_lines = [
            (i + 1, before.strip(), after.strip())
            for i, (before, after) in enumerate(zip(old_lines, new_lines))
            if before != after
        ]
        proposals.append({
            "name": name,
            "n_a": n_a,
            "n_b": n_b,
            "changes": modified_lines,
            "old_code": old_code,
            "new_code": new_code,
        })
        print("\nRECIPE:", name, "| case A:", n_a, "| case B:", n_b, flush=True)
        for line_no, before, after in modified_lines[:16]:
            print("  {:4d} | BEFORE: {}".format(line_no, before[:220]))
            print("       | AFTER:  {}".format(after[:220]))
        if len(modified_lines) > 16:
            print("  ... another {} modified lines".format(len(modified_lines) - 16))
    except Exception as exc:
        scan_errors.append((name, str(exc)))
        print("SCAN ERROR:", name, str(exc)[:250], flush=True)

print("\n" + "=" * 105)
print("SUMMARY")
print("=" * 105)
print("Recipes with proposed changes:", len(proposals))
print("Conditions changing case A:", sum(x["n_a"] for x in proposals))
print("Conditions changing case B:", sum(x["n_b"] for x in proposals))
print("Inspection errors:", len(scan_errors))

# Do not write partial results if inspection was incomplete.
if APPLY_CHANGES:
    if scan_errors:
        raise RuntimeError("No changes saved: inspect scan errors first")
    if not proposals:
        print("No matching old validity patterns found. Nothing to save.")
    else:
        print("\nAPPLYING CHANGES IN MAIN PROJECT")
        saved_names = []
        try:
            for proposal in proposals:
                live_settings = project.get_recipe(proposal["name"]).get_settings()
                live_code = live_settings.get_code() or ""
                if live_code != proposal["old_code"]:
                    raise RuntimeError("Recipe changed since scan: " + proposal["name"])
                live_settings.set_code(proposal["new_code"])
                live_settings.save()
                saved_names.append(proposal["name"])
                print("SAVED:", proposal["name"], flush=True)
        except Exception:
            print("IMPORTANT: SAVE STOPPED. Previously saved recipes:", saved_names)
            print("Review Git diff; no automatic rollback was attempted.")
            raise
        print("\nSUCCESS: saved", len(saved_names), "recipes. Review Git diff before running.")
else:
    print("\nPREVIEW ONLY. No recipe was changed. Set APPLY_CHANGES=True ONLY in MAIN after reviewing the proposed changes.")

print("No recipe executed, and no dataset was rebuilt.")
