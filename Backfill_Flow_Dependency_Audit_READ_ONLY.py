# Backfill Flow impact check: metadata only, no data reads or writes.
import dataiku
from collections import defaultdict, deque

project = dataiku.api_client().get_default_project()
check_pairs = [
    ("Retail_Portfolio_Daily", "vars_behavioural_daily"),
    ("Retail_Portfolio_Daily_Previous", "vars_behavioural_daily_Previous"),
    ("Retail_Portfolio_Daily", "Entity_Information"),
    ("Retail_Portfolio_Daily", "RM_Data_Daily_Run"),
]

def short_ref(value):
    return str(value).split(".")[-1]

producers = defaultdict(list)
inputs_by_recipe = {}
failures = []
for item in project.list_recipes():
    recipe_name = item.get("name") if isinstance(item, dict) else getattr(item, "name", None)
    if not recipe_name:
        continue
    try:
        settings = project.get_recipe(recipe_name).get_settings()
        raw = settings.get_recipe_raw_definition()
        in_refs = {
            short_ref(ref.get("ref"))
            for role in raw.get("inputs", {}).values()
            for ref in role.get("items", [])
            if ref.get("ref")
        }
        out_refs = {
            short_ref(ref.get("ref"))
            for role in raw.get("outputs", {}).values()
            for ref in role.get("items", [])
            if ref.get("ref")
        }
        inputs_by_recipe[recipe_name] = in_refs
        for out_name in out_refs:
            producers[out_name].append(recipe_name)
    except Exception as exc:
        failures.append((recipe_name, type(exc).__name__, str(exc)[:180]))

def path_to_dependency(start, target):
    # Trace a dataset backwards through producing recipes and their inputs.
    work = deque([(start, [start])])
    seen = set()
    while work:
        current, path = work.popleft()
        if current == target:
            return path
        if current in seen:
            continue
        seen.add(current)
        for recipe_name in producers.get(current, []):
            for input_ds in inputs_by_recipe.get(recipe_name, []):
                work.append((input_ds, path + ["recipe:" + recipe_name, input_ds]))
    return None

print("=" * 88)
print("BACKFILL FLOW DEPENDENCIES - READ ONLY")
print("=" * 88)
print("Recipes inspected:", len(inputs_by_recipe))
for source, target in check_pairs:
    path = path_to_dependency(source, target)
    print("\nScope source:", source)
    print("Proposed recipe output:", target)
    if path:
        print("RESULT: CYCLE RISK if adding scope as an input")
        print("Dependency path:", " -> ".join(path))
    else:
        print("RESULT: no declared dependency path found")
        print("NOTE: this does not rule out undeclared indirect SQL references")

if failures:
    print("\nWARNING: recipe metadata could not be read for", len(failures), "recipes")
    for failure in failures[:15]:
        print(failure)

for dataset_name in ["Entity_Information", "Retail_Portfolio_Daily", "Retail_Portfolio_Daily_Previous", "vars_behavioural_daily", "vars_behavioural_daily_Previous", "RM_Data_Daily_Run"]:
    print("\nDATASET:", dataset_name)
    try:
        loc = dataiku.Dataset(dataset_name).get_location_info()
        info = loc.get("info", {}) or {}
        print("Connection:", info.get("connectionName", "not provided"))
        print("Location:", loc.get("locationInfoType", loc.get("locationType", "not provided")))
    except Exception as exc:
        print("Location info unavailable:", type(exc).__name__, str(exc)[:180])

print("\nNO DATASETS READ / NO WRITES / NO JOBS RUN")
