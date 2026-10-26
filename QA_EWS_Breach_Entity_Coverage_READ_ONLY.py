# -*- coding: utf-8 -*-

import dataiku
import pandas as pd


# Check that every entity included in the breach output also appears
# in the entity information sent to the EWS tool.

def load_entity_ids(dataset_name):
    frame = dataiku.Dataset(dataset_name).get_dataframe()
    if "ENTITY_ID" not in frame.columns:
        raise KeyError("{} is missing ENTITY_ID".format(dataset_name))

    ids = (
        frame["ENTITY_ID"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .dropna()
    )
    ids = ids.loc[~ids.isin(["", "nan", "None", "<NA>"])]
    return set(ids)


breach_ids = load_entity_ids("ews_ewi_breach_information")
entity_ids = load_entity_ids("ews_entity_static_data")

missing_entity_records = breach_ids - entity_ids
extra_entity_records = entity_ids - breach_ids

print("Breached entities:", len(breach_ids))
print("Entities in static output:", len(entity_ids))
print("Breached entities missing from static output:", len(missing_entity_records))
print("Static entities without a breach:", len(extra_entity_records))

if missing_entity_records:
    raise ValueError(
        "The breach and entity outputs are inconsistent: {} breached entities "
        "have no entity static record.".format(len(missing_entity_records))
    )

print("Breach / entity coverage check passed.")
