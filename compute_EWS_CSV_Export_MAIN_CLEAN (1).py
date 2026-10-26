# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import dataiku
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Export settings
# The four datasets are written to the same managed folder in batches.

EXPORT_FOLDER_ID = "tFtFJ7YO"

TIMEZONE = "Europe/Luxembourg"

CHUNK_SIZE = 50000

DATASETS_TO_EXPORT = [
    "ews_ewi_breach_information",
    "ews_ewi_historic_information",
    "ews_exposure_static_data",
    "ews_entity_static_data",
]


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Use Luxembourg local time to give all four files the same export timestamp.

run_timestamp = datetime.now(
    ZoneInfo(TIMEZONE)
).strftime("%Y%m%d_%H%M%S")

print(
    "Export timestamp: {}".format(
        run_timestamp
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Open the managed folder where the CSV files are stored.

export_folder = dataiku.Folder(
    EXPORT_FOLDER_ID
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Convert the text placeholder "MISS" into an empty CSV value.

def normalize_export_values(frame):

    frame = frame.copy()

    text_columns = frame.select_dtypes(
        include=[
            "object",
            "string",
        ]
    ).columns

    for column in text_columns:

        miss_mask = (
            frame[column]
            .astype("string")
            .str.strip()
            .str.upper()
            .eq("MISS")
        )

        frame.loc[
            miss_mask,
            column,
        ] = pd.NA

    return frame


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write a dataset in batches so large exports do not need to fit in memory.

def export_dataset_to_csv(
    dataset_name,
    folder,
    timestamp,
):

    dataset = dataiku.Dataset(
        dataset_name
    )

    file_name = "{}_{}.csv".format(
        dataset_name,
        timestamp,
    )

    print("")
    print("=" * 80)
    print(
        "Exporting dataset: {}".format(
            dataset_name
        )
    )
    print(
        "File name: {}".format(
            file_name
        )
    )

    total_rows = 0
    total_miss_replaced = 0
    first_chunk = True

    # Open a new file in the managed folder.
    with folder.get_writer(
        file_name
    ) as writer:

        for chunk in dataset.iter_dataframes(
            chunksize=CHUNK_SIZE
        ):

            text_columns = chunk.select_dtypes(
                include=[
                    "object",
                    "string",
                ]
            ).columns

            miss_count = 0

            for column in text_columns:
                miss_count += int(
                    chunk[column]
                    .astype("string")
                    .str.strip()
                    .str.upper()
                    .eq("MISS")
                    .sum()
                )

            chunk = normalize_export_values(
                chunk
            )

            total_miss_replaced += miss_count

            # Write headers once and leave missing values empty.
            csv_content = chunk.to_csv(
                index=False,
                header=first_chunk,
                sep=",",
                na_rep="",
                lineterminator="\n",
            )

            writer.write(
                csv_content.encode(
                    "utf-8"
                )
            )

            total_rows += len(
                chunk
            )

            first_chunk = False

            print(
                "Rows exported so far: {:,}".format(
                    total_rows
                )
            )

        # Keep the column headers even when there are no records to export.
        if first_chunk:

            schema = dataset.read_schema()

            columns = [
                column["name"]
                for column in schema
            ]

            empty_df = pd.DataFrame(
                columns=columns
            )

            csv_content = empty_df.to_csv(
                index=False,
                header=True,
                sep=",",
                na_rep="",
                lineterminator="\n",
            )

            writer.write(
                csv_content.encode(
                    "utf-8"
                )
            )

    print(
        "MISS values replaced with NULL: {:,}".format(
            total_miss_replaced
        )
    )

    print(
        "Finished: {} | {:,} rows".format(
            file_name,
            total_rows,
        )
    )

    # Keep the file details for the final export summary.
    return {
        "dataset": dataset_name,
        "file_name": file_name,
        "rows": total_rows,
        "miss_replaced": total_miss_replaced,
    }


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Export the four datasets using the same timestamp for easy identification.

export_results = []

for dataset_name in DATASETS_TO_EXPORT:

    result = export_dataset_to_csv(
        dataset_name=dataset_name,
        folder=export_folder,
        timestamp=run_timestamp,
    )

    export_results.append(
        result
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Show the files created, row counts and missing-value replacements.

export_summary = pd.DataFrame(
    export_results
)

print("")
print("=" * 80)
print("EWS CSV EXPORT COMPLETED")
print("=" * 80)

print(
    export_summary.to_string(
        index=False
    )
)

print("")
print(
    "Managed Folder: Credit_EWS ({})".format(
        EXPORT_FOLDER_ID
    )
)

print(
    "Timestamp: {}".format(
        run_timestamp
    )
)

print(
    "Files created: {}".format(
        len(export_results)
    )
)

print(
    "Total MISS values replaced: {:,}".format(
        export_summary[
            "miss_replaced"
        ].sum()
    )
)
