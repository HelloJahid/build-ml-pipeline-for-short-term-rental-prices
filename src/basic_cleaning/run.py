#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact
"""
import argparse
import logging

import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info("Downloading input artifact: %s", args.input_artifact)
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(artifact_local_path)
    logger.info("Raw data has %s rows and %s columns", *df.shape)

    # Basic cleaning
    df = df.drop_duplicates()
    df = df.dropna(subset=["price"])

    # Drop price outliers outside the [min_price, max_price] range agreed
    # with the stakeholders
    logger.info(
        "Dropping price outliers outside [%s, %s]", args.min_price, args.max_price
    )
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review from string to datetime
    logger.info("Converting last_review to datetime")
    df["last_review"] = pd.to_datetime(df["last_review"])

    logger.info("Cleaned data has %s rows and %s columns", *df.shape)

    # Save cleaned data
    df.to_csv("clean_sample.csv", index=False)

    # Upload the cleaned data to W&B
    logger.info("Uploading output artifact: %s", args.output_artifact)
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully qualified name of the raw input artifact in W&B (e.g. sample.csv:latest)",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the cleaned output artifact to create in W&B (e.g. clean_sample.csv)",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact, used to categorize it in W&B (e.g. clean_sample)",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="A brief description of the cleaned output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum accepted nightly price; rows below this value are dropped",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum accepted nightly price; rows above this value are dropped",
        required=True
    )

    args = parser.parse_args()

    go(args)