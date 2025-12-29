import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Board Game recommendation system")

    parser.add_argument("--ingestion", required=False, default=False)
