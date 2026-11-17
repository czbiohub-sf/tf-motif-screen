import argparse

from af3tfm.dataset import DATA_DIR
from af3tfm.jobdb import Database


def main():
    parser = argparse.ArgumentParser(description="Build or update the job database.")
    parser.add_argument(
        "--dataset", default="tfdb4_rep", help="Dataset name (default: cisbp)"
    )
    parser.add_argument(
        "--run", default="hero", help="Run name (default: motif-nonmotif)"
    )
    parser.add_argument(
        "--direc", default="fold_all", help="Output subdirectory suffix (default: fold)"
    )
    args = parser.parse_args()

    db = Database(
        job_path=f"{DATA_DIR}/jobs/{args.run}",
        output_path=f"{DATA_DIR}/jobs/{args.run}/output_{args.direc}",
        dataset=args.dataset,
    )

    db.process_runs()


if __name__ == "__main__":
    main()
