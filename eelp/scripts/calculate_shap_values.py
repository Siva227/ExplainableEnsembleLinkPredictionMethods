import logging
from logging.handlers import TimedRotatingFileHandler
from multiprocessing import Pool, cpu_count
from pathlib import Path

import click
import numpy as np

from eelp.utils.parallel_utils import chunk, process_shap


@click.command()
@click.option(
    "--data-path", "data_path", required=True, type=click.Path(exists=True, dir_okay=True)
)
@click.option("--num-procs", type=click.INT, default=-1)
def main(data_path, num_procs):
    procs = num_procs if num_procs > 0 else cpu_count()
    data_path = Path(data_path).resolve()

    output_paths = [i for i in data_path.glob("*") if i.is_dir()]
    num_graphs_per_proc = len(output_paths) / float(procs)
    num_graphs_per_proc = int(np.ceil(num_graphs_per_proc))
    chunked_paths = chunk(output_paths, num_graphs_per_proc)
    payloads = []

    for idx, graph_set in enumerate(chunked_paths):
        data = {"id": idx, "graph_paths": graph_set}
        payloads.append(data)

    # Now we use multiprocessing
    logger.info("Launching pool using {} processes...".format(procs))
    pool = Pool(processes=procs)
    pool.map(process_shap, payloads)
    pool.close()
    pool.join()
    logger.info("Multiprocessing complete")


if __name__ == "__main__":
    # Finding project home
    project_dir = Path(__file__).resolve().parents[2]
    # load_dotenv(find_dotenv())
    # Setting up logging configuration for the module

    # Create a custom logger
    logger = logging.getLogger()
    # Setting root logger level to the lowest so all info messages can be logged to the file
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = TimedRotatingFileHandler(
        project_dir.joinpath("logs", "shap_process.log"),
        when="midnight",
    )
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    main()
