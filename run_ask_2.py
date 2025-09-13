# run_ber.py
"""
BER simulation
"""

import numpy as np
import argparse
import os
import pickle
import sys
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import gc
import logging
import psutil
import shutil
from collections import deque

import msprs
import msprs.system_ask_2
from msprs.ldpc import make_ldpc

multiprocessing.set_start_method("spawn", force=True)
gc.collect()


def log_resource_usage(tag=""):
    vm = psutil.virtual_memory()
    logging.info(
        f"[{tag}] Memory Usage: {vm.used / (1024**2):.2f} MB / {vm.total / (1024**2):.2f} MB\n"
    )


def run_ask_2(args):
    # Optional: Set CPU affinity for each subprocess
    try:
        pid = os.getpid()
        cpu_id = pid % multiprocessing.cpu_count()
        os.sched_setaffinity(pid, {cpu_id})
    except Exception as e:
        logging.warning(f"Affinity setting failed: {e}")
    return msprs.system_ask_2.ask_2_system_model(*args)


def main(cfgs, n_cores, timestamp):

    n_workers = 1
    chnl_params = msprs.channels.channel_set_up(cfgs["eb_no_range"], 1)

    output_dir = f"outputs/BER/ask_2"
    file_name = f"EbNo_{cfgs['eb_no_range'][0]}-{cfgs['eb_no_range'][1]}.pkl"

    logging.info(f"Available CPUs: {n_cores}, using {n_workers} workers")
    logging.info(f"Channel Eb/N0 range: {chnl_params['eb_no_db']}")
    log_resource_usage("Start")

    sim_elapsed = np.zeros(len(chnl_params["eb_no_db"]), dtype=np.float32)
    ber_results = np.zeros((len(chnl_params["eb_no_db"])), dtype=np.float32)
    ers_cnt = np.zeros((len(chnl_params["eb_no_db"])), dtype=np.uint32)

    for i, ebno_db in enumerate(chnl_params["eb_no_db"]):
        logging.info(f"Processing Eb/N0 = {ebno_db:.2f} dB")
        pkts_tx = 0
        sim_start = datetime.now()
        relative_change = np.inf

        ber_window = deque(maxlen=cfgs["ber_window"])
        ber_avg = np.inf

        while (
            relative_change > cfgs["ber_tol"]
            and pkts_tx * cfgs["src_bits"] < cfgs["ber_max"]
            and ber_avg > cfgs["ber_min"]
        ):

            if n_workers == 1:
                args = (
                    cfgs["src_bits"],
                    chnl_params["noise_std"][i],
                )

                results_array = msprs.system_ask_2.ask_2_system_model(*args)
                pkts_tx += 1
                ers_cnt[i] += results_array

            else:

                args = [
                    (
                        cfgs["src_bits"],
                        chnl_params["noise_std"][i],
                    )
                    for _ in range(n_workers)
                ]

                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    results = list(executor.map(run_ask_2, args))
                pkts_tx += n_workers
                results_array = np.vstack(results)  # type: ignore
                ers_cnt[i] += np.sum(results_array, axis=0)

            ber_est = ers_cnt[i] / (pkts_tx * cfgs["src_bits"])
            ber_window.append(ber_est)
            if len(ber_window) == ber_window.maxlen:
                relative_change = np.abs(
                    (ber_window[-1] - ber_window[0]) / (ber_window[0] + 1e-12)
                )
                ber_avg = np.average(ber_window)

            logging.info(
                f"Pkts: {pkts_tx} | Bits: {pkts_tx * cfgs['src_bits']:.2e} | Convergence Error: {relative_change:.2e} | Avg. BER: {ber_avg:.2e}"
            )
            logging.info(f"BERs: {ber_est}")
            logging.info(f"Errors: {ers_cnt[i]}")
            log_resource_usage(f"Eb/N0 {ebno_db:.2f} dB")

        sim_finish = datetime.now()
        sim_elapsed[i] = (sim_finish - sim_start).total_seconds()
        ber_results[i] = ers_cnt[i] / (pkts_tx * cfgs["src_bits"])
        logging.info(f"Completed Eb/N0 = {ebno_db:.2f} dB in {sim_elapsed[i]:.2f} sec")
        logging.info(
            f"Pkts: {pkts_tx} | Bits: {pkts_tx * cfgs['src_bits']:.2e} | Convergence Error: {relative_change:.2e} | Avg. BER: {ber_avg:.2e}"
        )
        if ber_results[i] < cfgs["ber_min"]:
            break

    sim_results = {
        "cfgs": cfgs,
        "chnl_params": chnl_params,
        "ber_results": ber_results,
        "ers_cnt": ers_cnt,
        "metadata": {
            "timestamp": timestamp,
            "sim_elapsed": sim_elapsed,
        },
    }

    total_time = np.sum(sim_elapsed)
    logging.info(f"Total simulation time: {total_time:.2f} seconds")
    logging.info(f"Final BER matrix shape: {ber_results.shape}")

    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/" + file_name

    with open(output_file, "wb") as f:
        pickle.dump(sim_results, f)

    logging.info(f"Simulation results saved to: {output_file}")


if __name__ == "__main__":

    cfgs = {
        "src_bits": 4998,
        "ber_window": 20,
        "ber_tol": 1e-3,
        "ber_max": 1e7,
        "ber_min": 1e-4,
    }

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    parser = argparse.ArgumentParser(description="Run BER simulation.")
    parser.add_argument("--eb_no_min", type=int, default=1, help="minimum Eb/No")
    parser.add_argument("--eb_no_max", type=int, default=10, help="maximum Eb/No")
    parser.add_argument("--eb_no_step", type=int, default=1, help="Eb/No step size")
    args = parser.parse_args()

    cfgs["eb_no_range"] = (args.eb_no_min, args.eb_no_max, args.eb_no_step)
    print(cfgs["eb_no_range"])

    log_dir = f".logs/{timestamp}"

    n_cores = multiprocessing.cpu_count()

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"run_ber_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(console_handler)

    main(cfgs, n_cores, timestamp)
