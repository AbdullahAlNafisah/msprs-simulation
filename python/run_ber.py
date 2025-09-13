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
import msprs.system_msprs
import msprs.system_ask_2
import msprs.system_ask_4
import msprs.system_ldpc
from msprs.ldpc import make_ldpc

multiprocessing.set_start_method("spawn", force=True)
gc.collect()


def log_resource_usage(tag=""):
    vm = psutil.virtual_memory()
    logging.info(
        f"[{tag}] Memory Usage: {vm.used / (1024**2):.2f} MB / {vm.total / (1024**2):.2f} MB\n"
    )


def run_uncoded_msprs(args):
    # Optional: Set CPU affinity for each subprocess
    try:
        pid = os.getpid()
        cpu_id = pid % multiprocessing.cpu_count()
        os.sched_setaffinity(pid, {cpu_id})
    except Exception as e:
        logging.warning(f"Affinity setting failed: {e}")
    return msprs.system_msprs.uncoded_msprs_system_model(*args)


def run_coded_msprs(args):
    # Optional: Set CPU affinity for each subprocess
    try:
        pid = os.getpid()
        cpu_id = pid % multiprocessing.cpu_count()
        os.sched_setaffinity(pid, {cpu_id})
    except Exception as e:
        logging.warning(f"Affinity setting failed: {e}")
    return msprs.system_msprs.coded_msprs_system_model(*args)


def run_coded_ask_2(args):
    # Optional: Set CPU affinity for each subprocess
    try:
        pid = os.getpid()
        cpu_id = pid % multiprocessing.cpu_count()
        os.sched_setaffinity(pid, {cpu_id})
    except Exception as e:
        logging.warning(f"Affinity setting failed: {e}")
    return msprs.system_ask_2.coded_ask_2_system_model(*args)


def run_coded_ask_4_noGray(args):
    # Optional: Set CPU affinity for each subprocess
    try:
        pid = os.getpid()
        cpu_id = pid % multiprocessing.cpu_count()
        os.sched_setaffinity(pid, {cpu_id})
    except Exception as e:
        logging.warning(f"Affinity setting failed: {e}")
    return msprs.system_ask_4.coded_ask_4_system_model(*args)


def run_coded_ask_4_Gray(args):
    # Optional: Set CPU affinity for each subprocess
    try:
        pid = os.getpid()
        cpu_id = pid % multiprocessing.cpu_count()
        os.sched_setaffinity(pid, {cpu_id})
    except Exception as e:
        logging.warning(f"Affinity setting failed: {e}")
    return msprs.system_ask_4.coded_ask_4_system_model(*args)


def main(cfgs, n_cores, timestamp):

    L0 = cfgs["L0"]
    n_workers = 1
    chnl_params = msprs.channels.channel_set_up(cfgs["eb_no_range"], 1)
    code_params = msprs.channel_coding.precompute_code_params(
        cfgs["coder"], cfgs["src_bits"]
    )
    opt_coefficients = "constraint_equal_energy"
    mode_params = msprs.modulation.precompute_modulation_params(
        L0, code_params, opt_coefficients
    )
    noGray = True

    ldpc_params = {
        "n_code": 2 * cfgs["src_bits"],
        "d_v": 3,
        "d_c": 6,
    }
    H, G = make_ldpc(
        ldpc_params["n_code"],
        ldpc_params["d_v"],
        ldpc_params["d_c"],
        systematic=True,
        sparse=True,
    )
    ldpc_params["H"] = H
    ldpc_params["G"] = G
    ldpc_params["k"] = G.shape[1]
    ldpc_params["rate"] = ldpc_params["k"] / ldpc_params["n_code"]

    if cfgs["system"] == 0:
        output_dir = f"outputs/BER/uncoded_msprs/" + opt_coefficients + f"/L0_{L0}"
        file_name = f"EbNo_{cfgs['eb_no_range'][0]}-{cfgs['eb_no_range'][1]}.pkl"
    elif cfgs["system"] == 1:
        output_dir = f"outputs/BER/coded_msprs/" + opt_coefficients + f"/L0_{L0}"
        file_name = f"{cfgs['max_iters']}-Iters_EbNo_{cfgs['eb_no_range'][0]}-{cfgs['eb_no_range'][1]}.pkl"
        L0 = cfgs["L0"]
    elif cfgs["system"] == 2:
        output_dir = f"outputs/BER/coded_ask_2"
        file_name = f"EbNo_{cfgs['eb_no_range'][0]}-{cfgs['eb_no_range'][1]}.pkl"
    elif cfgs["system"] == 3:
        output_dir = f"outputs/BER/coded_ask_4_noGray"
        file_name = f"{cfgs['max_iters']}-Iters_EbNo_{cfgs['eb_no_range'][0]}-{cfgs['eb_no_range'][1]}.pkl"
        noGray = True
    elif cfgs["system"] == 4:
        output_dir = f"outputs/BER/coded_ask_4_Gray"
        file_name = f"{cfgs['max_iters']}-Iters_EbNo_{cfgs['eb_no_range'][0]}-{cfgs['eb_no_range'][1]}.pkl"
        noGray = False
    elif cfgs["system"] == 5:
        output_dir = f"outputs/BER/ldpc"
        file_name = f"{cfgs['max_iters']}-Iters_EbNo_{cfgs['eb_no_range'][0]}-{cfgs['eb_no_range'][1]}.pkl"
        cfgs["src_bits"] = cfgs["src_bits"] + 2
    else:
        print(chnl_params)
        print(code_params)
        print(mode_params)
        exit()

    logging.info(f"Available CPUs: {n_cores}, using {n_workers} workers")
    logging.info(f"Channel Eb/N0 range: {chnl_params['eb_no_db']}")
    log_resource_usage("Start")

    sim_elapsed = np.zeros(len(chnl_params["eb_no_db"]), dtype=np.float32)
    max_iters_valid = cfgs["max_iters"]
    ber_results = np.zeros(
        (max_iters_valid + 1, len(chnl_params["eb_no_db"])), dtype=np.float32
    )
    ers_cnt = np.zeros(
        (max_iters_valid + 1, len(chnl_params["eb_no_db"])), dtype=np.uint32
    )

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

            interleaver_indices = np.arange(
                code_params["coding_length"], dtype=np.int32
            )
            np.random.shuffle(interleaver_indices)

            if n_workers == 1:
                args = (
                    cfgs["src_bits"],
                    code_params["coding_length"],
                    code_params["polynomials"],
                    code_params["rate"],
                    code_params["memory"],
                    code_params["total_states"],
                    code_params["next_states"],
                    code_params["outputs"],
                    mode_params["modulation_length"],
                    L0,
                    mode_params["weighted_h0"],
                    mode_params["weighted_h1"],
                    mode_params["branch_labels"],
                    mode_params["memory"],
                    mode_params["total_states"],
                    mode_params["next_states"],
                    mode_params["branch_indices"],
                    interleaver_indices,
                    max_iters_valid,
                    chnl_params["noise_var"][i],
                    chnl_params["noise_std"][i],
                    noGray,
                )

                if cfgs["system"] == 0:
                    results_array = msprs.system_msprs.uncoded_msprs_system_model(*args)
                elif cfgs["system"] == 1:
                    results_array = msprs.system_msprs.coded_msprs_system_model(*args)
                elif cfgs["system"] == 2:
                    results_array = msprs.system_ask_2.coded_ask_2_system_model(*args)
                elif cfgs["system"] == 3:
                    results_array = msprs.system_ask_4.coded_ask_4_system_model(*args)
                elif cfgs["system"] == 4:
                    results_array = msprs.system_ask_4.coded_ask_4_system_model(*args)
                elif cfgs["system"] == 5:
                    results_array = msprs.system_ldpc.ldpc_ask_2_system_model(
                        cfgs["src_bits"],
                        code_params["coding_length"],
                        interleaver_indices,
                        max_iters_valid,
                        chnl_params["noise_var"][i],
                        chnl_params["noise_std"][i],
                        ldpc_params["G"],
                        ldpc_params["H"],
                    )
                else:
                    exit()

                pkts_tx += 1
                ers_cnt[: max_iters_valid + 1, i] += results_array

            else:

                args = [
                    (
                        cfgs["src_bits"],
                        code_params["coding_length"],
                        code_params["polynomials"],
                        code_params["rate"],
                        code_params["memory"],
                        code_params["total_states"],
                        code_params["next_states"],
                        code_params["outputs"],
                        mode_params["modulation_length"],
                        L0,
                        mode_params["weighted_h0"],
                        mode_params["weighted_h1"],
                        mode_params["branch_labels"],
                        mode_params["memory"],
                        mode_params["total_states"],
                        mode_params["next_states"],
                        mode_params["branch_indices"],
                        interleaver_indices,
                        max_iters_valid,
                        chnl_params["noise_var"][i],
                        chnl_params["noise_std"][i],
                        noGray,
                    )
                    for _ in range(n_workers)
                ]

                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    if cfgs["system"] == 0:
                        results = list(executor.map(run_uncoded_msprs, args))
                    elif cfgs["system"] == 1:
                        results = list(executor.map(run_coded_msprs, args))
                    elif cfgs["system"] == 2:
                        results = list(executor.map(run_coded_ask_2, args))
                    elif cfgs["system"] == 3:
                        results = list(executor.map(run_coded_ask_4_noGray, args))
                    elif cfgs["system"] == 4:
                        results = list(executor.map(run_coded_ask_4_Gray, args))
                    else:
                        exit()
                pkts_tx += n_workers
                results_array = np.vstack(results)  # type: ignore
                ers_cnt[: max_iters_valid + 1, i] += np.sum(results_array, axis=0)

            ber_est = ers_cnt[: max_iters_valid + 1, i] / (pkts_tx * cfgs["src_bits"])
            ber_window.append(ber_est[-1])
            if len(ber_window) == ber_window.maxlen:
                relative_change = np.abs(
                    (ber_window[-1] - ber_window[0]) / (ber_window[0] + 1e-12)
                )
                ber_avg = np.average(ber_window)

            logging.info(
                f"Pkts: {pkts_tx} | Bits: {pkts_tx * cfgs['src_bits']:.2e} | Convergence Error: {relative_change:.2e} | Avg. BER: {ber_avg:.2e}"
            )
            logging.info(f"BERs: {[f'{v:.2e}' for v in ber_est]}")
            logging.info(
                f"Errors: {[f'{e}' for e in ers_cnt[: max_iters_valid + 1, i]]}"
            )
            log_resource_usage(f"L0 {L0} | Eb/N0 {ebno_db:.2f} dB")

        sim_finish = datetime.now()
        sim_elapsed[i] = (sim_finish - sim_start).total_seconds()
        ber_results[:, i] = ers_cnt[:, i] / (pkts_tx * cfgs["src_bits"])
        logging.info(f"Completed Eb/N0 = {ebno_db:.2f} dB in {sim_elapsed[i]:.2f} sec")
        logging.info(
            f"Pkts: {pkts_tx} | Bits: {pkts_tx * cfgs['src_bits']:.2e} | Convergence Error: {relative_change:.2e} | Avg. BER: {ber_avg:.2e}"
        )

        while ber_results[max_iters_valid, i] < cfgs["ber_min"] and max_iters_valid > 0:
            max_iters_valid -= 1

        if ber_results[0, i] < cfgs["ber_min"]:
            break

    sim_results = {
        "cfgs": cfgs,
        "ldpc_params": ldpc_params,
        "code_params": code_params,
        "mode_params": mode_params,
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

    # "system":
    # 0: uncoded_msprs
    # 1: coded_msprs
    # 2: coded_ask_2
    # 3: coded_ask_4_noGray
    # 4: coded_ask_4_Gray
    # else: debug
    cfgs = {
        "system": 5,
        "eb_no_range": (1, 12, 1),
        "coder": {"K": 3, "octal_code": [0o5, 0o7]},
        "src_bits": 498,
        "max_iters": 30,
        "ber_window": 20,
        "ber_tol": 1e-3,
        "ber_max": 1e7,
        "ber_min": 1e-4,
    }

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if cfgs["system"] == 1 or cfgs["system"] == 2:
        parser = argparse.ArgumentParser(description="Run BER simulation with MSPRS.")
        parser.add_argument(
            "L0", type=int, help="Filter length h0 (integer between 2 and 10)"
        )
        args = parser.parse_args()

        cfgs["L0"] = args.L0

        if not (1 < cfgs["L0"] < 11):
            print("Error: Filter length (h0) out of range (2-10)")
            sys.exit(1)
        log_dir = f".logs/L0_{cfgs['L0']}"
    else:
        cfgs["L0"] = 2
        log_dir = f".logs/{timestamp}"

    # Delete previous cache files
    if os.path.exists("msprs/__pycache__"):
        shutil.rmtree("msprs/__pycache__")

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
