# run_exit.py
"""
EXIT chart analysis
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


def main(cfgs, n_cores, timestamp):

    L0 = cfgs["L0"]
    n_workers = 1
    chnl_params = msprs.channels.channel_set_up(cfgs["eb_no_range"], 1 / 2)
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
        output_dir = f"outputs/EXIT/msprs/" + opt_coefficients + f"/L0_{L0}"
    elif cfgs["system"] == 1:
        output_dir = f"outputs/EXIT/ask_4_noGray"
        noGray = True
    elif cfgs["system"] == 2:
        output_dir = f"outputs/EXIT/ask_4_Gray"
        noGray = False
    elif cfgs["system"] == 3:
        output_dir = f"outputs/EXIT/ldpc"
        cfgs["src_bits"] = cfgs["src_bits"] + 2
    else:
        print(chnl_params)
        print(code_params)
        print(mode_params)
        exit()
    file_name = f"EbNo_{cfgs['eb_no_range'][0]}-{cfgs['eb_no_range'][1]}.pkl"

    logging.info(f"Available CPUs: {n_cores}, using {n_workers} workers")
    logging.info(f"Channel Eb/N0 range: {chnl_params['eb_no_db']}")
    log_resource_usage("Start")

    # A priori mutual information points
    ia_points = cfgs["ia_points"]
    # IA = 0.999 * np.arange(0.001, 1 + 1 / ia_points, (1 + 1 / ia_points) / ia_points)
    IA = 0.999 * np.arange(0.1, 1 + 1 / ia_points, (1 + 1 / ia_points) / ia_points)

    # EXIT Chart for Coder
    logging.info(f"Analyzing channel decoder...")
    coder_ie_hist, coder_ie_avg = msprs.system_ask_2.coder_exit_analysis(
        IA,
        ia_points,
        cfgs["src_bits"],
        code_params["coding_length"],
        code_params["polynomials"],
        code_params["rate"],
        code_params["memory"],
        code_params["total_states"],
        code_params["next_states"],
        code_params["outputs"],
    )

    # EXIT Chart for Modulator with different optimizations
    sim_elapsed = np.zeros(len(chnl_params["eb_no_db"]), dtype=np.float32)
    mod_ie_hist = np.zeros((len(chnl_params["eb_no_db"]), ia_points), dtype=np.float32)
    for i, ebno_db in enumerate(chnl_params["eb_no_db"]):
        logging.info(f"Processing Eb/N0 = {ebno_db:.2f} dB")
        sim_start = datetime.now()

        args = (
            IA,
            ia_points,
            cfgs["src_bits"],
            code_params["coding_length"],
            code_params["polynomials"],
            mode_params["modulation_length"],
            L0,
            mode_params["weighted_h0"],
            mode_params["weighted_h1"],
            mode_params["branch_labels"],
            mode_params["memory"],
            mode_params["total_states"],
            mode_params["next_states"],
            mode_params["branch_indices"],
            chnl_params["noise_var"][i],
            chnl_params["noise_std"][i],
            noGray,
        )

        if cfgs["system"] == 0:
            mod_ie_hist[i] = msprs.system_msprs.modulator_exit_analysis(*args)
        elif cfgs["system"] == 1:
            mod_ie_hist[i] = msprs.system_ask_4.exit_analysis_4_ask(*args)
        elif cfgs["system"] == 2:
            mod_ie_hist[i] = msprs.system_ask_4.exit_analysis_4_ask(*args)
        elif cfgs["system"] == 3:
            mod_ie_hist[i] = msprs.system_ldpc.modulator_exit_analysis(
                IA,
                ia_points,
                cfgs["src_bits"],
                code_params["coding_length"],
                chnl_params["noise_var"][i],
                chnl_params["noise_std"][i],
                ldpc_params["G"],
                ldpc_params["H"],
            )
        else:
            exit()

        sim_finish = datetime.now()
        sim_elapsed[i] = (sim_finish - sim_start).total_seconds()
        logging.info(f"IE: {mod_ie_hist[i]}")
        logging.info(f"Completed Eb/N0 = {ebno_db:.2f} dB in {sim_elapsed[i]:.2f} sec")

    sim_results = {
        "cfgs": cfgs,
        "ldpc_params": ldpc_params,
        "code_params": code_params,
        "mode_params": mode_params,
        "chnl_params": chnl_params,
        "coder_IE_hist": coder_ie_hist,
        "mod_ie_hist": mod_ie_hist,
        "ia_points": ia_points,
        "IA": IA,
        "metadata": {
            "timestamp": timestamp,
            "sim_elapsed": sim_elapsed,
        },
    }

    total_time = np.sum(sim_elapsed)
    logging.info(f"Total simulation time: {total_time:.2f} seconds")
    logging.info(f"Final EXIT-code matrix shape: {coder_ie_hist.shape}")
    logging.info(f"Final EXIT-modulator matrix shape: {mod_ie_hist.shape}")

    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/" + file_name

    with open(output_file, "wb") as f:
        pickle.dump(sim_results, f)

    logging.info(f"Simulation results saved to: {output_file}")


if __name__ == "__main__":

    # "system":
    # 0: msprs
    # 1: ask_4_noGray
    # 2: ask_4_Gray
    # else: debug
    cfgs = {
        "system": 3,
        "eb_no_range": (1, 12, 1),
        "coder": {"K": 3, "octal_code": [0o5, 0o7]},
        "src_bits": 498,
        "max_iters": 16,
        "ia_points": 20,
    }

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if cfgs["system"] == 0:
        parser = argparse.ArgumentParser(description="Run EXIT simulation with MSPRS.")
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
    log_file = os.path.join(log_dir, f"run_exit_{timestamp}.log")

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
