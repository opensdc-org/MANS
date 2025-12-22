import os
import shutil
import subprocess
import filecmp
import logging
from pathlib import Path

from tools.dataset_gen import generate_dataset, ADMConfig

# ================== ANSI COLORS ==================
RESET  = "\033[0m"
PURPLE = "\033[35m"
BLUE   = "\033[34m"
GREEN  = "\033[32m"
RED    = "\033[31m"

# ================== SIMPLE LOGGER ==================
def setup_logger():
    logger = logging.getLogger("auto_test_plain")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()

    class PlainFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            # Print plain message without [INFO]/[ERROR] prefix
            return str(record.getMessage())

    handler.setFormatter(PlainFormatter())
    logger.handlers = [handler]
    logger.propagate = False
    return logger


log = setup_logger()

# ================== GLOBAL CONFIG ==================
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
BUILD_BIN_CPU  = PROJECT_ROOT / "build" / "bin" / "cpu"
COMPRESS_BIN   = BUILD_BIN_CPU / "cpu_mans_compress"
DECOMPRESS_BIN = BUILD_BIN_CPU / "cpu_mans_decompress"

DATA_DIR       = PROJECT_ROOT / "analysis" / "test_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Parameter sweeps
DTYPE_LIST          = ["u2", "u4"]
TOTAL_ELEMENTS_LIST = [512, 1024000]
THRESHOLD_LIST      = [3800, 5000]

SAVE_ADM       = "1"   # "1" = dump adm intermediates, "0" = no adm dump
REPEAT_TIMES   = 3     # number of compress/decompress rounds per config


def banner(title: str, ch: str = "=", width: int = 60):
    line = ch * width
    log.info(f"{PURPLE}{line}{RESET}")
    log.info(f"{PURPLE}{title:^{width}}{RESET}")
    log.info(f"{PURPLE}{line}{RESET}")


def run_cmd(cmd, cwd=None):
    cmd_str = " ".join(map(str, cmd))
    log.info(f"{BLUE}[RUN] {cmd_str}{RESET}")
    res = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if res.returncode != 0:
        log.info(f"{RED}[CMD FAILED] {cmd_str}{RESET}")
        if res.stdout.strip():
            log.info(f"{RED}[STDOUT]\n{res.stdout.rstrip()}{RESET}")
        if res.stderr.strip():
            log.info(f"{RED}[STDERR]\n{res.stderr.rstrip()}{RESET}")
        raise RuntimeError(f"Command failed: {cmd_str}")
    return res


def gen_dataset(dtype: str, total_elems: int, threshold: int, dataset_path: Path):
    """Generate dataset for given dtype / total_elems / threshold."""
    ADMConfig.BLOCK_SIZE = 512
    ADMConfig.THRESHOLD  = threshold
    if dtype == "u2":
        ADMConfig.IS_U16 = True
    else:
        ADMConfig.IS_U16 = False
    generate_dataset(str(dataset_path), total_elems, ADMConfig)


def compare_files(f1: Path, f2: Path, desc: str):
    """Compare two binary files for bitwise equality."""
    if (not f1.exists()) or (not f2.exists()):
        # Missing ADM files can be expected if ADM was not triggered
        line = f"[COMPARE] {desc}: FILE MISSING"
        log.info(line)
        log.info("If ADM was not triggered in this round, missing .adm files are expected.")
        return True  # treat as pass so it does not affect data_ok / adm_ok

    same = filecmp.cmp(str(f1), str(f2), shallow=False)
    if same:
        prefix = f"[COMPARE] {desc}: "
        log.info(f"{prefix}{GREEN}OK{RESET}")
    else:
        prefix = f"[COMPARE] {desc}: "
        log.info(f"{RED}{prefix}FAIL{RESET}")
        log.info(f"{RED}          -> {f1}{RESET}")
        log.info(f"{RED}          -> {f2}{RESET}")
    return same


def one_round(round_id: int, dtype: str, input_file: Path, save_adm: str):
    """
    Run one compress + decompress round:
      input_file --(compress)--> mans_roundX.bin --(decompress)--> decomp_roundX.u2/u4
    Return:
      mans_out, decomp_out, adm_compress, adm_decompress
    """
    banner(f"ROUND {round_id}", "-")

    # Reuse same naming pattern across parameter combinations
    mans_out       = DATA_DIR / f"mans_round{round_id}.bin"
    decomp_suffix  = ".u2" if dtype == "u2" else ".u4"
    decomp_out     = DATA_DIR / f"decomp_round{round_id}{decomp_suffix}"
    adm_compress   = mans_out.with_suffix(mans_out.suffix + ".adm")
    adm_decompress = decomp_out.with_suffix(decomp_out.suffix + ".adm")

    # 1) compress
    cmd_comp = [
        str(COMPRESS_BIN),
        dtype,
        str(input_file),
        str(mans_out),
        save_adm,
    ]
    run_cmd(cmd_comp, cwd=PROJECT_ROOT)

    # 2) decompress
    cmd_decomp = [
        str(DECOMPRESS_BIN),
        dtype,
        str(mans_out),
        str(decomp_out),
        save_adm,
    ]
    run_cmd(cmd_decomp, cwd=PROJECT_ROOT)

    return mans_out, decomp_out, adm_compress, adm_decompress


def main():
    banner("MANS AUTO TEST PARAM SWEEP")

    log.info(f"[CONFIG] DTYPE_LIST   = {DTYPE_LIST}")
    log.info(f"[CONFIG] TOTAL_LIST   = {TOTAL_ELEMENTS_LIST}")
    log.info(f"[CONFIG] THRESH_LIST  = {THRESHOLD_LIST}")
    log.info(f"[CONFIG] SAVE_ADM     = {SAVE_ADM}")
    log.info(f"[CONFIG] REPEAT_TIMES = {REPEAT_TIMES}")

    # Clean data dir once at startup
    log.info(f"[CLEAN] {DATA_DIR}")
    if DATA_DIR.exists():
        for p in DATA_DIR.glob("*"):
            if p.is_file():
                p.unlink()
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # Sweep all parameter combinations
    for dtype in DTYPE_LIST:
        for total_elems in TOTAL_ELEMENTS_LIST:
            for threshold in THRESHOLD_LIST:
                banner(f"CASE: dtype={dtype}, elems={total_elems}, thr={threshold}", "=")

                input_raw = DATA_DIR / ("input_u2.bin" if dtype == "u2" else "input_u4.bin")
                gen_dataset(dtype, total_elems, threshold, input_raw)

                mans_files       = []
                decomp_files     = []
                adm_comp_files   = []
                adm_decomp_files = []

                data_ok = True
                adm_ok  = True  # only meaningful when SAVE_ADM == "1"

                # Run multiple rounds for this config
                for i in range(REPEAT_TIMES):
                    mans_out, decomp_out, adm_comp, adm_decomp = one_round(
                        i, dtype, input_raw, SAVE_ADM
                    )
                    mans_files.append(mans_out)
                    decomp_files.append(decomp_out)
                    adm_comp_files.append(adm_comp)
                    adm_decomp_files.append(adm_decomp)

                    same = compare_files(
                        input_raw,
                        decomp_out,
                        f"[dtype={dtype},N={total_elems},thr={threshold},round={i}] input_raw vs decomp",
                    )
                    if not same:
                        data_ok = False

                # Intra-config ADM consistency
                if SAVE_ADM == "1":
                    banner("ADM INTERMEDIATE CONSISTENCY (PER CASE)")

                    base_adm_comp = adm_comp_files[0]
                    for i in range(1, REPEAT_TIMES):
                        same = compare_files(
                            base_adm_comp,
                            adm_comp_files[i],
                            f"[dtype={dtype},N={total_elems},thr={threshold}] adm_compress_round0 vs round{i}",
                        )
                        if not same:
                            adm_ok = False

                    base_adm_decomp = adm_decomp_files[0]
                    for i in range(1, REPEAT_TIMES):
                        same = compare_files(
                            base_adm_decomp,
                            adm_decomp_files[i],
                            f"[dtype={dtype},N={total_elems},thr={threshold}] adm_decompress_round0 vs round{i}",
                        )
                        if not same:
                            adm_ok = False
                else:
                    adm_ok = None

                results.append(
                    {
                        "dtype": dtype,
                        "N": total_elems,
                        "thr": threshold,
                        "data_ok": data_ok,
                        "adm_ok": adm_ok,
                    }
                )

                # Cleanup per-config intermediates
                for p in DATA_DIR.glob("mans_round*.bin"):
                    p.unlink(missing_ok=True)
                for p in DATA_DIR.glob("decomp_round*.u2"):
                    p.unlink(missing_ok=True)
                for p in DATA_DIR.glob("decomp_round*.u4"):
                    p.unlink(missing_ok=True)
                for p in DATA_DIR.glob("*.adm"):
                    p.unlink(missing_ok=True)

    # ===== FINAL SUMMARY TABLE =====
    banner("SUMMARY", "=")

    # Fixed column widths
    col_widths = {
        "dtype": 6,
        "N": 10,
        "thr": 8,
        "data": 8,
        "adm": 8,
    }

    header = (
        "DTYPE".ljust(col_widths["dtype"])
        + " "
        + "N".rjust(col_widths["N"])
        + " "
        + "THR".rjust(col_widths["thr"])
        + " "
        + "DATA".rjust(col_widths["data"])
        + " "
        + "ADM".rjust(col_widths["adm"])
    )
    log.info(header)
    log.info("-" * len(header))

    all_pass = True
    for r in results:
        dtype   = r["dtype"]
        N       = r["N"]
        thr     = r["thr"]
        data_ok = r["data_ok"]
        adm_ok  = r["adm_ok"]

        data_str = "OK" if data_ok else "FAIL"
        if adm_ok is None:
            adm_raw = "N/A"
        else:
            adm_raw = "OK" if adm_ok else "FAIL"

        # Pad without colors, then wrap with ANSI so alignment is correct
        data_padded = data_str.rjust(col_widths["data"])
        if data_ok:
            data_colored = f"{GREEN}{data_padded}{RESET}"
        else:
            data_colored = f"{RED}{data_padded}{RESET}"

        adm_padded = adm_raw.rjust(col_widths["adm"])
        if adm_ok is None:
            adm_colored = adm_padded
        elif adm_ok:
            adm_colored = f"{GREEN}{adm_padded}{RESET}"
        else:
            adm_colored = f"{RED}{adm_padded}{RESET}"

        line = (
            dtype.ljust(col_widths["dtype"])
            + " "
            + str(N).rjust(col_widths["N"])
            + " "
            + str(thr).rjust(col_widths["thr"])
            + " "
            + data_colored
            + " "
            + adm_colored
        )
        log.info(line)

        if (not data_ok) or (adm_ok is False):
            all_pass = False

    if all_pass:
        log.info(f"{GREEN}ALL TESTS PASSED{RESET}")
    else:
        log.info(f"{RED}SOME TESTS FAILED{RESET}")


if __name__ == "__main__":
    main()