import os
import subprocess
import selectors
import time


def execute_script(script_name: str, work_dir: str = ".",
                   device: str = "0", timeout: int = 3600) -> str:
    """
    Execute a Python script and capture its output.

    Parameters
    ----------
    script_name : filename of the script (e.g. 'train_0.py')
    work_dir    : working directory to run the script in
    device      : CUDA_VISIBLE_DEVICES value
    timeout     : seconds before the process is killed (default 3600 = 1 hour)
                  Set to None to disable timeout (not recommended).

    Returns
    -------
    str : captured stdout (or stderr on failure), prefixed with a header line.
    """
    if not os.path.exists(os.path.join(work_dir, script_name)):
        raise Exception(f"The file {script_name} does not exist.")

    try:
        cmd = f"CUDA_VISIBLE_DEVICES={device} python -u {script_name}"
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            cwd=work_dir,
        )

        stdout_lines = []
        stderr_lines = []
        timed_out    = False

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        selector.register(process.stderr, selectors.EVENT_READ)

        start_time = time.time()

        while process.poll() is None and selector.get_map():
            # Check timeout
            if timeout is not None and (time.time() - start_time) > timeout:
                process.kill()
                timed_out = True
                print(f"[execute_script] TIMEOUT after {timeout}s — killed {script_name}")
                break

            events = selector.select(timeout=1)
            for key, _ in events:
                line = key.fileobj.readline()
                if key.fileobj == process.stdout:
                    print("STDOUT:", line, end=" ")
                    stdout_lines.append(line)
                else:
                    print("STDERR:", line, end=" ")
                    stderr_lines.append(line)

        # Drain any remaining output after the loop
        for line in process.stdout:
            print("STDOUT:", line, end=" ")
            stdout_lines.append(line)
        for line in process.stderr:
            print("STDERR:", line, end=" ")
            stderr_lines.append(line)

        selector.close()

        if timed_out:
            return (
                f"The script was killed after exceeding the timeout of {timeout}s.\n"
                + "".join(stderr_lines)
            )

        return_code = process.returncode
        if return_code != 0:
            observation = "".join(stderr_lines)
        else:
            observation = "".join(stdout_lines)
        if observation == "" and return_code == 0:
            observation = "".join(stderr_lines)

        return "The script has been executed. Here is the output:\n" + observation

    except Exception as e:
        print("++++", "Wrong!")
        raise Exception(
            f"Something went wrong in executing {script_name}: {e}. "
            "Please check if it is ready to be executed."
        )