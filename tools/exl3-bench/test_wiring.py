#!/usr/bin/env python3
"""Executable wiring tests: real scripts, argv sinks, no model and no production path.

Each test builds a sandbox EXP root, replaces every entry point with a recording stub, and runs the
actual shell driver. String-presence tests are not used: the assertions are on captured argv, exit
codes, and whether a process group was really reaped.
"""
import os, shutil, subprocess, sys, time, signal

HERE = os.path.dirname(os.path.abspath(__file__))
W3 = os.path.join(HERE, "window3.sh")
RP = os.path.join(HERE, "run_probe.sh")
CR = os.path.join(HERE, "cleanup-run.sh")


def sandbox(tmp_path):
    exp = tmp_path / "exp"
    (exp / "raw").mkdir(parents=True)
    (exp / "ext-cache-clean" / "exllamav3_ext").mkdir(parents=True)
    (exp / "ext-cache-instr" / "exllamav3_ext").mkdir(parents=True)
    for v in ("clean", "instr"):
        (exp / f"ext-cache-{v}" / "exllamav3_ext" / "exllamav3_ext.so").write_bytes(b"stub")
    for name in ("window3.sh", "run_probe.sh", "cleanup-run.sh"):
        shutil.copy(os.path.join(HERE, name), exp / name)
        os.chmod(exp / name, 0o755)
    return exp


def stub(path, marker, rc=0):
    path.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > {marker}\nexit {rc}\n')
    os.chmod(path, 0o755)


def env(exp, **kw):
    e = dict(os.environ, EXP_ROOT_DIR=str(exp), X3_PREFLIGHT_WAIT_S="1")
    e.pop("X3_MOCK", None)
    e.update({k: str(v) for k, v in kw.items()})
    return e


def test_run_probe_argv_sink(tmp_path):
    exp = sandbox(tmp_path)
    argvlog = tmp_path / "argv.log"
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    stub(venv / "python", argvlog)
    envscript = tmp_path / "env.sh"
    envscript.write_text("#!/usr/bin/env bash\ntrue\n")
    r = subprocess.run([str(exp / "run_probe.sh"), str(tmp_path / "probe.json"),
                        str(tmp_path / "tensors"), "18"],
                       env=env(exp, ENV_SCRIPT=str(envscript), VENV=str(tmp_path / "venv"),
                               FORK=str(tmp_path / "exl3-v164-rocm10-port"), TORCH_EXTENSIONS_DIR=str(exp / "ext-cache-instr"),
                               PROBE_ENTRY=str(tmp_path / "probe_mtp_causal.py")),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    argv = argvlog.read_text().split()
    assert argv[0].endswith("probe_mtp_causal.py")
    for flag in ("--index", "--out", "--tensor-dir", "--ctx", "--prompt"):
        assert flag in argv, argv
    assert "--max-new" not in argv
    assert argv[argv.index("--index") + 1] == "18"


def test_run_probe_refuses_cache_tree_mismatch(tmp_path):
    exp = sandbox(tmp_path)
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    stub(venv / "python", tmp_path / "argv.log")
    envscript = tmp_path / "env.sh"
    envscript.write_text("true\n")
    r = subprocess.run([str(exp / "run_probe.sh"), str(tmp_path / "p.json")],
                       env=env(exp, ENV_SCRIPT=str(envscript), VENV=str(tmp_path / "venv"),
                               FORK="/nonexistent/tree", TORCH_EXTENSIONS_DIR=str(exp / "ext-cache-instr")),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 8 and "cache/tree mismatch" in r.stderr
    assert not (tmp_path / "argv.log").exists()


def test_window_selfcheck_is_passed_explicitly_and_aborts_before_any_arm(tmp_path):
    exp = sandbox(tmp_path)
    argvlog = tmp_path / "argv.log"
    stub(exp / "run_arm3.sh", argvlog, rc=1)          # selfcheck failure must stop the window
    stub(exp / "cleanup-run.sh", tmp_path / "cleanup.log")
    r = subprocess.run([str(exp / "window3.sh"), "--execute"], env=env(exp),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 3, (r.returncode, r.stdout[-500:])   # ABORT: selfcheck failed
    argv = argvlog.read_text().split()
    assert argv[0] == "selfcheck" and argv[5] == "selfcheck", argv   # sixth arg, not defaulted


def test_mocks_never_reach_a_real_entrypoint(tmp_path):
    exp = sandbox(tmp_path)
    for name in ("run_arm3.sh", "run_probe.sh"):
        marker = tmp_path / f"{name}.called"
        (exp / name).write_text(f'#!/usr/bin/env bash\ntouch {marker}\nexit 0\n')
        os.chmod(exp / name, 0o755)
    stub(exp / "cleanup-run.sh", tmp_path / "cleanup.log")
    r = subprocess.run([str(exp / "window3.sh"), "--execute"], env=env(exp, X3_MOCK="ok"),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, (r.returncode, r.stdout[-500:])
    assert not (tmp_path / "run_arm3.sh.called").exists(), "mock mode reached run_arm3.sh"
    assert not (tmp_path / "run_probe.sh.called").exists(), "mock mode reached run_probe.sh"


def test_probe_shaped_mock_is_reaped_by_run_scoped_cleanup(tmp_path):
    exp = sandbox(tmp_path)
    shutil.copy(CR, exp / "cleanup-run.sh")
    os.chmod(exp / "cleanup-run.sh", 0o755)
    r = subprocess.run([str(exp / "window3.sh"), "--execute"],
                       env=env(exp, X3_MOCK="probe-shape", X3_MOCK_SLEEP="30"),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, (r.returncode, r.stdout[-800:])
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=60).stdout
    assert "probe_mtp_causal.py --index 18" not in out, "probe-shaped group survived cleanup"


def _reap_if_ours(pgid):
    """Ownership-checked teardown: kill only a group whose leader still carries our mock argv."""
    try:
        cmd = open(f"/proc/{pgid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    if "run_arm_mock" not in cmd:
        return False
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass
    return True


def test_memory_guard_failure_is_nonzero_even_when_the_arm_exits_zero(tmp_path):
    """Forced trip: exactly one per-run marker, real start_guard provenance, matching run id and pgid,
    and a nonzero window even though the arm exits 0 and the cleanup refuses."""
    exp = sandbox(tmp_path)
    cleanup_calls = tmp_path / "cleanup.calls"
    (exp / "cleanup-run.sh").write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$1" >> {cleanup_calls}\nexit 4\n')  # refusing cleanup
    os.chmod(exp / "cleanup-run.sh", 0o755)
    pgid = None
    try:
        r = subprocess.run([str(exp / "window3.sh"), "--execute"],
                           env=env(exp, X3_MOCK="guard-trip", X3_MOCK_SLEEP="30"),
                           capture_output=True, text=True, timeout=60)
        assert r.returncode != 0, (r.returncode, r.stdout[-800:])
        markers = list((exp / "raw").glob("*.guard"))
        assert len(markers) == 1, f"expected exactly one per-run marker, got {markers}"
        fields = dict(line.split("=", 1) for line in markers[0].read_text().splitlines() if "=" in line)
        assert fields.get("source") == "start_guard", "marker did not come from the real guard"
        assert fields.get("reason") == "mem_floor" and fields.get("floor_mib") == "99999999"
        run_id = fields["run_id"]
        assert markers[0].name == f"window3-{run_id}.guard", "marker is not this run's file"
        pgid = int(fields["pgid"])
        assert f"reaping own group {pgid}" in r.stdout, "marker pgid is not the guarded group"
        assert (exp / "raw" / f"window3-{run_id}.log").exists(), \
            "marker run_id does not match the window's own log (run init not shared)"
        assert cleanup_calls.exists(), "the guard did not invoke the run-scoped cleanup"
    finally:
        if pgid:
            _reap_if_ours(pgid)
        survivors = subprocess.run(["ps", "-eo", "pgid=,args="], capture_output=True, text=True,
                                   timeout=30).stdout
        assert "run_arm_mock" not in survivors, "the rehearsal left a sleeper behind"


def test_memory_guard_healthy_cancellation_is_success(tmp_path):
    """A guard stopped on purpose must not fail the arm and must leave no marker (the 143 regression)."""
    exp = sandbox(tmp_path)
    pgid = None
    try:
        r = subprocess.run([str(exp / "window3.sh"), "--execute"],
                           env=env(exp, X3_MOCK="guard-healthy", X3_MOCK_SLEEP="30"),
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, (r.returncode, r.stdout[-800:])
        assert not list((exp / "raw").glob("*.guard")), "a healthy cancellation wrote a trip marker"
        assert "stopped orderly" in r.stdout
        for line in r.stdout.splitlines():
            if "cleanup-run: member" in line:
                pgid = int(line.split()[2])
    finally:
        if pgid:
            _reap_if_ours(pgid)
        survivors = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True,
                                   timeout=30).stdout
        assert "run_arm_mock" not in survivors, "the rehearsal left a sleeper behind"


def test_cleanup_ownership_is_run_scoped_and_fails_nonzero(tmp_path):
    def make_state(pgid, start, owner):
        st = tmp_path / "state"
        st.write_text(f"RUN_ID=test\nRAW={tmp_path}\nPHASE=running\nDRIVER_PID=1\nDRIVER_START=1\n"
                      f"ARM_PGID={pgid}\nARM_START={start}\nARM_OWNER_PATTERN={owner}\n")
        return st

    def spawn_probe_shaped():
        p = subprocess.Popen(["bash", "-c",
                              "exec -a 'python /x/probe_mtp_causal.py --index 18' sleep 30"],
                             start_new_session=True)
        time.sleep(0.3)
        start = open(f"/proc/{p.pid}/stat").read().split()[21]
        return p, start

    # wrong owner: refuse, nonzero, process survives
    p, start = spawn_probe_shaped()
    try:
        r = subprocess.run([str(CR), str(make_state(p.pid, start, "bench_ornith.py"))],
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 4, (r.returncode, r.stdout)
        assert p.poll() is None
    finally:
        os.killpg(p.pid, signal.SIGKILL); p.wait()

    # right owner: reaped, zero
    p, start = spawn_probe_shaped()
    r = subprocess.run([str(CR), str(make_state(p.pid, start, "probe_mtp_causal.py"))],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, (r.returncode, r.stdout)
    time.sleep(0.3)
    assert p.poll() is not None, "owned group was not reaped"
