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


def probe_entry(tmp_path):
    """The wrapper validates that its entry point exists; give it a hermetic one."""
    entry = tmp_path / "probe_mtp_causal.py"
    entry.write_text("# stub entry for the argv sink\n")
    return entry


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
                               PROBE_ENTRY=str(probe_entry(tmp_path))),
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
                               FORK="/nonexistent/tree", TORCH_EXTENSIONS_DIR=str(exp / "ext-cache-instr"),
                               PROBE_ENTRY=str(probe_entry(tmp_path))),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 8 and "cache/tree mismatch" in r.stderr
    assert not (tmp_path / "argv.log").exists()


REPLACED_OBSERVATIONS = {
    "avail_mib() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }":
        'avail_mib() { cat "$EXP/raw/fake-avail-mib"; }',
    "gtt_bytes() { cat /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | head -1; }":
        'gtt_bytes() { cat "$EXP/raw/fake-gtt-bytes"; }',
    '  ss -ltn 2>/dev/null | grep -E ":1815[0-9] " && return 1':
        '  [ -e "$EXP/raw/fake-owner-present" ] && return 1',
    '  ps -eo args | grep -E "[o]rnith122-serve|[t]ools/glm/serve" && return 1':
        '  [ -e "$EXP/raw/fake-owner-present" ] && return 1',
}


def sandbox_with_fake_observations(exp, avail_mib=200000, gtt_bytes=0):
    """Point the sandbox copy's resource observations at files the test controls.

    window3.sh observes `/proc/meminfo`, `/sys/class/drm/*/device/mem_info_gtt_used` and the
    compute owners (`ss`/`ps`) by absolute path, so a CPU-only rehearsal on a host with a resident GPU
    owner - or with an unrelated monitor whose argv names a production service - cannot observe an
    idle box. Only those observation lines are replaced, and only in the sandbox copy: the thresholds,
    the preflight decision, the guard, the watchdog, the marker and cleanup paths are the real file
    text. The replacements still drive the real decision logic (the owner marker makes `compute_owners`
    refuse), so the rehearsal exercises both outcomes hermetically.
    """
    script = exp / "window3.sh"
    src = script.read_text()
    for old in REPLACED_OBSERVATIONS:
        assert src.count(old) == 1, f"unexpected observation line in window3.sh: {old!r}"
    (exp / "raw" / "fake-avail-mib").write_text(f"{int(avail_mib)}\n")
    (exp / "raw" / "fake-gtt-bytes").write_text(f"{int(gtt_bytes)}\n")
    patched = src
    for old, new in REPLACED_OBSERVATIONS.items():
        patched = patched.replace(old, new)
    script.write_text(patched)
    os.chmod(script, 0o755)
    pairs = [(a, b) for a, b in zip(src.splitlines(), patched.splitlines()) if a != b]
    assert [a for a, _ in pairs] == list(REPLACED_OBSERVATIONS), (
        f"the rehearsal patched more than the observation lines: {pairs}")
    assert [b for _, b in pairs] == list(REPLACED_OBSERVATIONS.values())
    return script


def test_window_probe_only_passes_the_opt_in_capture_through_the_guards(tmp_path):
    """The authorised entrypoint is `window3.sh --probe-only`, not a bare wrapper.

    The capture is opt-in through the environment, so the rehearsal checks the *actual* driver: the
    window's own selfcheck/preflight/guard/cleanup run, `run_probe.sh` receives the capture flags and
    the operator's unique nonce directory (which is independent of the RUN_ID the window generates),
    and nothing here starts a model or signals another run. The only fabricated inputs are the three
    resource observations (hermetic fixture files) and the already-stubbed entrypoints.
    """
    exp = sandbox(tmp_path)
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    argvlog = tmp_path / "argv.log"
    stub(venv / "python", argvlog)
    envscript = tmp_path / "env.sh"
    envscript.write_text("#!/usr/bin/env bash\ntrue\n")
    selfcheck = tmp_path / "selfcheck.log"
    stub(exp / "run_arm3.sh", selfcheck)
    cleanup = tmp_path / "cleanup.log"
    stub(exp / "cleanup-run.sh", cleanup)
    sandbox_with_fake_observations(exp)
    nonce = f"{tmp_path.name}-{int(time.time() * 1000)}"
    capture_root = exp / "raw" / f"capture-{nonce}-layer0"
    e = env(exp, X3_PROBE_RUNTIME="rocm713",
            ENV_SCRIPT=str(envscript), VENV=str(tmp_path / "venv"),
            FORK=str(exp / "exl3-v164-pristine"),
            TORCH_EXTENSIONS_DIR=str(exp / "ext-cache-pristine"),
            PROBE_ENTRY=str(probe_entry(tmp_path)),
            CAPTURE_DIR=str(capture_root), CAPTURE_POSITIONS="7,8,9",
            # the proposed real preflight spells out 80 GiB (81920 MiB); the driver floor and the
            # GTT ceiling stay at their defaults
            X3_ARM_MIN_AVAIL_MIB="81920", X3_PROBE_INDEX="18")
    r = subprocess.run([str(exp / "window3.sh"), "--probe-only"], env=e,
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, (r.returncode, r.stdout[-1500:], r.stderr[-800:])
    # the real preflight consumed the fake observations and admitted the arm on the real thresholds
    assert "arm preflight ok: avail=200000MiB gtt=0B owners=none" in r.stdout, r.stdout[-1200:]
    # the guard suite still ran: selfcheck, arm start, run-scoped cleanup
    assert "probe arm start" in r.stdout and "probe-only window complete" in r.stdout
    assert "selfcheck" in selfcheck.read_text().split()
    assert "runtime=rocm713" in r.stdout, "the rocm713 runtime binding did not reach run_probe.sh"
    assert cleanup.exists(), "the window did not run its run-scoped cleanup"
    argv = argvlog.read_text().split()
    assert argv[argv.index("--capture-dir") + 1] == str(capture_root)
    assert argv[argv.index("--capture-positions") + 1] == "7,8,9"
    assert capture_root.is_dir(), "run_probe.sh did not create the operator's capture directory"
    # the probe's own --out is the window RUN_ID's artifact; the capture nonce is separate
    out = argv[argv.index("--out") + 1]
    assert "-probe-causal" in out
    assert str(capture_root) not in out, "the capture nonce was assumed to be the window RUN_ID"
    # the dry run must not advertise a capture it was not given
    r = subprocess.run([str(exp / "window3.sh")], env=env(exp), capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0 and "DRY RUN" in r.stdout


def test_window_refuses_the_probe_arm_when_another_compute_owner_is_present(tmp_path):
    """The same rehearsal, with the owner observation set: the real preflight must refuse."""
    exp = sandbox(tmp_path)
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    argvlog = tmp_path / "argv.log"
    stub(venv / "python", argvlog)
    envscript = tmp_path / "env.sh"
    envscript.write_text("#!/usr/bin/env bash\ntrue\n")
    stub(exp / "run_arm3.sh", tmp_path / "selfcheck.log")
    stub(exp / "cleanup-run.sh", tmp_path / "cleanup.log")
    sandbox_with_fake_observations(exp)
    (exp / "raw" / "fake-owner-present").write_text("")
    e = env(exp, X3_PROBE_RUNTIME="rocm713",
            ENV_SCRIPT=str(envscript), VENV=str(tmp_path / "venv"),
            FORK=str(exp / "exl3-v164-pristine"),
            TORCH_EXTENSIONS_DIR=str(exp / "ext-cache-pristine"),
            PROBE_ENTRY=str(probe_entry(tmp_path)),
            CAPTURE_DIR=str(exp / "raw" / "capture-owner-layer0"))
    r = subprocess.run([str(exp / "window3.sh"), "--probe-only"], env=e,
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 20, (r.returncode, r.stdout[-1200:])
    assert "ARM PREFLIGHT FAIL: another compute owner present" in r.stdout
    assert "probe arm start" not in r.stdout, "the arm started despite the refused preflight"
    assert not argvlog.exists(), "the probe entrypoint ran despite the refused preflight"


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
