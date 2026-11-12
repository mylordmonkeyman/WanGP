#!/usr/bin/env python3
"""Guarded one-time replacement of the non-Compose LTX-2.5 V100 runner.

Run on mdsserver. Dry-run by default; apply with LTX25_APPLY=1. This script
uses docker inspect, not guesses. It preserves the exact V100, 3 host bind
mounts, runner env/registration, entrypoint, user, network and restart policy.
The old stopped container is retained for rollback. Secrets are never printed,
written to Git, or passed on command lines. No build, downloads or model SHA.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

CONTAINER = os.environ.get("LTX25_RUNNER_CONTAINER", "gha-v100-ltx25")
IMAGE = os.environ.get("LTX25_COMPLETE_IMAGE",
                       "local/gha-v100-ltx25:cu128-ltx25-complete-v1")
EXPECTED_GPU = "GPU-8bbdb0ea-0a01-5e00-9309-081498b35342"
REQUIRED_MOUNTS = {
    "/cache": "/var/lib/llama/gha-runners/ltx25/cache",
    "/runner": "/var/lib/llama/gha-runners/ltx25/runner",
    "/work": "/var/lib/llama/gha-runners/ltx25/work",
}


def fatal(message):
    raise RuntimeError(message)


def run(args, *, quiet=False, input_text=None):
    # Never print complete argv: Docker container environment can be secret.
    opts = {"check": True, "text": True}
    if quiet:
        opts["stdout"] = subprocess.PIPE
        opts["stderr"] = subprocess.PIPE
    if input_text is not None:
        opts["input"] = input_text
    return subprocess.run(args, **opts)


def inspect(kind, name):
    return json.loads(run(["docker", kind, "inspect", name], quiet=True).stdout)[0]


def safe_github_idle():
    # Check all branch runs; do not interrupt another queued/in-progress job.
    args = ["gh", "api",
            "repos/mylordmonkeyman/WanGP/actions/runs?per_page=100",
            "--jq",
            '[.workflow_runs[] | select(.status=="in_progress" or .status=="queued" or .status=="waiting" or .status=="pending" or .status=="requested")] | length']
    result = run(args, quiet=True).stdout.strip()
    if not result.isdecimal() or int(result) != 0:
        fatal("GitHub Actions still has " + result + " active/queued runs")
    # Query the exact repository-registered runner where permissions allow.
    # If the endpoint is denied, abort instead of assuming idle.
    response = run(["gh", "api",
                    "repos/mylordmonkeyman/WanGP/actions/runners?per_page=100"],
                   quiet=True).stdout
    matching = [r for r in json.loads(response).get("runners", [])
                if r.get("name") == "mdsserver-v100-ltx25"]
    if len(matching) != 1:
        fatal("Could not identify exactly one mdsserver-v100-ltx25 runner via GitHub API")
    if matching[0].get("busy"):
        fatal("GitHub reports V100 runner as busy; refusing to stop it")
    return matching[0].get("status")


def require_default(host, config):
    def is_default(field, allowed):
        if host.get(field) not in allowed:
            fatal("Unsupported customized Docker HostConfig." + field
                  + "; refusing a lossy migration")
    is_default("Privileged", (False, None))
    is_default("ReadonlyRootfs", (False, None))
    is_default("AutoRemove", (False, None))
    is_default("NetworkMode", ("bridge",))
    is_default("PidMode", ("", None))
    is_default("UsernsMode", ("", None))
    is_default("CgroupParent", ("", None))
    is_default("OomKillDisable", (False, None))
    is_default("Memory", (0, None))
    is_default("MemorySwap", (0, None))
    is_default("NanoCpus", (0, None))
    is_default("CpuShares", (0, None))
    is_default("CpuQuota", (0, None))
    is_default("CpuPeriod", (0, None))
    # CPU/NUMA affinity is preserved explicitly by main(); never silently
    # drop cpuset restrictions that keep the V100 on the local NUMA node.
    for field in ("CpusetCpus", "CpusetMems"):
        value = host.get(field)
        if value not in ("", None) and not (
            isinstance(value, str)
            and re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", value)
        ):
            fatal("Unexpected Docker HostConfig." + field + " format")
    is_default("PidsLimit", (0, -1, None))
    for key in ("Devices", "DeviceCgroupRules", "CapAdd", "CapDrop",
                "SecurityOpt", "Ulimits", "GroupAdd", "ExtraHosts",
                "Dns", "DnsOptions", "DnsSearch", "Links", "VolumesFrom",
                "Tmpfs", "Sysctls", "PortBindings", "PublishAllPorts"):
        value = host.get(key)
        if value not in (None, [], {}, "", False):
            fatal("Unsupported custom HostConfig." + key
                  + "; inspect manually before any recreation")
    for key in ("ExposedPorts", "Volumes", "Healthcheck", "OnBuild"):
        if config.get(key) not in (None, {}, [], ""):
            fatal("Unsupported custom container Config." + key
                  + "; refusing to guess")
    if host.get("RestartPolicy", {}).get("Name") != "unless-stopped":
        fatal("Unexpected restart policy")
    if host.get("IpcMode") not in ("private", "", None):
        fatal("Unexpected custom IPC namespace")
    if host.get("CgroupnsMode") not in ("private", "", None):
        fatal("Unexpected custom cgroup namespace")
    if host.get("ShmSize", 67108864) != 67108864:
        fatal("Unexpected custom --shm-size")
    log = host.get("LogConfig") or {}
    if log.get("Type") not in ("json-file", "", None) or log.get("Config"):
        fatal("Unexpected custom log driver/options")
    if config.get("Tty") or config.get("OpenStdin") or config.get("StdinOnce"):
        fatal("Unexpected interactive/TTY configuration")
    if host.get("Init") not in (False, None):
        fatal("Unexpected init configuration")


def main():
    if os.environ.get("LTX25_APPLY", "0") not in ("0", "1"):
        fatal("LTX25_APPLY must be 0 (dry run) or 1 (explicit apply)")
    dry_run = os.environ.get("LTX25_APPLY", "0") == "0"
    old = inspect("container", CONTAINER)
    img = inspect("image", IMAGE)
    if not old["State"]["Running"]:
        fatal("Existing runner is not running; no changes made")
    if old["Image"] == img["Id"]:
        print("Already using complete checkpoint image", img["Id"])
        return
    host, cfg = old["HostConfig"], old["Config"]
    if cfg.get("Image") != "local/gha-v100-ltx25:cuda12.8":
        fatal("Unexpected original image " + str(cfg.get("Image")))
    if cfg.get("User") != "runner":
        fatal("Unexpected runner user")
    if cfg.get("Entrypoint") != ["/usr/local/bin/start-runner.sh"]:
        fatal("Unexpected original entrypoint")
    if cfg.get("Cmd") not in (None, []):
        fatal("Unexpected original Cmd")
    requests = host.get("DeviceRequests") or []
    if len(requests) != 1 or requests[0].get("DeviceIDs") != [EXPECTED_GPU]:
        fatal("GPU device request does not match documented V100 UUID")
    if requests[0].get("Capabilities") != [["gpu"]]:
        fatal("Unexpected GPU capabilities")
    if set((old.get("NetworkSettings") or {}).get("Networks", {})) != {"bridge"}:
        fatal("Runner is attached to unexpected networks")
    require_default(host, cfg)

    mounts = old.get("Mounts") or []
    if len(mounts) != 3:
        fatal("Expected exactly three runner bind mounts")
    actual = {}
    for m in mounts:
        if m.get("Type") != "bind" or not m.get("RW") or m.get("Propagation") not in ("rprivate", ""):
            fatal("Custom/readonly mount; refusing recreation")
        actual[m["Destination"]] = m["Source"]
    if actual != REQUIRED_MOUNTS:
        fatal("Bind mount destinations/sources differ from original runner")
    for source in actual.values():
        if not Path(source).is_dir():
            fatal("Original host bind source missing: " + source)

    vals = cfg.get("Env") or []
    for entry in vals:
        if "\n" in entry or "\r" in entry or "\0" in entry:
            fatal("Multiline Docker environment cannot be safely serialized")
        if "=" not in entry or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", entry.split("=", 1)[0]):
            fatal("Nonstandard Docker env key; fail closed")
    if not vals:
        fatal("Existing Docker env unexpectedly empty")
    ready_status = safe_github_idle()
    print("Verified runner idle via GitHub API, status:", ready_status)
    print("Old image:", cfg["Image"])
    print("New image:", IMAGE)
    print("V100:", EXPECTED_GPU)
    print("User/entrypoint: runner /usr/local/bin/start-runner.sh")
    print("Mounts:", ", ".join(f"{dst} <- {src}" for dst, src in sorted(actual.items())))
    print("Network/restart: bridge / unless-stopped")
    print("CPU affinity:", host.get("CpusetCpus") or "unrestricted")
    print("NUMA mem affinity:", host.get("CpusetMems") or "unrestricted")
    print("Existing environment: preserving", len(vals), "variables without displaying values")
    print("Image size:", img["Size"], "bytes")
    if dry_run:
        print("DRY_RUN_OK: no container changes, run with LTX25_APPLY=1 to deploy")
        return

    # Preflight the new image *before* touching the current runner. No model
    # bytes are read beyond stat metadata; no downloads and no full hashes.
    check = (
        'from pathlib import Path; import json,torch,mutagen; '
        'root=Path("/opt/ltx25/checkpoints"); '
        'm=json.loads(Path("/opt/ltx25/model-image-manifest.json").read_text()); '
        'assert len(m["files"])==17; '
        'assert all((root/x["path"]).is_file() and '
        '(root/x["path"]).stat().st_size==x["bytes"] for x in m["files"]); '
        'assert torch.cuda.is_available() and torch.cuda.get_device_capability(0)==(7,0); '
        'print("COMPLETE_IMAGE_PREFLIGHT_OK",torch.__version__,len(m["files"]))'
    )
    affinity_args = []
    for field, flag in (("CpusetCpus", "--cpuset-cpus"),
                        ("CpusetMems", "--cpuset-mems")):
        if host.get(field):
            affinity_args += [flag, host[field]]
    run(["docker", "run", "--rm", "--gpus", "device=" + EXPECTED_GPU,
         *affinity_args, "--entrypoint",
         "/opt/venvs/ltx25-cu128/bin/python", IMAGE, "-c", check])
    # Recheck immediately before stopping old runner.
    safe_github_idle()

    old_backup = CONTAINER + "-rollback-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    command = [
        "docker", "run", "-d", "--name", CONTAINER,
        "--user", "runner", "--entrypoint", "/usr/local/bin/start-runner.sh",
        "--restart", "unless-stopped", "--network", "bridge",
        "--gpus", "device=" + EXPECTED_GPU,
        *affinity_args,
    ]
    if cfg.get("WorkingDir"):
        command += ["--workdir", cfg["WorkingDir"]]
    if cfg.get("Hostname"):
        command += ["--hostname", cfg["Hostname"]]
    for name, value in (cfg.get("Labels") or {}).items():
        command += ["--label", name + "=" + value]
    for dst, src in sorted(actual.items()):
        # Docker --mount is read-write unless readonly is specified.
        command += ["--mount", "type=bind,src=" + src + ",dst=" + dst]
    # An --env-file means secrets NEVER enter argv/ps or command output.
    with tempfile.TemporaryDirectory(prefix="ltx25-docker-env-") as folder:
        ef = Path(folder) / "env"
        ef.write_text("\n".join(vals) + "\n", encoding="utf-8")
        ef.chmod(0o600)
        command += ["--env-file", str(ef),
                    "--env", "LTX25_CHECKPOINT_ROOT=/opt/ltx25/checkpoints",
                    "--env", "LTX25_MODEL_ASSETS_BAKED=1", IMAGE]

        stopped = renamed = created = False
        try:
            print("Stopping idle old runner...")
            run(["docker", "stop", "-t", "90", CONTAINER], quiet=True)
            stopped = True
            run(["docker", "rename", CONTAINER, old_backup], quiet=True)
            renamed = True
            print("Old container preserved (stopped):", old_backup)
            run(command, quiet=True)
            created = True
            new = inspect("container", CONTAINER)
            if new["Image"] != img["Id"] or not new["State"]["Running"]:
                fatal("New container missing/incorrect image or not running")
            if new["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] != [EXPECTED_GPU]:
                fatal("New runner GPU differs")
            for field in ("CpusetCpus", "CpusetMems"):
                if (new["HostConfig"].get(field) or "") != (host.get(field) or ""):
                    fatal("New runner CPU/NUMA affinity differs: " + field)
            if {m["Destination"]: m["Source"] for m in new["Mounts"]} != actual:
                fatal("New container lost a bind mount")
            run(["docker", "exec", CONTAINER,
                 "/opt/venvs/ltx25-cu128/bin/python", "-c", check])
            print("COMPLETE_RUNNER_DEPLOYED", img["Id"])
            print("Retained stopped rollback container:", old_backup)
            print("No GPU benchmark started; verify runner online before quality dispatch.")
        except Exception:
            print("DEPLOYMENT_FAILED: attempting automatic rollback (no images/volumes deleted)",
                  file=sys.stderr)
            # docker run can create a container and then return failure.
            # Inspect the name even if run() never returned successfully.
            probe = subprocess.run(["docker", "inspect", CONTAINER],
                                   check=False, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
            if renamed and probe.returncode == 0:
                subprocess.run(["docker", "rm", "-f", CONTAINER], check=False,
                               stdout=subprocess.DEVNULL)
            if renamed:
                subprocess.run(["docker", "rename", old_backup, CONTAINER],
                               check=False, stdout=subprocess.DEVNULL)
            if stopped:
                subprocess.run(["docker", "start", CONTAINER], check=False,
                               stdout=subprocess.DEVNULL)
            raise


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            print("ERROR: command failed, exit status", exc.returncode, file=sys.stderr)
        else:
            print("ERROR:", exc, file=sys.stderr)
        sys.exit(1)
