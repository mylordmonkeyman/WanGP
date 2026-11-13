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
    unsupported = []
    def is_default(field, allowed):
        if host.get(field) not in allowed:
            unsupported.append("HostConfig." + field)
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
    # Keep the original capability restrictions. "ALL" and named
    # capabilities are the only accepted Docker cap-drop forms.
    drops = host.get("CapDrop") or []
    if not isinstance(drops, list) or any(
        not isinstance(cap, str) or
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", cap)
        for cap in drops
    ):
        unsupported.append("HostConfig.CapDrop (unexpected format)")
    # Preserve Docker's no-new-privileges security restriction if present.
    # Fail closed on SELinux/AppArmor/seccomp customizations until audited.
    security_opts = host.get("SecurityOpt") or []
    if not isinstance(security_opts, list) or any(
        opt not in ("no-new-privileges", "no-new-privileges:true")
        for opt in security_opts
    ):
        unsupported.append("HostConfig.SecurityOpt (unsupported value)")
    # Exactly replay Docker resource limits from the old runner. Each
    # inspect entry is {Name, Soft, Hard}; -1 represents unlimited.
    ulimits = host.get("Ulimits") or []
    if not isinstance(ulimits, list):
        unsupported.append("HostConfig.Ulimits (not an array)")
    else:
        names = set()
        for item in ulimits:
            if not isinstance(item, dict) or set(item) != {"Name", "Soft", "Hard"}:
                unsupported.append("HostConfig.Ulimits (unexpected entry fields)")
                continue
            name, soft, hard = item["Name"], item["Soft"], item["Hard"]
            if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                unsupported.append("HostConfig.Ulimits (invalid limit name)")
            if type(soft) is not int or type(hard) is not int or soft < -1 or hard < -1:
                unsupported.append("HostConfig.Ulimits (invalid soft/hard value)")
            if name in names:
                unsupported.append("HostConfig.Ulimits (duplicate resource)")
            names.add(name)
            if type(soft) is int and type(hard) is int and hard != -1 and (soft == -1 or soft > hard):
                unsupported.append("HostConfig.Ulimits (soft exceeds hard)")
    for key in ("Devices", "DeviceCgroupRules", "CapAdd",
                "GroupAdd", "ExtraHosts",
                "Dns", "DnsOptions", "DnsSearch", "Links", "VolumesFrom",
                "Tmpfs", "Sysctls", "PortBindings", "PublishAllPorts"):
        value = host.get(key)
        if value not in (None, [], {}, "", False):
            unsupported.append("HostConfig." + key)
    for key in ("ExposedPorts", "Volumes", "Healthcheck", "OnBuild"):
        if config.get(key) not in (None, {}, [], ""):
            unsupported.append("Config." + key)
    if host.get("RestartPolicy", {}).get("Name") != "unless-stopped":
        unsupported.append("HostConfig.RestartPolicy")
    if host.get("IpcMode") not in ("private", "", None):
        unsupported.append("HostConfig.IpcMode")
    if host.get("CgroupnsMode") not in ("private", "", None):
        unsupported.append("HostConfig.CgroupnsMode")
    # Docker's /dev/shm size is often intentionally enlarged for PyTorch
    # multiprocess/video workloads. Read it from the actual container and
    # replay the exact number of bytes instead of assuming 64 MiB.
    shm_size = host.get("ShmSize", 67108864)
    if type(shm_size) is not int or not (4096 <= shm_size <= 1024**4):
        unsupported.append("HostConfig.ShmSize (invalid byte count)")
    log = host.get("LogConfig") or {}
    if log.get("Type") not in ("json-file", "", None) or log.get("Config"):
        unsupported.append("HostConfig.LogConfig")
    if config.get("Tty") or config.get("OpenStdin") or config.get("StdinOnce"):
        unsupported.append("Config.Tty/OpenStdin/StdinOnce")
    if host.get("Init") not in (False, None):
        unsupported.append("HostConfig.Init")
    if unsupported:
        fatal("Unsupported Docker settings (all detected): "
              + ", ".join(unsupported)
              + "; no changes were made; refusing lossy migration")


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
    print("Shared memory (/dev/shm):", host.get("ShmSize", 67108864), "bytes")
    print("Resource limits (Ulimits):", ", ".join(
        f"{entry['Name']}={entry['Soft']}:{entry['Hard']}"
        for entry in (host.get("Ulimits") or [])
    ) or "none")
    print("Dropped capabilities:", ", ".join(host.get("CapDrop") or []) or "none")
    print("Security options:", ", ".join(host.get("SecurityOpt") or []) or "default")
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
    # --shm-size accepts a byte count; applying it also to the preflight
    # ensures the image check runs with the real runner's memory settings.
    shm_args = ["--shm-size", str(host.get("ShmSize", 67108864))]
    ulimit_args = []
    for entry in host.get("Ulimits") or []:
        ulimit_args += [
            "--ulimit", f"{entry['Name']}={entry['Soft']}:{entry['Hard']}"
        ]
    restriction_args = []
    for cap in host.get("CapDrop") or []:
        restriction_args += ["--cap-drop", cap]
    for opt in host.get("SecurityOpt") or []:
        restriction_args += ["--security-opt", opt]
    run(["docker", "run", "--rm", "--gpus", "device=" + EXPECTED_GPU,
         *affinity_args, *shm_args, *ulimit_args, *restriction_args, "--entrypoint",
         "/opt/venvs/ltx25-cu128/bin/python", IMAGE, "-c", check])
    # Recheck immediately before stopping old runner.
    safe_github_idle()

    old_backup = CONTAINER + "-rollback-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    command = [
        "docker", "run", "-d", "--name", CONTAINER,
        "--user", "runner", "--entrypoint", "/usr/local/bin/start-runner.sh",
        "--restart", "unless-stopped", "--network", "bridge",
        "--gpus", "device=" + EXPECTED_GPU,
        *affinity_args, *shm_args, *ulimit_args, *restriction_args,
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
        # Create with 0600 atomically: avoid a brief world-readable window
        # between writing runner credentials and chmod().
        fd = os.open(ef, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.write("\n".join(vals) + "\n")
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
            if new["HostConfig"].get("ShmSize") != host.get("ShmSize"):
                fatal("New runner /dev/shm size differs from original")
            def limits_by_name(raw):
                return {e["Name"]: (e["Soft"], e["Hard"]) for e in raw or []}
            if limits_by_name(new["HostConfig"].get("Ulimits")) != limits_by_name(host.get("Ulimits")):
                fatal("New runner resource limits differ from original")
            # Docker may normalize capitalization and the implicit :true
            # suffix. Compare the restrictions semantically, not formatting.
            def normalized_cap(value):
                value = value.upper()
                return value[4:] if value.startswith("CAP_") else value
            old_drops = sorted(normalized_cap(x) for x in (host.get("CapDrop") or []))
            new_drops = sorted(normalized_cap(x) for x in
                               (new["HostConfig"].get("CapDrop") or []))
            if old_drops != new_drops:
                fatal("New runner dropped Linux capabilities differ")
            def normalized_security(value):
                return ("no-new-privileges:true"
                        if value == "no-new-privileges" else value)
            old_opts = sorted(normalized_security(x) for x in (host.get("SecurityOpt") or []))
            new_opts = sorted(normalized_security(x) for x in
                              (new["HostConfig"].get("SecurityOpt") or []))
            if old_opts != new_opts:
                fatal("New runner security options differ")
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
