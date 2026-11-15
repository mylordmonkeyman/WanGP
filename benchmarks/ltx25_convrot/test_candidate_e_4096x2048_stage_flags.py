#!/usr/bin/env python3
"""Environment/phase contract for default-on Candidate E (4096,2048).

Runs isolated child Python interpreters to reset all import-time env flags.
The only supported per-stage controls are the two independent opt-out flags.
Previously duplicated experiment switches are rejected with an actionable error.
"""
import json
import os
import subprocess
import sys

VARS=(
    "WAN2GP_LTX25_E_TEST_4096X2048",
    "WAN2GP_LTX25_E_TEST_4096X2048_PHASES",
    "WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE1",
    "WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE2",
)
SCRIPT="""\
import json
from shared.kernels import ltx25_cublas_sm70 as mod
result={}
for phase in ('stage1','stage2'):
    mod.set_phase(phase)
    result[phase]=(4096,2048) in mod.PASS_SHAPES and phase in mod.ENABLED_4096X2048_PHASES
result['active']=sorted(mod.ENABLED_4096X2048_PHASES)
result['stat_active']=mod._STATS['active_4096x2048_phases']
result['stat_disabled']=mod._STATS['disabled_4096x2048_stages']
result['base_shape']=(2048,2048) in mod.PASS_SHAPES
result['min_m']=mod.MIN_M
result['max_m']=mod.MAX_M
print("E_SHAPE_FLAG_JSON="+json.dumps(result,sort_keys=True),flush=True)
"""

CASES=[
    ("default_on",{},["stage1","stage2"]),
    ("disable_stage1",{"WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE1":"1"},["stage2"]),
    ("disable_stage2",{"WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE2":"1"},["stage1"]),
    ("disable_both",{"WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE1":"1",
                     "WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE2":"1"},[]),
    ("explicit_zeros",{"WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE1":"0",
                       "WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE2":"0"},["stage1","stage2"]),

]
BAD=[
    {"WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE1":"YES"},
    {"WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE2":""},
    {"WAN2GP_LTX25_E_TEST_4096X2048_PHASES":"foo"},
    {"WAN2GP_LTX25_E_TEST_4096X2048":"true"},
    {"WAN2GP_LTX25_E_TEST_4096X2048":"0"},
    {"WAN2GP_LTX25_E_TEST_4096X2048":"1"},
    {"WAN2GP_LTX25_E_TEST_4096X2048_PHASES":"stage2"},
    {"WAN2GP_LTX25_E_TEST_4096X2048_PHASES":"stage1,stage2"},
    {"WAN2GP_LTX25_E_TEST_4096X2048_PHASES":"none,stage2"},
]


def child(specific):
    env=os.environ.copy()
    for key in VARS:
        env.pop(key,None)
    env.update(specific)
    return subprocess.run([sys.executable,"-c",SCRIPT],env=env,
                          capture_output=True,text=True,timeout=60)


def main():
    for name,flags,active in CASES:
        p=child(flags)
        if p.returncode:
            raise RuntimeError(f"{name}: code {p.returncode}: {p.stderr[-1200:]}")
        payloads=[l.split("E_SHAPE_FLAG_JSON=",1)[1] for l in p.stdout.splitlines()
                  if "E_SHAPE_FLAG_JSON=" in l]
        assert len(payloads)==1,(name,p.stdout[-1800:])
        got=json.loads(payloads[0])
        expected=sorted(active)
        assert got["active"]==expected,(name,got,expected)
        assert got["stat_active"]==expected,(name,got,expected)
        assert got["stage1"]==("stage1" in active),(name,got)
        assert got["stage2"]==("stage2" in active),(name,got)
        assert got["base_shape"] is True and got["min_m"]==64 and got["max_m"]==8192,(name,got)
        expect_disabled=sorted(k.replace("WAN2GP_LTX25_E_DISABLE_4096X2048_","").lower()
                                for k,v in flags.items()
                                if k.startswith("WAN2GP_LTX25_E_DISABLE_4096X2048_") and v=="1")
        assert got["stat_disabled"]==expect_disabled,(name,got,expect_disabled)
        print("CANDIDATE_E_FLAG_CONTRACT_PASS",name,got["active"],flush=True)
    for flags in BAD:
        p=child(flags)
        assert p.returncode!=0,(flags,p.stdout)
        assert "ValueError" in p.stderr,(flags,p.stderr[-1500:])
        print("CANDIDATE_E_INVALID_FLAG_REJECTED",flags,flush=True)
    print(f"LTX25_CANDIDATE_E_FLAG_CONTRACT_ALL_PASS {len(CASES)} good / {len(BAD)} invalid")


if __name__=="__main__":
    main()
