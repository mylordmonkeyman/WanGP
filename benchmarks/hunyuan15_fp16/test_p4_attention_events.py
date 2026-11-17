"""CPU-only tests for P4 CUDA-event aggregation and no-GPU fallback."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parent))
import p0_trace as trace


class Start:
    def __init__(self, end_ms): self.end_ms=end_ms
    def elapsed_time(self, other): return self.end_ms


class P4EventsTests(unittest.TestCase):
    def test_disabled_and_no_request_do_not_touch_cuda(self):
        with mock.patch.object(trace,"_active",None), mock.patch.dict(
                "os.environ",{"HUNYUAN15_P4_ATTENTION_EVENTS":"1"}):
            self.assertIsNone(trace.p4_attention_begin([], "pay_attention"))

    def test_216_pairs_from_two_levels(self):
        events=[]
        for kind,ms in (("pay_attention",5.0),("sdpa_core",4.5)):
            for _ in range(216):
                events.append({"kind":kind,"start":Start(ms),
                               "stop":object(),"host_ms":5.5,
                               "shape":(1,14159,16,128),"dtype":"torch.float16"})
        d=trace.p4_attention_summarize(events,34.0)
        self.assertEqual(d["intervals"]["pay_attention"]["count"],216)
        self.assertAlmostEqual(d["intervals"]["pay_attention"]["gpu_stream_total_ms"],1080.0)
        self.assertAlmostEqual(d["intervals"]["sdpa_core"]["gpu_stream_total_ms"],972.0)
        self.assertAlmostEqual(d["sdpa_stream_fraction_of_request"],972/34000)
        self.assertEqual(d["intervals"]["sdpa_core"]["first_shape"],[1,14159,16,128])

    def test_missing_inner_and_incomplete_intervals_fail(self):
        entries=[{"kind":"pay_attention","start":Start(1),"stop":object(),
                  "host_ms":1,"shape":(1,2,3),"dtype":"fp16"}]
        with self.assertRaises(RuntimeError):trace.p4_attention_summarize(entries,1)
        entries[0].pop("stop")
        with self.assertRaises(RuntimeError):trace.p4_attention_summarize(entries,1)

    def test_reject_wrong_call_count(self):
        entries=[]
        for kind in ("pay_attention","sdpa_core"):
            entries.append({"kind":kind,"start":Start(2),"stop":object(),
                            "host_ms":2,"shape":(1,2,3),"dtype":"fp16"})
        with self.assertRaisesRegex(RuntimeError,"216"):
            trace.p4_attention_summarize(entries,1)


if __name__ == "__main__":unittest.main()
