"""CPU regression fixture for P4 attention summarizer; no GPU or MP4 inputs."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import p4_attention_compare as report


class P4ReportTests(TestCase):
    def test_control_and_instrumented_reports_preserve_references(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            for label,events_on,median in (("control",False,34.0),("events",True,34.1)):
                path=root/label/"p4"/f"events_{int(events_on)}"
                path.mkdir(parents=True)
                recorded={
                    "status":"P4_ATTENTION_EVENTS_RECORDED",
                    "intervals":{
                        "sdpa_core":{"count":216,"gpu_stream_total_ms":900.0,"host_call_total_ms":1300.0},
                        "pay_attention":{"count":216,"gpu_stream_total_ms":1000.0,"host_call_total_ms":1400.0},
                    },
                    "sdpa_stream_fraction_of_request":0.026,
                }
                data={
                    "p4_cuda_events_enabled":events_on,
                    "effective_mmgp_profile_requested":1,
                    "effective_vae_config_requested":0,
                    "resolved_attention":"sdpa",
                    "warmup_generations":1,
                    "measured_generations":5,
                    "clips":[{"clip":f"{label}_{i}.mp4"} for i in range(6)],
                    "request_e2e_samples_seconds":[median]*5,
                    "checkpoint_sha256":"reference",
                    "converted_lora_sha256":"lora",
                    "effective_settings":{"video_length":33},
                    "p4_attention_hot_samples":[recorded]*5 if events_on else [None]*5,
                }
                (path/"baseline.json").write_text(json.dumps(data))
            with mock.patch.object(report,"frame_md5",return_value=["ok"]*33):
                outcome=report.evaluate(root)
            self.assertEqual(outcome["decoded_h264_frame_mismatch_counts"]["control"],[0]*5)
            self.assertEqual(outcome["decoded_h264_frame_mismatch_counts"]["events"],[0]*5)
            self.assertAlmostEqual(outcome["attention"]["median_sdpa_gpu_stream_seconds"],0.9)
            self.assertAlmostEqual(outcome["instrumentation_overhead_median_delta_seconds"],0.1)

    def test_missing_report_fails(self):
        with TemporaryDirectory() as tmp, self.assertRaises(RuntimeError):
            report.evaluate(Path(tmp))
