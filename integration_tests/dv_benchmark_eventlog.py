# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read-only Spark event-log summary for DV_BENCH job groups; does not start Spark."""

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import gzip
import io
import json
from pathlib import Path
import shutil
import subprocess


@contextmanager
def open_events(path):
    if ".zstd" in path.name:
        try:
            import zstandard
        except ImportError:
            if not shutil.which("zstd"):
                raise RuntimeError("Reading .zstd needs the zstandard Python module or zstd command")
            with subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE,
                                  text=True) as process:
                yield process.stdout
                if process.wait() != 0:
                    raise RuntimeError(f"zstd failed for {path}")
        else:
            with path.open("rb") as source:
                with zstandard.ZstdDecompressor().stream_reader(source) as reader:
                    with io.TextIOWrapper(reader) as lines:
                        yield lines
    elif ".gz" in path.name:
        with gzip.open(path, "rt") as lines:
            yield lines
    else:
        with path.open() as lines:
            yield lines


def plan_nodes(plan):
    names = Counter()
    if plan:
        names[plan.get("nodeName", "unknown")] += 1
        for child in plan.get("children", []):
            names.update(plan_nodes(child))
    return names


def read_profile(paths):
    jobs, ends, stages, sql = {}, {}, {}, {}
    tasks = defaultdict(Counter)
    for path in paths:
        with open_events(path) as lines:
            for line in lines:
                event = json.loads(line)
                kind = event.get("Event", "")
                if kind == "SparkListenerJobStart":
                    jobs[event["Job ID"]] = event
                elif kind == "SparkListenerJobEnd":
                    ends[event["Job ID"]] = event
                elif kind == "SparkListenerStageCompleted":
                    info = event["Stage Info"]
                    stages[(info["Stage ID"], info["Stage Attempt ID"])] = info
                elif kind == "SparkListenerTaskEnd":
                    counts = tasks[(event["Stage ID"], event["Stage Attempt ID"])]
                    metrics = event.get("Task Metrics", {})
                    counts["task_attempts"] += 1
                    counts["failed_attempts"] += (
                        event.get("Task End Reason", {}).get("Reason") != "Success")
                    counts["executor_run_ms"] += metrics.get("Executor Run Time", 0)
                    counts["executor_cpu_ms"] += metrics.get("Executor CPU Time", 0) / 1e6
                    counts["jvm_gc_ms"] += metrics.get("JVM GC Time", 0)
                    counts["memory_spilled_bytes"] += metrics.get("Memory Bytes Spilled", 0)
                    counts["disk_spilled_bytes"] += metrics.get("Disk Bytes Spilled", 0)
                    counts["input_bytes"] += metrics.get("Input Metrics", {}).get("Bytes Read", 0)
                    counts["shuffle_write_bytes"] += metrics.get(
                        "Shuffle Write Metrics", {}).get("Shuffle Bytes Written", 0)
                elif kind.endswith("SparkListenerSQLExecutionStart"):
                    sql[str(event["executionId"])] = {
                        "description": event.get("description", "")[:240],
                        "plan_nodes": dict(plan_nodes(event.get("sparkPlanInfo"))),
                    }
    return jobs, ends, stages, tasks, sql


def emit(kind, **fields):
    print("DV_PROFILE " + json.dumps(dict(kind=kind, **fields), sort_keys=True))


def select_jobs(jobs, operation, trial):
    execution_groups = {}
    for job in jobs.values():
        properties = job.get("Properties") or {}
        group = properties.get("spark.jobGroup.id", "")
        parts = group.split("/")
        if (len(parts) == 5 and parts[0] == "DV_BENCH"
                and parts[2].startswith(operation + "-") and parts[3] == str(trial)):
            execution = properties.get("spark.sql.execution.id")
            if execution is not None:
                execution_groups[str(execution)] = group
    selected = []
    for job_id, job in sorted(jobs.items()):
        properties = job.get("Properties") or {}
        group = properties.get("spark.jobGroup.id", "")
        parts = group.split("/")
        if (len(parts) == 5 and parts[0] == "DV_BENCH"
                and parts[2].startswith(operation + "-") and parts[3] == str(trial)):
            selected.append((job_id, job, properties, group))
        elif not group:
            # GPU broadcast jobs can retain the SQL execution ID but not the job group.
            # Do not reassociate jobs that explicitly belong to another group.
            inferred = execution_groups.get(str(properties.get("spark.sql.execution.id")))
            if inferred:
                selected.append((job_id, job, properties, inferred))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eventlog", type=Path)
    parser.add_argument("--operation", choices=["DELETE", "UPDATE", "MERGE"], default="MERGE")
    parser.add_argument("--trial", type=int, default=1)
    args = parser.parse_args()
    paths = [args.eventlog] if args.eventlog.is_file() else sorted(
        path for path in args.eventlog.rglob("*") if path.is_file()
        and not path.name.endswith(".crc")
        and path.name.startswith(("events_", "local-")))
    if not paths:
        parser.error("No Spark event files found; supply the eventlog directory or event file")
    jobs, ends, stages, tasks, sql = read_profile(paths)
    selected = select_jobs(jobs, args.operation, args.trial)
    if not selected:
        parser.error("No matching DV_BENCH jobs; check operation, trial, and eventlog path")
    emit("notes", files=[str(path) for path in paths],
         timing="Job/stage durations can overlap; task times sum all attempts, not wall time.",
         attribution="Job groups can include inherited background work; inspect SQL descriptions.",
         spilling="Spark spill counters do not measure all RAPIDS GPU/host spilling.")
    for job_id, job, properties, group in selected:
        end = ends.get(job_id, {})
        elapsed = (end["Completion Time"] - job["Submission Time"]) if end else None
        execution = properties.get("spark.sql.execution.id", "")
        emit("job", group=group, job_id=job_id, start_ms=job["Submission Time"],
             elapsed_ms=elapsed, stage_ids=job["Stage IDs"],
             attributed_by="job_group" if properties.get("spark.jobGroup.id") else "sql_execution",
             result=end.get("Job Result"), sql_id=execution, sql=sql.get(execution))
    selected_stages = {stage for _, job, _, _ in selected for stage in job["Stage IDs"]}
    for key, info in sorted(stages.items()):
        if key[0] in selected_stages:
            start, end = info.get("Submission Time"), info.get("Completion Time")
            emit("stage", stage_id=key[0], attempt=key[1], name=info.get("Stage Name"),
                 start_ms=start, elapsed_ms=end - start if start is not None and end else None,
                 tasks=info.get("Number of Tasks"), failure=info.get("Failure Reason"),
                 task_metrics=dict(tasks[key]))


if __name__ == "__main__":
    main()
