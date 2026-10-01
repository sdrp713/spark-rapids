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

"""Opt-in DBR DV benchmark. Select this file explicitly; see delta_dml_dv_benchmark.md.

All table generation, cloning, plan inspection, and validation are outside the timer.
Artifacts are retained in a unique child of pytest's explicit --tmp_path.
"""

import json
import statistics
import time
import uuid
from urllib.parse import urlparse

import pytest
from pyspark.sql import functions as f

from conftest import get_inject_oom_conf
from spark_session import is_databricks173_or_later, with_spark_session


pytestmark = [pytest.mark.delta_lake, pytest.mark.allow_non_gpu(any=True),
              pytest.mark.spark_job_timeout(seconds=14400)]

# Benchmark inputs, not plugin configuration. Increase full-scale rows if scans are too short.
PILOT = dict(rows=1_000_000, files=8, repeats=1, thresholds=[1], existing_dvs=[True])
MEDIUM = dict(rows=10_000_000, files=32, repeats=3, thresholds=[1], existing_dvs=[True])
FULL = dict(rows=50_000_000, files=128, repeats=5,
            thresholds=[1, 100], existing_dvs=[False, True])
MODULUS = 1000
PAYLOAD_COLUMNS = 4  # Four deterministic SHA-256 strings: 256 payload bytes per row.

CONF = {
    "spark.rapids.sql.format.delta.write.enabled": "true",
    "spark.rapids.sql.format.parquet.enabled": "true",
    "spark.rapids.sql.format.parquet.write.enabled": "true",
    "spark.rapids.sql.delta.deletionVectors.predicatePushdown.enabled": "true",
    "spark.databricks.delta.deletionVectors.useMetadataRowIndex": "true",
    "spark.databricks.delta.delete.deletionVectors.persistent": "true",
    "spark.databricks.delta.update.deletionVectors.persistent": "true",
    "spark.databricks.delta.merge.deletionVectors.persistent": "true",
    "spark.databricks.delta.autoCompact.enabled": "false",
    "spark.databricks.delta.optimizeWrite.enabled": "false",
    "spark.databricks.io.cache.enabled": "false",
    # Match the tested DBR 17.3 MERGE path; compare both engines with AQE disabled.
    "spark.sql.adaptive.enabled": "false",
    "spark.sql.shuffle.partitions": "128",
    "spark.rapids.sql.explain": "NONE",
    "spark.rapids.sql.test.enabled": "false",
    "spark.rapids.sql.test.injectRetryOOM": "false",
    "spark.rapids.sql.test.retryContextCheck.enabled": "false",
    "spark.sql.session.timeZone": "UTC",
}
for _command in ("Delete", "Update", "MergeInto"):
    for _suffix in ("Command", "CommandEdge"):
        CONF[f"spark.rapids.sql.command.{_command}{_suffix}"] = "true"


def _emit(kind, **fields):
    print("DV_BENCH " + json.dumps(dict(kind=kind, **fields), sort_keys=True), flush=True)


def _session(func, engine="CPU"):
    return with_spark_session(func, conf=dict(
        CONF, **{"spark.rapids.sql.enabled": str(engine == "GPU").lower()}))


def _table(path):
    assert "`" not in path, "Benchmark paths must not contain backticks"
    return f"delta.`{path}`"


def _history(spark, path):
    return spark.sql(f"DESCRIBE HISTORY {_table(path)}").orderBy(
        f.desc("version")).first().asDict(recursive=True)


def _clone(spark, source, target, version):
    # No replace/overwrite: a collision fails instead of changing an existing table.
    spark.sql(f"CREATE TABLE {_table(target)} SHALLOW CLONE {_table(source)} "
              f"VERSION AS OF {version}").collect()


def _write_base(spark, path, settings):
    df = spark.range(settings["rows"], numPartitions=settings["files"])
    df = df.withColumn("v", f.col("id") * 7)
    for index in range(PAYLOAD_COLUMNS):
        df = df.withColumn(f"payload_{index}", f.sha2(
            f.concat_ws(":", f.col("id").cast("string"), f.lit(str(index))), 256))
    df.write.format("delta").mode("errorifexists") \
        .option("delta.enableDeletionVectors", "true") \
        .option("delta.enableChangeDataFeed", "false") \
        .option("delta.enableRowTracking", "false") \
        .option("delta.autoOptimize.autoCompact", "false") \
        .option("delta.autoOptimize.optimizeWrite", "false").save(path)
    detail = spark.sql(f"DESCRIBE DETAIL {_table(path)}").first().asDict()
    return {key: detail[key] for key in ("numFiles", "sizeInBytes")}


def _assert_dvs(history, command):
    metrics = history["operationMetrics"]
    prefix = "numTargetDeletionVectors" if command == "MERGE" else "numDeletionVectors"
    assert sum(int(metrics.get(prefix + suffix, 0)) for suffix in ("Added", "Updated")) > 0, \
        f"Persistent DV use was not proved: {metrics}"


def _sql(command, path, source_path, threshold):
    predicate = f"pmod(id, {MODULUS}) < {threshold}"
    if command == "DELETE":
        return f"DELETE FROM {_table(path)} WHERE {predicate}"
    if command == "UPDATE":
        return f"UPDATE {_table(path)} SET v = v + 1 WHERE {predicate}"
    # Matched-update MERGE isolates the existing-row/DV path. Source is materialized Parquet.
    return (f"MERGE INTO {_table(path)} AS t USING parquet.`{source_path}` AS s "
            "ON t.id = s.id WHEN MATCHED THEN UPDATE SET t.v = s.v")


def _execute(spark, sql, command, engine, label, capture):
    callback = spark.sparkContext._jvm.org.apache.spark.sql.rapids.ExecutionPlanCaptureCallback
    spark.sparkContext.setJobGroup(label, label)
    try:
        if capture:
            callback.startCapture()
        _emit("phase", phase="dml", label=label, engine=engine)
        start = time.perf_counter()
        result = [row.asDict() for row in spark.sql(sql).collect()]
        seconds = time.perf_counter() - start
        if capture:
            plans = list(callback.getResultsWithTimeout(10000))
            classes = ["GpuDeleteCommand", "GpuUpdateCommand", "GpuMergeIntoCommand",
                       "GpuFileSourceScanExec", "FileSourceScanExec", "RapidsDeltaWrite"]
            present = [name for name in classes
                       if any(callback.contains(plan, name) for plan in plans)]
            _emit("plans", label=label, engine=engine, classes=present,
                  plans=[plan.toString() for plan in plans])
            if engine == "GPU":
                expected = "Gpu" + ("MergeInto" if command == "MERGE" else command.title())
                assert expected + "Command" in present, "DML command fell back to CPU"
                assert "GpuFileSourceScanExec" in present, "No GPU file scan captured"
                if command != "DELETE":
                    assert "RapidsDeltaWrite" in present, "No GPU Parquet write captured"
        return dict(seconds=seconds, result=result)
    finally:
        if capture:
            callback.endCapture()
        spark.sparkContext.setLocalProperty("spark.jobGroup.id", None)
        spark.sparkContext.setLocalProperty("spark.job.description", None)


def _validate(spark, base, version, cpu, gpu, command, threshold, expected_rows):
    # Exact multiset comparison on CPU, including all payloads; nothing collected to the driver.
    expected = spark.read.format("delta").option("versionAsOf", version).load(base)
    predicate = f.pmod(f.col("id"), f.lit(MODULUS)) < threshold
    if command == "DELETE":
        expected = expected.where(~predicate)
    else:
        expected = expected.withColumn("v", f.when(predicate, f.col("v") + 1)
                                       .otherwise(f.col("v")))
    cpu_df = spark.read.format("delta").load(cpu).select(expected.columns)
    gpu_df = spark.read.format("delta").load(gpu).select(expected.columns)
    _emit("phase", phase="validation_counts", cpu=cpu, gpu=gpu)
    assert cpu_df.count() == gpu_df.count() == expected_rows
    # Equal cardinalities plus a one-way multiset difference imply multiset equality.
    _emit("phase", phase="validation_expected_vs_cpu", cpu=cpu)
    assert expected.exceptAll(cpu_df).limit(1).count() == 0, "CPU differs from expected rows"
    _emit("phase", phase="validation_cpu_vs_gpu", cpu=cpu, gpu=gpu)
    assert cpu_df.exceptAll(gpu_df).limit(1).count() == 0, "CPU/GPU rows differ"


def _environment(spark):
    sc = spark.sparkContext
    keys = ["spark.driver.memory", "spark.executor.memory", "spark.executor.cores",
            "spark.executor.instances", "spark.plugins", "spark.jars",
            "spark.eventLog.enabled", "spark.eventLog.dir", "spark.eventLog.compress",
            "spark.rapids.memory.host.spillStorageSize", "spark.rapids.sql.batchSizeBytes",
            "spark.rapids.memory.gpu.allocSize",
            "spark.rapids.sql.concurrentGpuTasks", "spark.dynamicAllocation.enabled",
            "spark.databricks.clusterUsageTags.sparkVersion",
            "spark.databricks.photon.enabled"]
    return dict(spark_version=spark.version, master=sc.master, app_id=sc.applicationId,
                default_parallelism=sc.defaultParallelism,
                jvm_max_heap_bytes=sc._jvm.java.lang.Runtime.getRuntime().maxMemory(),
                parquet_reader={key: spark.conf.get(key) for key in (
                    "spark.sql.parquet.columnarReaderBatchSize",
                    "spark.sql.parquet.enableVectorizedReader")},
                startup={key: sc.getConf().get(key, "<unset>") for key in keys},
                session=CONF)


def _run(request, settings, scale):
    assert is_databricks173_or_later(), "This benchmark requires the DBR 17.3 DV implementation"
    assert not hasattr(request.config, "workerinput"), "Set TEST_PARALLEL=0"
    assert get_inject_oom_conf() is None, "Use --test_oom_injection_mode=never for timings"
    root = request.config.getoption("tmp_path")
    assert root, "Provide an explicit --tmp_path for benchmark artifacts"
    root = root.rstrip("/") + "/dv-benchmark-" + uuid.uuid4().hex
    environment = _session(_environment)
    if not environment["master"].startswith("local["):
        assert urlparse(root).scheme in ("s3", "s3a", "abfs", "abfss", "hdfs", "dbfs"), \
            "Use shared storage for workers"
    _emit("environment", root=root, scale=scale, settings=settings, **environment)
    assert environment["startup"]["spark.databricks.photon.enabled"].lower() != "true", \
        "Disable Photon for this CPU/GPU comparison"

    base = root + "/base"
    detail = _session(lambda spark: _write_base(spark, base, settings))
    _emit("base", path=base, clean_version=0, **detail)

    def seed(spark):
        spark.sql(f"DELETE FROM {_table(base)} "
                  f"WHERE pmod(id, {MODULUS}) = {MODULUS - 1}").collect()
        history = _history(spark, base)
        _assert_dvs(history, "DELETE")
        return history["version"]

    seeded_version = _session(seed)
    assert settings["rows"] % MODULUS == 0
    for threshold in settings["thresholds"]:
        source_path = root + f"/source-{threshold}"

        def write_source(spark):
            spark.range(settings["rows"], numPartitions=settings["files"]) \
                .where(f.pmod(f.col("id"), f.lit(MODULUS)) < threshold) \
                .selectExpr("id", "id * 7 + 1 AS v") \
                .write.mode("errorifexists").parquet(source_path)

        _session(write_source)
        affected = settings["rows"] // MODULUS * threshold
        for existing_dv in settings["existing_dvs"]:
            version = seeded_version if existing_dv else 0
            initial_rows = settings["rows"] - (settings["rows"] // MODULUS if existing_dv else 0)
            for command in ("DELETE", "UPDATE", "MERGE"):
                case = f"{command}-{threshold}permille-existingdv-{existing_dv}"
                measured = {engine: [] for engine in ("CPU", "GPU")}
                paired_speedups = []
                # Round zero verifies plans and warms both engines; exclude it from all statistics.
                for trial in range(settings["repeats"] + 1):
                    paths = {engine: root + f"/{case}/{trial}/{engine}"
                             for engine in ("CPU", "GPU")}
                    for path in paths.values():
                        _session(lambda spark: _clone(spark, base, path, version))
                    order = ("CPU", "GPU") if trial % 2 == 0 else ("GPU", "CPU")
                    results = {}
                    # Validate only after BOTH timed commands to avoid warming one side via checks.
                    for engine in order:
                        sql = _sql(command, paths[engine], source_path, threshold)
                        label = f"DV_BENCH/{scale}/{case}/{trial}/{engine}"
                        results[engine] = _session(lambda spark: _execute(
                            spark, sql, command, engine, label, capture=trial == 0), engine)
                    assert results["CPU"]["result"] == results["GPU"]["result"]
                    for engine in order:
                        history = _session(lambda spark: _history(spark, paths[engine]))
                        assert history["operation"] == command
                        _assert_dvs(history, command)
                        row_key = {"DELETE": "numDeletedRows", "UPDATE": "numUpdatedRows",
                                   "MERGE": "numTargetRowsUpdated"}[command]
                        assert int(history["operationMetrics"][row_key]) == affected
                        _emit("sample", case=case, engine=engine, trial=trial,
                              warmup=trial == 0, path=paths[engine],
                              metrics=history["operationMetrics"], **results[engine])
                    _session(lambda spark: _validate(
                        spark, base, version, paths["CPU"], paths["GPU"], command, threshold,
                        initial_rows - affected if command == "DELETE" else initial_rows))
                    _emit("validated", case=case, trial=trial)
                    if trial > 0:
                        for engine in order:
                            measured[engine].append(results[engine]["seconds"])
                        paired_speedups.append(results["CPU"]["seconds"] /
                                               results["GPU"]["seconds"])
                cpu_median, gpu_median = (statistics.median(measured[engine])
                                          for engine in ("CPU", "GPU"))
                _emit("summary", case=case, scale=scale, rows=settings["rows"],
                      target_bytes=detail["sizeInBytes"], affected_rows=affected,
                      cpu_seconds=measured["CPU"], gpu_seconds=measured["GPU"],
                      cpu_median=cpu_median, gpu_median=gpu_median,
                      speedup=cpu_median / gpu_median,
                      paired_speedup_median=statistics.median(paired_speedups),
                      paired_speedups=paired_speedups)
    _emit("complete", root=root, scale=scale)


def test_dv_dml_benchmark_pilot(request):
    _run(request, PILOT, "pilot")


@pytest.mark.parametrize("shuffle_partitions", [128, 8], ids=["shuffle128", "shuffle8"])
def test_dv_dml_benchmark_shuffle_pilot(request, monkeypatch, shuffle_partitions):
    # Six measured pairs give each engine three first/second positions.
    # Override the benchmark session configuration, not just Spark startup defaults.
    monkeypatch.setitem(CONF, "spark.sql.shuffle.partitions", str(shuffle_partitions))
    _run(request, dict(PILOT, repeats=6), f"pilot-shuffle{shuffle_partitions}")


def test_dv_dml_benchmark_medium(request, monkeypatch):
    monkeypatch.setitem(CONF, "spark.sql.shuffle.partitions", "32")
    _run(request, MEDIUM, "medium-shuffle32")


def test_dv_dml_benchmark_full(request):
    _run(request, FULL, "full")
