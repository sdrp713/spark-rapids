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
The store_sales variant expects a separately validated local Parquet sample in
<--tmp_path>/raw; it is never part of automatic integration-test discovery.
It reports whether each command persisted DVs or rewrote files.
"""

import json
import os
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
# The 32-file medium run triggered a post-MERGE CPU OPTIMIZE; retain the pilot's file count.
MEDIUM = dict(rows=10_000_000, files=8, repeats=3, thresholds=[1], existing_dvs=[True])
# Approximate the 10% changed-row density reported in the OSS DML benchmark.
MEDIUM_DENSE = dict(MEDIUM, thresholds=[100])
FULL = dict(rows=50_000_000, files=128, repeats=5,
            thresholds=[1, 100], existing_dvs=[False, True])
STORE_SALES_THRESHOLD = 100  # Match the 10% changed-row density of the OSS comparison.
STORE_SALES_REPEATS = 3
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


def _command_history(spark, path, command):
    # DBR can append an OPTIMIZE commit after DML. Inspect the DML commit itself.
    row = spark.sql(f"DESCRIBE HISTORY {_table(path)}").where(
        f.col("operation") == command).orderBy(f.desc("version")).first()
    assert row is not None, f"No {command} commit found at {path}"
    return row.asDict(recursive=True)


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


def _write_store_sales_base(spark, source, path):
    # Persist the surrogate key once; it is unique even when TPC-DS business keys repeat.
    df = spark.read.parquet(source).withColumn("id", f.monotonically_increasing_id())
    assert "ss_quantity" in df.columns and "ss_sold_date_sk" in df.columns, \
        "Expected a store_sales Parquet sample"
    df.write.format("delta").mode("errorifexists") \
        .option("delta.enableDeletionVectors", "true") \
        .option("delta.enableChangeDataFeed", "false") \
        .option("delta.enableRowTracking", "false") \
        .option("delta.autoOptimize.autoCompact", "false") \
        .option("delta.autoOptimize.optimizeWrite", "false") \
        .partitionBy("ss_sold_date_sk").save(path)
    detail = spark.sql(f"DESCRIBE DETAIL {_table(path)}").first().asDict()
    return {key: detail[key] for key in ("numFiles", "sizeInBytes")}


def _assert_dvs(history, command):
    metrics = history["operationMetrics"]
    prefix = "numTargetDeletionVectors" if command == "MERGE" else "numDeletionVectors"
    assert sum(int(metrics.get(prefix + suffix, 0)) for suffix in ("Added", "Updated")) > 0, \
        f"Persistent DV use was not proved: {metrics}"


def _dml_strategy(metrics, command):
    dv_prefix = "numTargetDeletionVectors" if command == "MERGE" else "numDeletionVectors"
    file_key = "numTargetFilesRemoved" if command == "MERGE" else "numRemovedFiles"
    has_dvs = sum(int(metrics.get(dv_prefix + suffix, 0))
                  for suffix in ("Added", "Updated")) > 0
    has_rewrites = int(metrics.get(file_key, 0)) > 0
    if has_dvs and has_rewrites:
        return "mixed"
    if has_dvs:
        return "persistent-dv"
    if has_rewrites:
        return "file-rewrite"
    return "unknown"


def _sql(command, path, source_path, threshold):
    predicate = f"pmod(id, {MODULUS}) < {threshold}"
    if command == "DELETE":
        return f"DELETE FROM {_table(path)} WHERE {predicate}"
    if command == "UPDATE":
        return f"UPDATE {_table(path)} SET v = v + 1 WHERE {predicate}"
    # Matched-update MERGE isolates the existing-row/DV path. Source is materialized Parquet.
    return (f"MERGE INTO {_table(path)} AS t USING parquet.`{source_path}` AS s "
            "ON t.id = s.id WHEN MATCHED THEN UPDATE SET t.v = s.v")


def _store_sales_sql(command, path, source_path, threshold):
    predicate = f"pmod(id, {MODULUS}) < {threshold}"
    if command == "DELETE":
        return f"DELETE FROM {_table(path)} WHERE {predicate}"
    if command == "UPDATE":
        return (f"UPDATE {_table(path)} SET ss_quantity = ss_quantity + 1 "
                f"WHERE {predicate}")
    return (f"MERGE INTO {_table(path)} AS t USING parquet.`{source_path}` AS s "
            "ON t.id = s.id WHEN MATCHED THEN UPDATE SET t.ss_quantity = s.ss_quantity")


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
                       "GpuFileSourceScanExec", "FileSourceScanExec", "RapidsDeltaWrite",
                       "GpuHashAggregateExec", "ObjectHashAggregateExec",
                       "GpuRowToColumnarExec", "GpuColumnarToRowExec",
                       "MapPartitionsExec", "DeserializeToObjectExec",
                       "SerializeFromObjectExec", "LocalTableScanExec"]
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


def _store_sales_fingerprint(df):
    # A full-table aggregate avoids the large shuffle of exceptAll on this 50 GiB input.
    row_hash = f.xxhash64(*(f.col(name) for name in df.columns))
    row = df.agg(f.count("*").alias("rows"),
                 f.sum(row_hash.cast("decimal(38,0)")).alias("hash_sum")).first()
    return row.asDict()


def _validate_store_sales(spark, base, version, cpu, gpu, command, threshold):
    expected = spark.read.format("delta").option("versionAsOf", version).load(base)
    predicate = f.pmod(f.col("id"), f.lit(MODULUS)) < threshold
    if command == "DELETE":
        expected = expected.where(~predicate)
    else:
        expected = expected.withColumn(
            "ss_quantity", f.when(predicate, f.col("ss_quantity") + 1)
            .otherwise(f.col("ss_quantity")))
    signature = _store_sales_fingerprint(expected)
    for engine, path in (("CPU", cpu), ("GPU", gpu)):
        actual = spark.read.format("delta").load(path).select(expected.columns)
        assert _store_sales_fingerprint(actual) == signature, \
            f"{engine} result differs from the expected store_sales fingerprint"
    _emit("validated", method="count-and-xxhash64-sum", cpu=cpu, gpu=gpu,
          rows=signature["rows"])


def _environment(spark):
    sc = spark.sparkContext
    keys = ["spark.driver.memory", "spark.executor.memory", "spark.executor.cores",
            "spark.executor.instances", "spark.plugins", "spark.jars",
            "spark.eventLog.enabled", "spark.eventLog.dir", "spark.eventLog.compress",
            "spark.rapids.memory.host.spillStorageSize", "spark.rapids.sql.batchSizeBytes",
            "spark.rapids.memory.gpu.allocSize",
            "spark.rapids.sql.concurrentGpuTasks", "spark.dynamicAllocation.enabled",
            "spark.rapids.sql.expression.InputFileName",
            "spark.sql.files.maxPartitionBytes",
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


def _run_store_sales(request, commands=("DELETE", "UPDATE", "MERGE"),
                     repeats=STORE_SALES_REPEATS, scale="store-sales-local-10pct"):
    assert is_databricks173_or_later(), "This benchmark requires the DBR 17.3 DV implementation"
    assert not hasattr(request.config, "workerinput"), "Set TEST_PARALLEL=0"
    assert get_inject_oom_conf() is None, "Use --test_oom_injection_mode=never for timings"
    input_parent = request.config.getoption("tmp_path")
    assert input_parent and os.path.isabs(input_parent), \
        "Set --tmp_path to the absolute parent of the downloaded raw directory"
    source = os.path.join(input_parent, "raw")
    assert os.path.isdir(source), f"Downloaded store_sales sample not found: {source}"
    files = [os.path.join(directory, name)
             for directory, _, names in os.walk(source)
             for name in names if name.endswith(".parquet")]
    assert files, f"No Parquet files found under {source}"
    input_bytes = sum(os.path.getsize(path) for path in files)
    root = os.path.join(input_parent, "dv-benchmark-" + uuid.uuid4().hex)
    environment = _session(_environment)
    _emit("environment", root=root, scale=scale, source=source,
          input_files=len(files), input_bytes=input_bytes,
          threshold=STORE_SALES_THRESHOLD, repeats=repeats, **environment)
    assert environment["master"].startswith("local["), \
        "The downloaded sample is driver-local; run Spark in local mode"
    assert environment["jvm_max_heap_bytes"] >= 8 * 1024 ** 3, \
        "Use DRIVER_MEMORY=16g and SPARK_SUBMIT_FLAGS='--driver-memory 16g'"
    assert environment["startup"]["spark.databricks.photon.enabled"].lower() != "true", \
        "Disable Photon for this CPU/GPU comparison"

    base = root + "/base"
    detail = _session(lambda spark: _write_store_sales_base(spark, source, base))
    _emit("base", path=base, clean_version=0, **detail)

    def prepare(spark):
        df = spark.read.format("delta").load(base)
        rows = df.count()
        predicate = f.pmod(f.col("id"), f.lit(MODULUS)) < STORE_SALES_THRESHOLD
        affected = df.where(predicate).count()
        assert rows > affected > 0
        spark.sql(f"DELETE FROM {_table(base)} WHERE pmod(id, {MODULUS}) = "
                  f"{MODULUS - 1}").collect()
        seed = _command_history(spark, base, "DELETE")
        _assert_dvs(seed, "DELETE")
        seed_deleted = int(seed["operationMetrics"]["numDeletedRows"])
        assert seed_deleted > 0
        return rows, affected, seed_deleted, seed["version"]

    rows, affected, seed_deleted, seeded_version = _session(prepare)
    initial_rows = rows - seed_deleted
    _emit("seeded", rows=rows, affected_rows=affected, seed_deleted_rows=seed_deleted,
          version=seeded_version)
    source_path = root + "/merge-source"

    def write_source(spark):
        spark.read.format("delta").option("versionAsOf", seeded_version).load(base) \
            .where(f.pmod(f.col("id"), f.lit(MODULUS)) < STORE_SALES_THRESHOLD) \
            .select(f.col("id"), (f.col("ss_quantity") + 1).alias("ss_quantity")) \
            .write.mode("errorifexists").parquet(source_path)

    if "MERGE" in commands:
        _session(write_source)
    for command in commands:
        case = f"{command}-{STORE_SALES_THRESHOLD}permille-existingdv-True"
        measured = {engine: [] for engine in ("CPU", "GPU")}
        strategies = {engine: [] for engine in ("CPU", "GPU")}
        paired_speedups = []
        for trial in range(repeats + 1):
            paths = {engine: root + f"/{case}/{trial}/{engine}"
                     for engine in ("CPU", "GPU")}
            for path in paths.values():
                _session(lambda spark: _clone(spark, base, path, seeded_version))
            order = ("CPU", "GPU") if trial % 2 == 0 else ("GPU", "CPU")
            results = {}
            for engine in order:
                sql = _store_sales_sql(command, paths[engine], source_path,
                                       STORE_SALES_THRESHOLD)
                label = f"DV_BENCH/{scale}/{case}/{trial}/{engine}"
                results[engine] = _session(lambda spark: _execute(
                    spark, sql, command, engine, label, capture=trial == 0), engine)
            assert results["CPU"]["result"] == results["GPU"]["result"]
            row_key = {"DELETE": "numDeletedRows", "UPDATE": "numUpdatedRows",
                       "MERGE": "numTargetRowsUpdated"}[command]
            for engine in order:
                history = _session(lambda spark: _command_history(
                    spark, paths[engine], command))
                metrics = history["operationMetrics"]
                strategy = _dml_strategy(metrics, command)
                strategies[engine].append(strategy)
                _emit("sample", case=case, engine=engine, trial=trial,
                      warmup=trial == 0, path=paths[engine],
                      strategy=strategy, metrics=metrics, **results[engine])
                assert int(metrics[row_key]) == affected
                if command == "MERGE":
                    # This workload may rewrite files despite DV persistence being enabled.
                    # Report that strategy, but never label it as a DV update.
                    assert strategy != "unknown", f"MERGE write strategy is unknown: {metrics}"
                else:
                    _assert_dvs(history, command)
            if trial == 0:
                _session(lambda spark: _validate_store_sales(
                    spark, base, seeded_version, paths["CPU"], paths["GPU"],
                    command, STORE_SALES_THRESHOLD))
            if trial > 0:
                for engine in order:
                    measured[engine].append(results[engine]["seconds"])
                paired_speedups.append(results["CPU"]["seconds"] /
                                       results["GPU"]["seconds"])
        cpu_median, gpu_median = (statistics.median(measured[engine])
                                  for engine in ("CPU", "GPU"))
        _emit("summary", case=case, scale=scale, rows=rows,
              input_bytes=input_bytes, target_bytes=detail["sizeInBytes"],
              affected_rows=affected, initial_rows=initial_rows,
              strategies=strategies,
              same_strategy=all(cpu == gpu for cpu, gpu in
                                zip(strategies["CPU"], strategies["GPU"])),
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
    monkeypatch.setitem(CONF, "spark.sql.shuffle.partitions", "8")
    _run(request, MEDIUM, "medium-shuffle8")


def test_dv_dml_benchmark_medium_dense(request, monkeypatch):
    monkeypatch.setitem(CONF, "spark.sql.shuffle.partitions", "8")
    _run(request, MEDIUM_DENSE, "medium-dense-shuffle8")


def test_dv_dml_benchmark_full(request):
    _run(request, FULL, "full")


def test_dv_dml_benchmark_store_sales(request, monkeypatch):
    monkeypatch.setitem(CONF, "spark.sql.shuffle.partitions", "8")
    _run_store_sales(request)


def test_dv_dml_benchmark_store_sales_delete_probe(request, monkeypatch):
    monkeypatch.setitem(CONF, "spark.sql.shuffle.partitions", "8")
    _run_store_sales(request, commands=("DELETE",), repeats=1,
                     scale="store-sales-local-10pct-delete-probe")
