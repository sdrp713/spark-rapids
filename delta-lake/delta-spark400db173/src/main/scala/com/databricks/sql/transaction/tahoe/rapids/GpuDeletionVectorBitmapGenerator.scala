/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.databricks.sql.transaction.tahoe.rapids

import com.databricks.sql.transaction.tahoe.OptimisticTransaction
import com.databricks.sql.transaction.tahoe.actions.{AddFile, FileAction, RemoveFile}
import com.databricks.sql.transaction.tahoe.commands.{
  DeletionVectorBitmapGenerator,
  DeletionVectorResult,
  DMLWithDeletionVectorsHelper,
  TouchedFileWithDV
}
import com.databricks.sql.transaction.tahoe.deletionvectors.{RoaringBitmapArray,
  RoaringBitmapArrayFormat}
import com.databricks.sql.transaction.tahoe.util.{Utils => DeltaUtils}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path

import org.apache.spark.paths.SparkPath
import org.apache.spark.sql.{Column, Encoder, SparkSession}
import org.apache.spark.sql.functions.{broadcast, col, collect_list, count, lit}
import org.apache.spark.sql.nvidia.DFUDFShims

private[rapids] object GpuDeletionVectorBitmapGenerator {
  private val FilePathColumn = "__delta_gpu_dv_file_path"
  private val FileIdColumn = "__delta_gpu_dv_file_id"
  private val RowIndexColumn = "__delta_gpu_dv_row_index"
  private val RowIndexesColumn = "__delta_gpu_dv_row_indexes"
  private val MatchCountColumn = "__delta_gpu_dv_match_count"
  private val DvDataClassName =
    "com.databricks.sql.transaction.tahoe.commands.DeletionVectorData"
  private val DvWriterClassName =
    "com.databricks.sql.transaction.tahoe.commands.DeletionVectorWriter$"

  // DBR's DeletionVectorData signature references a Scala 2.12-only Sizing trait in some
  // compile jars. Resolve this private API at runtime so the Scala 2.13 compiler need not load it.
  private def dvDataEncoder: Encoder[AnyRef] =
    Class.forName(DvDataClassName).getMethod("encoder").invoke(null)
      .asInstanceOf[Encoder[AnyRef]]

  private def newDvData(path: String, existingDv: Option[String], bitmap: Array[Byte],
      cardinality: Long): AnyRef = {
    val constructor = Class.forName(DvDataClassName).getConstructor(
      classOf[String], classOf[Option[_]], classOf[Array[Byte]], java.lang.Long.TYPE)
    constructor.newInstance(path, existingDv, bitmap, Long.box(cardinality))
  }

  private def dvWriter(spark: SparkSession, conf: Configuration, dataPath: Path,
      prefixLength: Int): Iterator[AnyRef] => Iterator[DeletionVectorResult] = {
    val writerClass = Class.forName(DvWriterClassName)
    val module = writerClass.getField("MODULE$").get(null)
    writerClass.getMethod("createMapperToStoreDeletionVectors", classOf[SparkSession],
      classOf[Configuration], classOf[Path], java.lang.Integer.TYPE)
      .invoke(module, spark, conf, dataPath, Int.box(prefixLength))
      .asInstanceOf[Iterator[AnyRef] => Iterator[DeletionVectorResult]]
  }

  /**
   * DELETE can aggregate matched row indexes by a compact file ID on GPU. DBR's legacy bitmap
   * helper groups every matched row by its full path on CPU; retaining the path only in the small
   * file lookup avoids that expensive row-wise string aggregation and GPU-to-CPU transition.
   * The existing DBR writer still merges any old DV and persists the replacement DV.
   */
  def findTouchedFilesForDelete(
      spark: SparkSession,
      txn: OptimisticTransaction,
      hasReadableDVs: Boolean,
      targetScan: GpuTargetScan,
      condition: Column,
      nameToAddFileMap: Map[String, AddFile]): Seq[TouchedFileWithDV] = {
    import spark.implicits._

    val fileMetadata = nameToAddFileMap.toSeq.sortBy(_._1).map { case (path, add) =>
      val existingDv = if (hasReadableDVs) {
        Option(add.deletionVector).map(_.serializeToBase64())
      } else {
        None
      }
      // Match the encoded file-path key used by Delta's existing-DV lookup join.
      (SparkPath.fromPath(new Path(path)).urlEncoded, path, existingDv)
    }.toArray
    require(fileMetadata.map(_._1).distinct.length == fileMetadata.length,
      "Different DELETE candidate files have the same encoded scan path")
    val fileIds = spark.createDataset(fileMetadata.indices.map { id =>
      (fileMetadata(id)._1, id.toLong)
    }).toDF(FilePathColumn, FileIdColumn)

    val gpuTargetDf = DMLWithDeletionVectorsHelperShims.withGpuExecutionContext(
      spark, targetScan.dataFrame)
    val matchedRows = gpuTargetDf.filter(condition).select(
      targetScan.filePathColumn.as(FilePathColumn),
      targetScan.rowIndexColumn.as(RowIndexColumn))
    val rowIndexesByFile = matchedRows
      .join(broadcast(fileIds), Seq(FilePathColumn), "left_outer")
      .groupBy(col(FileIdColumn))
      .agg(
        collect_list(col(RowIndexColumn)).as(RowIndexesColumn),
        count(lit(1)).as(MatchCountColumn))

    // Only one row per touched file crosses back to CPU. Fail rather than silently dropping rows
    // if input_file_name and the candidate-file map disagree, or the scan loses a row index.
    val rowIndexData = rowIndexesByFile.mapPartitions { rows =>
      rows.map { row =>
        require(!row.isNullAt(0), "A matched DELETE file was absent from the candidate-file map")
        val fileId = row.getLong(0)
        require(fileId >= 0 && fileId < fileMetadata.length,
          s"Invalid DELETE file ID: $fileId")
        val indexes = row.getSeq[Long](1)
        require(indexes.size.toLong == row.getLong(2),
          "The DELETE scan produced a null physical row index")
        val bitmap = new RoaringBitmapArray()
        indexes.foreach(bitmap.add)
        bitmap.runOptimize()
        val (_, path, existingDv) = fileMetadata(fileId.toInt)
        newDvData(path, existingDv,
          bitmap.serializeAsByteArray(RoaringBitmapArrayFormat.Portable), bitmap.cardinality)
      }
    }(dvDataEncoder)
    val prefixLength = DeltaUtils.getRandomPrefixLength(txn.metadata)
    val storedResults = rowIndexData.mapPartitions(dvWriter(spark,
      txn.deltaLog.newDeltaHadoopConf(), txn.deltaLog.dataPath, prefixLength))(
      DeletionVectorResult.encoder).collect().toSeq

    DMLWithDeletionVectorsHelper.findFilesWithMatchingRows(
      txn, nameToAddFileMap, storedResults)
  }

  /**
   * Finds candidate files containing matching rows and writes replacement deletion vectors.
   * Scanning and predicate evaluation remain GPU-eligible; DBR's native bitmap helper performs
   * the compact bitmap construction and DV persistence.
   */
  def findTouchedFiles(
      spark: SparkSession,
      txn: OptimisticTransaction,
      hasReadableDVs: Boolean,
      targetScan: GpuTargetScan,
      candidateFiles: Seq[AddFile],
      condition: Column,
      nameToAddFileMap: Map[String, AddFile]): Seq[TouchedFileWithDV] = {
    val gpuTargetDf = DMLWithDeletionVectorsHelperShims.withGpuExecutionContext(
      spark, targetScan.dataFrame)
    val candidatesHaveDVs =
      hasReadableDVs && candidateFiles.exists(_.deletionVector != null)
    val storedResults = DeletionVectorBitmapGenerator
      .buildRowIndexSetsForFilesMatchingCondition(
        spark,
        txn,
        candidatesHaveDVs,
        gpuTargetDf,
        candidateFiles,
        DFUDFShims.columnToExpr(condition),
        Some(targetScan.filePathColumn),
        Some(targetScan.rowIndexColumn))

    DMLWithDeletionVectorsHelper.findFilesWithMatchingRows(
      txn, nameToAddFileMap, storedResults)
  }

  def processUnmodifiedData(
      spark: SparkSession,
      touchedFiles: Seq[TouchedFileWithDV],
      txn: OptimisticTransaction): (Seq[FileAction], Map[String, Long]) = {
    // This helper works on file actions and snapshot statistics, not table rows. Keeping its
    // metadata joins on CPU avoids GPU broadcasts/transfers and CPU JSON bridges for small
    // driver-originated datasets. Bitmap generation above and subsequent DML remain GPU-enabled.
    val (actions, metrics) = GpuDeltaCpuFallback.withRapidsDisabled(spark) {
      DMLWithDeletionVectorsHelper.processUnmodifiedData(spark, touchedFiles, txn.snapshot)
    }
    // DBR rehydrates the replacement AddFile stats from the snapshot, but the paired RemoveFile
    // can retain the numRecords-only stats from the data-skipping candidate. Both actions describe
    // the same logical file state, so carry DBR's native wide-bound stats onto the remove action.
    val addStatsByPath = actions.collect {
      case add: AddFile if add.stats != null => add.path -> add.stats
    }.toMap
    val actionsWithStats = actions.map {
      case remove: RemoveFile if addStatsByPath.contains(remove.path) =>
        remove.copy(stats = addStatsByPath(remove.path))
      case action => action
    }
    (actionsWithStats, metrics)
  }
}
