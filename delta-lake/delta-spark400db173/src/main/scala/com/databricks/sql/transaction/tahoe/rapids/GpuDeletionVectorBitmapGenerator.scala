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
  DMLWithDeletionVectorsHelper,
  TouchedFileWithDV
}

import org.apache.spark.sql.{Column, SparkSession}
import org.apache.spark.sql.nvidia.DFUDFShims

private[rapids] object GpuDeletionVectorBitmapGenerator {

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
    val (actions, metrics) =
      DMLWithDeletionVectorsHelper.processUnmodifiedData(spark, touchedFiles, txn.snapshot)
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
