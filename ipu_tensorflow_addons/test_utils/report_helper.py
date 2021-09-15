# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import os
import pathlib
import shutil
import tempfile
from pva import ProgramVisitor

class ReportHelper():
  """ ReportHelper creates a temporary directory for reports to be generated
  in. `set_autoreport_options` configures poplar to use this directory.

  Reports are generated in unique subdirectories of the temporary directory and
  can be found by calling `find_report` or `find_reports`.

  Reports can also be cleared by calling `clear_reports`.

  All files are automatically cleaned up when the object is destroyed.
  """
  def __init__(self):
    self._directory = tempfile.mkdtemp(prefix=f"tf_")
    # Used to give a better error message if no reports were generated because
    # you forgot to call set_autoreport_options.
    self._set_options_called = False

  def _find_report_subdirectories(self):
    # Find all subdirectories in the report directory.
    directory = pathlib.Path(self._directory)
    if not directory.exists():
      if not self._set_options_called:
        raise RuntimeError("To use this helper you must setup the poplar " +
                           "autoReport options with set_autoreport_options.")
      raise IOError(
          f"Report directory does not exist: {self._directory}\nEither " +
          "no reports have been generated or the directory was deleted.")
    return directory.glob('tf_report_*/')

  def _find_report_files_in_subdirectory(self, directory):
    return directory.glob("*.pop")

  def set_autoreport_options(self,
                             cfg,
                             *,
                             output_graph_profile=True,
                             output_execution_profile=False,
                             max_execution_reports=1000):
    """Sets autoReport engine options in the IPUConfig.

    Set outputExecutionProfile to True to allow execution reports to be
    generated.

    If execution reports are enabled, max_execution_reports controls the
    maximum number of executions included in a report.
    """
    self._set_options_called = True
    options = {
        "autoReport.directory": self._directory,
        "autoReport.outputGraphProfile": str(output_graph_profile).lower(),
        "autoReport.outputExecutionProfile":
        str(output_execution_profile).lower(),
        "autoReport.executionProfileProgramRunCount":
        str(max_execution_reports),
    }
    cfg.compilation_poplar_options = options
    cfg._profiling.auto_assign_report_subdirectories = True  # pylint: disable=protected-access

  def find_reports(self):
    """Finds and returns the paths to generated report files in order of
    creation time (oldest first).
    """
    paths = []
    for d in self._find_report_subdirectories():
      files_ = list(self._find_report_files_in_subdirectory(d))
      # Only expect 1 report file per report subdirectory.
      if len(files_) != 1:
        raise IOError(f"Expected 1 report file in each report " +
                      f"subdirectory but found {len(files_)} in {d}:" +
                      "".join(f"\n   {f.name}" for f in files_))
      # Add report file absolute path to result.
      paths.append(str(files_[0]))

    # Sort by oldest first
    paths.sort(key=lambda p: os.stat(p).st_ctime)
    return paths

  def find_report(self):
    """Finds and returns the paths to the generated report file.
    Asserts the only one report has been generated.
    """
    reports = self.find_reports()
    num_reports = len(reports)
    assert num_reports == 1, f"Expected 1 report but found {num_reports}"
    return reports[0]

  def clear_reports(self):
    """Clears all existing reports and their subdirectories."""
    # Remove the whole directory and recreate it rather than removing each
    # subdirectory individually.
    shutil.rmtree(self._directory)
    os.mkdir(self._directory)

  # Automatically clean up all files when this instance is destroyed.
  def __del__(self):
    # Ignore errors to clean up as much as possible.
    shutil.rmtree(self._directory, ignore_errors=True)
