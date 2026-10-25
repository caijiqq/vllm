# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility entrypoint; model benchmarking is shared with TP8."""

from lab.ihc_fusion.benchmarks.tp8_validation.benchmark import main

if __name__ == "__main__":
    main()
