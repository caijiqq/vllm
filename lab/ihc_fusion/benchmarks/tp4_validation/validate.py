# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility entrypoint; correctness is shared with TP8."""

import sys

from lab.ihc_fusion.benchmarks.tp8_validation.validate import main

if __name__ == "__main__":
    sys.exit(main())
