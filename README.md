# jxl-perfhistory

Benchmark [jxl-rs](https://github.com/libjxl/jxl-rs) decoding performance across git revisions.

This tool helps track performance changes over time by building `jxl_cli` for multiple git revisions and measuring decoding speed with statistical confidence.

## Features

- Builds `jxl_cli` for each revision automatically
- Uses Bayesian credible intervals (order statistics with binomial distribution) for statistically sound measurements
- Continues benchmarking until achieving required confidence and error thresholds
- Caches built binaries to speed up repeated benchmarks
- **Multi-file benchmarking** with glob patterns - compare performance across different image types
- **System noise calibration** - detects noisy systems before wasting time on unreliable measurements
- **Process pausing** - temporarily freeze noisy applications during benchmarks for much more accurate results
- Generates formatted performance reports with ASCII graphs

## Installation

```bash
cargo install --path .
```

Or build from source:

```bash
cargo build --release
```

## Usage

Run from within the jxl-rs repository directory:

```bash
jxl-perfhistory -f /path/to/image.jxl
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-r, --revisions <N>` | Number of historical revisions to benchmark | 10 |
| `-f, --file <PATH>` | Path to a JXL file to decode | - |
| `-g, --glob <PATTERN>` | Glob pattern for multiple files (e.g., `"testdata/*.jxl"`) | - |
| `-c, --confidence <F>` | Required confidence interval (0.0-1.0) | 0.95 |
| `-e, --error <F>` | Maximum relative error threshold | 0.05 |
| `-m, --min-measurements <N>` | Minimum measurements per revision | 10 |
| `-b, --binary-directory <PATH>` | Persistent directory for built binaries | temp |
| `-d, --data-directory <PATH>` | Directory for persistent measurements (resumable) | - |
| `--pause-processes <LIST>` | Comma-separated process names to pause during benchmarks | - |
| `--ignore-noisy-system` | Continue even if system appears too noisy | false |

### Reducing Measurement Noise

**The `--pause-processes` option is highly recommended for accurate benchmarks!**

Background processes like web browsers can cause significant measurement noise. Chrome alone can cause 10-20% variance in benchmark results due to its background activity.

```bash
# Pause Chrome and Firefox during benchmarks
jxl-perfhistory -f test.jxl --pause-processes chrome,firefox,slack

# The tool will:
# 1. Send SIGSTOP to pause the processes
# 2. Run all benchmarks
# 3. Send SIGCONT to resume the processes (even if benchmark fails or is interrupted)
```

This uses SIGSTOP/SIGCONT signals, so paused applications will freeze completely (no CPU usage, no network activity) and resume exactly where they left off.

### Examples

Benchmark the last 20 revisions:
```bash
jxl-perfhistory -r 20 -f test.jxl
```

**Multi-file comparison** (great for detecting regressions that affect specific image types):
```bash
jxl-perfhistory -g "testdata/*.jxl" -r 10 --pause-processes chrome
```

Use a persistent binary cache for faster repeated runs:
```bash
jxl-perfhistory -f test.jxl -b ~/.cache/jxl-perfhistory
```

Resume an interrupted benchmark:
```bash
jxl-perfhistory -f test.jxl -d ~/.cache/jxl-perfhistory-data
```

Higher precision measurements (99% confidence, 2% max error):
```bash
jxl-perfhistory -f test.jxl -c 0.99 -e 0.02
```

## Requirements

- Must be run from within a clean jxl-rs git repository (no uncommitted changes)
- The repository must be on a branch (not in detached HEAD state)
- Rust toolchain for building jxl_cli
- Linux (for `--pause-processes` and load average detection)

## Output

### Single-file mode

Shows absolute performance over time:
- Performance statistics (min, max, average, improvement percentage)
- A visual performance graph with credible intervals
- Detailed results table with commit hashes and messages

### Multi-file mode

Shows relative performance changes between revisions:
- Each revision compared to its predecessor
- Speedup/slowdown ratios with confidence intervals
- Statistically significant changes highlighted (when the CI doesn't include 1.0)

## License

BSD-3-Clause
