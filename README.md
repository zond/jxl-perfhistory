# jxl-perfhistory

Benchmark [jxl-rs](https://github.com/libjxl/jxl-rs) decoding performance across git revisions.

This tool helps track performance changes over time by building `jxl_cli` for multiple git revisions and measuring decoding speed with statistical confidence.

## Features

- Builds `jxl_cli` for each revision automatically
- Uses Bayesian credible intervals (Student's t-distribution) for statistically sound measurements
- Continues benchmarking until achieving required confidence and error thresholds
- Caches built binaries to speed up repeated benchmarks
- Generates formatted performance reports with graphs

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
| `-f, --file <PATH>` | Path to a JXL file to decode (required) | - |
| `-c, --confidence <F>` | Required confidence interval (0.0-1.0) | 0.95 |
| `-e, --error <F>` | Maximum relative error threshold | 0.05 |
| `-m, --min-measurements <N>` | Minimum measurements per revision | 10 |
| `-b, --binary-directory <PATH>` | Persistent directory for built binaries | temp |
| `-d, --data-directory <DATA_DIRECTORY>` | Directory to store measurement data for resumable benchmarks | temp |

*Note: `-b` and `-d` cannot be the same directory.*

### Examples

Benchmark the last 20 revisions:
```bash
jxl-perfhistory -r 20 -f test.jxl
```

Use a persistent binary cache for faster repeated runs:
```bash
jxl-perfhistory -f test.jxl -b ~/.cache/jxl-perfhistory
```

Higher precision measurements (99% confidence, 2% max error):
```bash
jxl-perfhistory -f test.jxl -c 0.99 -e 0.02
```

## Requirements

- Must be run from within a clean jxl-rs git repository (no uncommitted changes)
- The repository must be on a branch (not in detached HEAD state)
- Rust toolchain for building jxl_cli
	- For Windows, the MSVC toolchain must be used

## Output

The tool produces a report showing:
- Performance statistics (min, max, average, improvement percentage)
- A visual performance graph normalized to the range
- Detailed results table with commit hashes and messages

## License

BSD-3-Clause
