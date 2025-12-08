// Copyright (c) 2025 the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use clap::Parser;
use color_eyre::eyre::{Result, eyre};
use git2::{Oid, Repository};
use glob::glob;
use memmap2::Mmap;
use rand::Rng;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::result;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};
use tempfile::TempDir;
use thiserror::Error;

/// Global registry of cleanup functions for Ctrl-C handler.
/// Uses HashMap with atomic counter to avoid unbounded growth and ensure proper cleanup.
type CleanupFn = Box<dyn FnMut() + Send>;
static CLEANUP_REGISTRY: LazyLock<Mutex<HashMap<usize, CleanupFn>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static CLEANUP_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn register_cleanup(f: CleanupFn) -> usize {
    let id = CLEANUP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut registry = CLEANUP_REGISTRY.lock().unwrap();
    registry.insert(id, f);
    id
}

fn unregister_cleanup(id: usize) {
    let mut registry = CLEANUP_REGISTRY.lock().unwrap();
    registry.remove(&id);
}

fn setup_ctrlc_handler() {
    ctrlc::set_handler(move || {
        eprintln!("\nInterrupted, cleaning up...");
        // Use try_lock to avoid panic if mutex is poisoned or held by another thread
        if let Ok(mut registry) = CLEANUP_REGISTRY.try_lock() {
            for cleanup_fn in registry.values_mut() {
                cleanup_fn();
            }
        } else {
            eprintln!("Warning: Could not acquire cleanup lock, some cleanup may be skipped");
        }
        std::process::exit(130); // Standard exit code for Ctrl-C
    })
    .expect("Error setting Ctrl-C handler");
}

/// Safe comparison for f64 values that handles NaN gracefully.
/// NaN values are treated as greater than all other values.
fn f64_cmp(a: &f64, b: &f64) -> std::cmp::Ordering {
    a.partial_cmp(b).unwrap_or_else(|| {
        // Handle NaN: treat NaN as greater than everything
        match (a.is_nan(), b.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => unreachable!(), // partial_cmp only fails for NaN
        }
    })
}

#[derive(Parser, Debug)]
#[command(name = "jxl-perfhistory")]
#[command(about = "Benchmark jxl_cli across git revisions", long_about = None)]
struct Args {
    /// Number of revisions to go back from HEAD
    #[arg(short = 'r', long = "revisions", default_value = "10")]
    revisions: usize,

    /// Path to the JXL file to decode (mutually exclusive with --glob)
    #[arg(short = 'f', long = "file", conflicts_with = "glob_pattern")]
    jxl_file: Option<String>,

    /// Glob pattern to select multiple JXL files (mutually exclusive with --file)
    #[arg(short = 'g', long = "glob", conflicts_with = "jxl_file")]
    glob_pattern: Option<String>,

    /// Required confidence interval for measured decoding speed
    #[arg(short = 'c', long = "confidence", default_value = "0.95")]
    confidence: f64,

    /// Maximum relative error for measured decoding speed
    #[arg(short = 'e', long = "error", default_value = "0.05")]
    rel_error: f64,

    /// Minimum number of measurements per revision
    #[arg(short = 'm', long = "min-measurements", default_value = "10")]
    min_measurements: usize,

    /// Persistent directory to put the built binaries in - a temporary directory will be used if not provided
    #[arg(short = 'b', long = "binary-directory")]
    binary_directory: Option<String>,

    /// Directory to store measurement data for resumable benchmarks
    #[arg(short = 'd', long = "data-directory")]
    data_directory: Option<String>,

    /// Continue benchmarking even if system appears noisy
    #[arg(long = "ignore-noisy-system")]
    ignore_noisy_system: bool,

    /// Comma-separated list of process names to pause during benchmarking (e.g., "chrome,firefox")
    #[arg(long = "pause-processes", value_delimiter = ',')]
    pause_processes: Vec<String>,
}

/// System noise metrics collected before benchmarking
struct NoiseMetrics {
    load_average: Option<f64>,
    calibration_cv: Option<f64>,
}

impl NoiseMetrics {
    /// Check system load and run calibration benchmarks
    fn measure(n_calibration_samples: usize) -> Self {
        let load_average = Self::get_load_average();
        let calibration_cv = Self::run_calibration(n_calibration_samples);
        Self {
            load_average,
            calibration_cv: Some(calibration_cv),
        }
    }

    /// Read system load average from /proc/loadavg (Linux only)
    fn get_load_average() -> Option<f64> {
        std::fs::read_to_string("/proc/loadavg")
            .ok()
            .and_then(|content| {
                content
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<f64>().ok())
            })
    }

    /// Run a CPU+memory bound calibration to measure system noise.
    /// Returns coefficient of variation (CV = stddev / mean).
    fn run_calibration(n_samples: usize) -> f64 {
        use std::time::Instant;

        let mut measurements = Vec::with_capacity(n_samples);

        // Allocate a 16MB buffer for memory-bound work
        const BUFFER_SIZE: usize = 16 * 1024 * 1024;
        let mut buffer = vec![0u8; BUFFER_SIZE];

        for i in 0..n_samples {
            // Fill buffer with varying data to prevent optimization
            for (j, byte) in buffer.iter_mut().enumerate() {
                *byte = ((i + j) & 0xFF) as u8;
            }

            let start = Instant::now();

            // Hash the buffer multiple times for ~100ms of work
            let mut hasher = Sha256::new();
            for _ in 0..10 {
                hasher.update(&buffer);
            }
            let _ = hasher.finalize();

            let elapsed = start.elapsed().as_secs_f64();
            measurements.push(elapsed);
        }

        // Compute CV = stddev / mean
        let n = measurements.len();
        if n < 2 {
            return 0.0; // Can't compute variance with fewer than 2 samples
        }
        let n_f = n as f64;
        let mean = measurements.iter().sum::<f64>() / n_f;
        let variance = measurements.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n_f - 1.0);
        let stddev = variance.sqrt();
        stddev / mean
    }

    fn is_noisy(&self) -> bool {
        // Consider system noisy if load > 1.0 or CV > 5%
        self.load_average.is_some_and(|l| l > 1.0)
            || self.calibration_cv.is_some_and(|cv| cv > 0.05)
    }

    fn warning_message(&self) -> Option<String> {
        if !self.is_noisy() {
            return None;
        }

        let mut warnings = vec![];
        if let Some(load) = self.load_average
            && load > 1.0
        {
            warnings.push(format!("high system load ({:.2})", load));
        }
        if let Some(cv) = self.calibration_cv
            && cv > 0.05
        {
            warnings.push(format!("high calibration variance (CV={:.1}%)", cv * 100.0));
        }

        if warnings.is_empty() {
            None
        } else {
            Some(format!(
                "WARNING: System appears noisy: {}. Results may be unreliable.",
                warnings.join(", ")
            ))
        }
    }
}

/// Validate that a process name contains only safe characters
fn validate_process_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(eyre!("Process name cannot be empty"));
    }
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.')
    {
        return Err(eyre!(
            "Invalid process name '{}': only alphanumeric, '_', '-', and '.' allowed",
            name
        ));
    }
    Ok(())
}

/// Unpause processes by sending SIGCONT
fn unpause_processes(processes: &[String]) {
    if processes.is_empty() {
        return;
    }
    eprintln!("Resuming processes: {}", processes.join(", "));
    for name in processes {
        let output = Command::new("killall").args(["-SIGCONT", name]).output();
        if let Ok(output) = output
            && !output.status.success()
        {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.contains("no process found") {
                eprintln!("Warning: failed to resume {}: {}", name, stderr.trim());
            }
        }
    }
}

/// RAII guard to ensure processes are resumed even on panic/error
struct ProcessPauser {
    processes: Vec<String>,
    cleanup_id: Option<usize>,
}

impl ProcessPauser {
    fn new(processes: Vec<String>) -> Result<Self> {
        // Validate all process names before doing anything
        for name in &processes {
            validate_process_name(name)?;
        }

        if !processes.is_empty() {
            eprintln!("Pausing processes: {}", processes.join(", "));
            for name in &processes {
                let output = Command::new("killall").args(["-SIGSTOP", name]).output()?;
                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    if !stderr.contains("no process found") {
                        eprintln!("Warning: failed to pause {}: {}", name, stderr.trim());
                    }
                }
            }
        }

        // Register cleanup closure
        let cleanup_id = if !processes.is_empty() {
            let procs = processes.clone();
            Some(register_cleanup(Box::new(move || {
                unpause_processes(&procs);
            })))
        } else {
            None
        };

        Ok(Self {
            processes,
            cleanup_id,
        })
    }

    fn resume(&mut self) -> Result<()> {
        unpause_processes(&self.processes);
        self.processes.clear();
        // Unregister from cleanup registry
        if let Some(id) = self.cleanup_id.take() {
            unregister_cleanup(id);
        }
        Ok(())
    }
}

impl Drop for ProcessPauser {
    fn drop(&mut self) {
        if !self.processes.is_empty() {
            eprintln!("Warning: ProcessPauser dropped without explicit resume, resuming now...");
            unpause_processes(&self.processes);
        }
        if let Some(id) = self.cleanup_id.take() {
            unregister_cleanup(id);
        }
    }
}

/// RAII guard to ensure git repo is restored to original branch even on panic/error.
///
/// Note: There's a small race window where if Ctrl-C fires during restore(), both the
/// explicit restore and the cleanup handler might run. This is harmless since restoring
/// to the same commit is idempotent.
struct RepoGuard {
    original_ref_name: Option<String>,
    cleanup_id: Option<usize>,
}

impl RepoGuard {
    fn new(original_ref_name: String) -> Self {
        // Register cleanup closure for Ctrl-C handler
        let ref_name = original_ref_name.clone();
        let cleanup_id = Some(register_cleanup(Box::new(move || {
            eprintln!("Restoring repo to: {}", ref_name);
            if let Ok(repo) = Repository::open(".") {
                let _ = restore_repo(&repo, ref_name.clone());
            }
        })));
        Self {
            original_ref_name: Some(original_ref_name),
            cleanup_id,
        }
    }

    /// Explicitly restore the repo to the original branch.
    /// Unregisters from cleanup registry to prevent double-restore on Ctrl-C.
    fn restore(&mut self) -> Result<()> {
        // Unregister from cleanup registry FIRST to minimize race window
        if let Some(id) = self.cleanup_id.take() {
            unregister_cleanup(id);
        }
        if let Some(ref original_ref_name) = self.original_ref_name {
            let repo = Repository::open(".")?;
            restore_repo(&repo, original_ref_name.clone())?;
            self.original_ref_name = None;
        }
        Ok(())
    }
}

impl Drop for RepoGuard {
    fn drop(&mut self) {
        // Safety net: try to restore if not already done
        if let Some(ref original_ref_name) = self.original_ref_name {
            eprintln!("Warning: RepoGuard dropped without explicit restore, restoring now...");
            if let Ok(repo) = Repository::open(".") {
                let _ = restore_repo(&repo, original_ref_name.clone());
            }
        }
        // Unregister from cleanup registry
        if let Some(id) = self.cleanup_id.take() {
            unregister_cleanup(id);
        }
    }
}

/// Handle for a measurement file that stays open for appending.
/// Contains both the file handle and the in-memory measurements.
struct MeasurementFile {
    file: Option<File>,
    measurements: Vec<f64>,
}

impl MeasurementFile {
    /// Open a measurement file, loading existing measurements.
    /// The file stays open for appending new measurements.
    fn open(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .create(true)
            .append(true)
            .open(path)?;

        let metadata = file.metadata()?;
        let len = metadata.len() as usize;

        let measurements = if len == 0 {
            Vec::new()
        } else {
            if !len.is_multiple_of(std::mem::size_of::<f64>()) {
                return Err(eyre!(
                    "Measurement file {} has invalid size {} (not a multiple of 8)",
                    path.display(),
                    len
                ));
            }

            // SAFETY:
            // - File size is validated to be a multiple of 8 bytes (lines 354-363)
            // - The file contains only f64 values written by append() in native endian
            // - f64 has natural alignment of 8 bytes, satisfied by mmap page alignment
            // - We use a read-only mmap and immediately copy data to Vec, avoiding TOCTOU
            // - Endianness is consistent: we write and read using native endian
            let mmap = unsafe { Mmap::map(&file)? };
            let bytes: &[f64] = bytemuck::cast_slice(&mmap);

            // Validate loaded measurements
            for (i, &val) in bytes.iter().enumerate() {
                if !val.is_finite() {
                    return Err(eyre!(
                        "Invalid measurement at index {} in {}: {} (expected finite number)",
                        i,
                        path.display(),
                        val
                    ));
                }
                if val <= 0.0 {
                    return Err(eyre!(
                        "Invalid measurement at index {} in {}: {} (expected positive number)",
                        i,
                        path.display(),
                        val
                    ));
                }
            }

            bytes.to_vec()
        };

        Ok(Self {
            file: Some(file),
            measurements,
        })
    }

    /// Append a measurement to both the file and the in-memory vector.
    fn append(&mut self, value: f64) -> Result<()> {
        if let Some(ref mut file) = self.file {
            file.write_all(&value.to_ne_bytes())?;
            file.flush()?;
        }
        self.measurements.push(value);
        Ok(())
    }

    /// Close the file handle explicitly, flushing any pending data.
    fn close(&mut self) {
        if let Some(ref mut file) = self.file {
            let _ = file.flush(); // Best effort flush before closing
        }
        self.file = None;
    }
}

struct MedianIndices {
    target_confidence: f64,
    mapping: HashMap<usize, (usize, usize, f64)>,
}

impl MedianIndices {
    fn new(confidence: f64) -> Result<MedianIndices> {
        if !(0f64..=1f64).contains(&confidence) {
            return Err(eyre!("Can't compute with confidence {}", confidence));
        }
        Ok(MedianIndices {
            target_confidence: confidence,
            mapping: HashMap::new(),
        })
    }
    /// Compute the (lo, hi) indices for a median confidence interval using exact binomial probabilities.
    ///
    /// Uses Pascal's triangle to compute binomial coefficients exactly,
    /// then finds the smallest symmetric interval around n/2 that achieves the target confidence.
    ///
    /// Returns (lo, hi, actual_confidence) where the CI is [sorted[lo], sorted[hi]].
    fn get(&mut self, n: usize) -> (usize, usize, f64) {
        if let Some(result) = self.mapping.get(&n) {
            return *result;
        }
        // Build row n of Pascal's triangle: coeffs[k] = C(n, k)
        // Using the recurrence: C(n, k) = C(n, k-1) * (n - k + 1) / k
        // Note: u128 can handle binomial coefficients up to about n=130 without overflow
        let mut coeffs = vec![0u128; n + 1];
        coeffs[0] = 1;
        for k in 1..=n {
            // Use saturating arithmetic to avoid overflow panic in release mode
            // If overflow occurs, coefficients will saturate at u128::MAX, giving
            // conservative (wider) confidence intervals
            coeffs[k] = coeffs[k - 1]
                .saturating_mul((n - k + 1) as u128)
                .saturating_div(k as u128);
        }

        // Total is 2^n (sum of all binomial coefficients)
        let total: u128 = coeffs.iter().sum();

        // Compute cumulative sums: cdf[k] = sum of coeffs[0..=k] = C(n,0) + ... + C(n,k)
        let mut cdf = vec![0u128; n + 1];
        cdf[0] = coeffs[0];
        for k in 1..=n {
            cdf[k] = cdf[k - 1] + coeffs[k];
        }

        // Find smallest symmetric interval [lo, hi] around n/2 with confidence >= target
        // Confidence of [lo, hi] = P(lo < W <= hi) = (cdf[hi] - cdf[lo]) / total
        let center = n / 2;
        let mut lo = center;
        let mut hi = center;

        loop {
            // P(lo < W <= hi) = (cdf[hi] - cdf[lo]) / total
            let confidence = (cdf[hi] - cdf[lo]) as f64 / total as f64;

            if confidence >= self.target_confidence || (lo == 0 && hi == n - 1) {
                self.mapping.insert(n, (lo, hi, confidence));
                return (lo, hi, confidence);
            }

            // Expand symmetrically
            lo = lo.saturating_sub(1);
            if hi < n - 1 {
                hi += 1;
            }
        }
    }
}

#[derive(Error, Debug)]
#[error("Need at least 3 measurements, got {0}")]
struct InsufficientSamples(usize);

/// Compute hash of file path for stable storage naming
fn file_path_hash(path: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(path.as_bytes());
    let result = hasher.finalize();
    format!("{:x}", result)[..16].to_string()
}

/// Measurements and statistics for a single file within a revision
struct FileResult {
    file_path: String,
    measurement_file: MeasurementFile,
    median: Option<f64>,
    rel_error: Option<f64>,
    /// Number of repetitions to use for benchmarking (calibrated on first run)
    num_reps: Option<u32>,
}

impl FileResult {
    fn new(file_path: String, measurement_file: MeasurementFile) -> Self {
        Self {
            file_path,
            measurement_file,
            median: None,
            rel_error: None,
            num_reps: None,
        }
    }

    fn n_measurements(&self) -> usize {
        self.measurement_file.measurements.len()
    }

    /// Compute credible interval for median using order statistics.
    fn compute_median(
        &mut self,
        mi: &mut MedianIndices,
    ) -> result::Result<(), InsufficientSamples> {
        let measurements = &self.measurement_file.measurements;
        let n = measurements.len();
        if n < 3 {
            return Err(InsufficientSamples(n));
        }

        let mut sorted = measurements.clone();
        sorted.sort_by(f64_cmp);

        let median = if n.is_multiple_of(2) {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        let (lo, hi, _actual_confidence) = mi.get(n);
        let ci_lower = sorted[lo];
        let ci_upper = sorted[hi];
        let half_width = (ci_upper - ci_lower) / 2.0;

        self.median = Some(median);
        self.rel_error = Some(half_width / median.abs());
        Ok(())
    }

    fn close(&mut self) {
        self.measurement_file.close();
    }
}

struct Revision {
    oid: Oid,
    summary: String,
    binary_path: Option<String>,
    file_results: Vec<FileResult>,
    ordinal: usize,
}

const TARGET_BENCHMARK_SECS: f64 = 0.5;

impl Revision {
    /// Run jxl_cli benchmark and return pixels/s
    fn run_benchmark(&self, file_path: &str, num_reps: u32) -> Result<f64> {
        let output = Command::new(self.binary_path.as_ref().unwrap())
            .arg(file_path)
            .args(["--speedtest", "--num-reps", &num_reps.to_string()])
            .output()?;
        if !output.status.success() {
            return Err(eyre!(
                "Benchmark failed for {:.8}!\n{}",
                self.oid,
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        stdout
            .lines()
            .find(|line| line.contains("pixels/s") && line.contains("Decoded"))
            .and_then(|line| line.split_whitespace().rev().nth(1))
            .ok_or_else(|| eyre!("Can't find decoding speed in `{}`", stdout))?
            .parse()
            .map_err(|e| eyre!("Can't parse decoding speed: {}", e))
    }

    /// Benchmark a specific file and append the result to its measurements.
    /// On first run, calibrates num_reps to achieve TARGET_BENCHMARK_SECS.
    pub fn benchmark(&mut self, file_idx: usize) -> Result<()> {
        use std::time::Instant;

        let file_path = self.file_results[file_idx].file_path.clone();
        let num_reps = self.file_results[file_idx].num_reps;

        // Calibrate num_reps on first measurement
        if num_reps.is_none() {
            let start = Instant::now();
            let pixels_per_sec = self.run_benchmark(&file_path, 1)?;
            let elapsed = start.elapsed().as_secs_f64();

            if elapsed < TARGET_BENCHMARK_SECS {
                let needed_reps = (TARGET_BENCHMARK_SECS / elapsed).ceil() as u32;
                self.file_results[file_idx].num_reps = Some(needed_reps);
                // Redo measurement with correct num_reps
                let pixels_per_sec = self.run_benchmark(&file_path, needed_reps)?;
                self.file_results[file_idx]
                    .measurement_file
                    .append(pixels_per_sec)?;
            } else {
                self.file_results[file_idx].num_reps = Some(1);
                self.file_results[file_idx]
                    .measurement_file
                    .append(pixels_per_sec)?;
            }
        } else {
            let pixels_per_sec = self.run_benchmark(&file_path, num_reps.unwrap())?;
            self.file_results[file_idx]
                .measurement_file
                .append(pixels_per_sec)?;
        }

        Ok(())
    }

    pub fn build(&mut self, binary_dir: &Path) -> Result<()> {
        let binary_path = binary_dir.join(self.oid.to_string());
        self.binary_path = Some(binary_path.to_string_lossy().to_string());

        if binary_path.exists() {
            return Ok(());
        }

        let build = Command::new("cargo")
            .args([
                "build",
                "--release",
                "--package",
                "jxl_cli",
                "--bin",
                "jxl_cli",
            ])
            .output()?;

        if !build.status.success() {
            return Err(eyre!(
                "Build failed for {:.8}!\n{}",
                self.oid,
                String::from_utf8_lossy(&build.stderr)
            ));
        }

        fs::copy(Path::new("target/release/jxl_cli"), &binary_path)?;

        Ok(())
    }
    fn clipped_summary(&self, len: usize) -> String {
        if self.summary.len() > len {
            format!("{}...", &self.summary[..(len - 3)])
        } else {
            self.summary.clone()
        }
    }
}

fn collect_revisions(
    repo: &Repository,
    count: usize,
    data_dir: Option<&Path>,
    files: &[String],
) -> Result<Vec<Revision>> {
    let mut revwalk = repo.revwalk()?;
    revwalk.push_head()?;

    let mut ordinal = 0;
    revwalk
        .take(count)
        .map(|oid| {
            let oid = oid?;
            ordinal += 1;

            let file_results: Result<Vec<FileResult>> = files
                .iter()
                .map(|file_path| {
                    let measurement_file = match data_dir {
                        Some(dir) => {
                            // Use hash-based naming for multi-file, plain oid for single file
                            let file_name = if files.len() == 1 {
                                oid.to_string()
                            } else {
                                format!("{}-{}", oid, file_path_hash(file_path))
                            };
                            MeasurementFile::open(&dir.join(file_name))?
                        }
                        None => MeasurementFile {
                            file: None,
                            measurements: Vec::new(),
                        },
                    };
                    Ok(FileResult::new(file_path.clone(), measurement_file))
                })
                .collect();

            Ok(Revision {
                oid,
                summary: repo
                    .find_commit(oid)?
                    .summary()
                    .unwrap_or("(no message)")
                    .to_string(),
                binary_path: None,
                file_results: file_results?,
                ordinal,
            })
        })
        .collect()
}

fn checkout_revision(repo: &Repository, oid: Oid) -> Result<()> {
    let commit = repo.find_commit(oid)?;

    let mut opts = git2::build::CheckoutBuilder::new();
    opts.safe(); // Don't overwrite modified files or remove untracked files

    repo.checkout_tree(commit.as_object(), Some(&mut opts))?;
    repo.set_head_detached(oid)?;

    Ok(())
}

fn verify_repo(repo: &Repository) -> Result<String> {
    let mut status_opts = git2::StatusOptions::new();
    status_opts.include_untracked(false);
    let statuses = repo.statuses(Some(&mut status_opts))?;
    if !statuses.is_empty() {
        return Err(eyre!(
            "Working directory has uncommitted changes. Please commit or stash them first."
        ));
    }

    // Save original HEAD state (branch or detached commit)
    let head = repo.head()?;
    if head.is_branch() {
        head.name()
            .ok_or(eyre!(
                "Working directory doesn't have a name, won't be able to restore it properly."
            ))
            .map(|s| s.into())
    } else {
        // Detached HEAD - save the commit hash
        let oid = head.target().ok_or(eyre!(
            "Detached HEAD doesn't point to a commit, won't be able to restore it properly."
        ))?;
        Ok(oid.to_string())
    }
}

fn restore_repo(repo: &Repository, original_ref: String) -> Result<()> {
    // Try parsing as OID first (handles SHA-1, SHA-256, and short hashes)
    // If that fails, treat it as a branch ref
    if let Ok(oid) = Oid::from_str(&original_ref) {
        // Detached HEAD - restore to specific commit
        repo.set_head_detached(oid)?;
    } else {
        // Branch ref
        repo.set_head(&original_ref)?;
    }
    let mut opts = git2::build::CheckoutBuilder::new();
    opts.force();
    repo.checkout_head(Some(&mut opts))?;
    Ok(())
}

/// Check if a file result is finished (has enough measurements with acceptable error)
fn is_file_result_finished(fr: &FileResult, min_measurements: usize, max_rel_error: f64) -> bool {
    fr.n_measurements() >= min_measurements
        && fr.median.is_some()
        && fr.rel_error.is_some_and(|e| e <= max_rel_error)
}

/// Check if all file results in a revision are finished
fn is_revision_finished(rev: &Revision, min_measurements: usize, max_rel_error: f64) -> bool {
    rev.file_results
        .iter()
        .all(|fr| is_file_result_finished(fr, min_measurements, max_rel_error))
}

/// Find unfinished (revision_idx, file_idx) pairs
fn find_unfinished_pairs(
    revisions: &[Revision],
    min_measurements: usize,
    max_rel_error: f64,
) -> Vec<(usize, usize)> {
    let mut pairs = vec![];
    for (rev_idx, rev) in revisions.iter().enumerate() {
        for (file_idx, fr) in rev.file_results.iter().enumerate() {
            if !is_file_result_finished(fr, min_measurements, max_rel_error) {
                pairs.push((rev_idx, file_idx));
            }
        }
    }
    pairs
}

fn main() -> Result<()> {
    color_eyre::install()?;
    setup_ctrlc_handler();
    let args = Args::parse();

    // Parse file list from --file or --glob
    let files: Vec<String> = if let Some(ref pattern) = args.glob_pattern {
        let paths: Vec<_> = glob(pattern)?
            .filter_map(|p| p.ok())
            .map(|p| p.to_string_lossy().to_string())
            .collect();
        if paths.is_empty() {
            return Err(eyre!("No files matched the glob pattern: {}", pattern));
        }
        paths
    } else if let Some(ref file) = args.jxl_file {
        vec![file.clone()]
    } else {
        return Err(eyre!("Either --file or --glob must be provided"));
    };

    let is_multi_file = files.len() > 1;

    let mut rng = rand::rng();
    let tmp_dir = TempDir::new()?;
    let binary_dir = match &args.binary_directory {
        Some(s) => Path::new(s),
        None => tmp_dir.path(),
    };

    let repo = Repository::open(".")?;
    let original_ref_name = verify_repo(&repo)?;
    let mut mi = MedianIndices::new(args.confidence)?;

    // Check system noise before starting
    eprint!("Calibrating system noise...");
    let noise_metrics = NoiseMetrics::measure(10);
    if let (Some(load), Some(cv)) = (noise_metrics.load_average, noise_metrics.calibration_cv) {
        eprintln!("done (load: {:.2}, CV: {:.1}%)", load, cv * 100.0);
    } else if let Some(cv) = noise_metrics.calibration_cv {
        eprintln!("done (CV: {:.1}%)", cv * 100.0);
    } else {
        eprintln!("done");
    }

    if noise_metrics.is_noisy() && !args.ignore_noisy_system {
        if let Some(warning) = noise_metrics.warning_message() {
            eprintln!("{}", warning);
        }
        return Err(eyre!(
            "System is too noisy for reliable benchmarks. Use --ignore-noisy-system to override."
        ));
    }

    let mut revisions = collect_revisions(
        &repo,
        args.revisions,
        args.data_directory.as_deref().map(Path::new),
        &files,
    )?;

    // Compute medians for any file results that have enough measurements
    for rev in &mut revisions {
        for fr in &mut rev.file_results {
            let _ = fr.compute_median(&mut mi);
        }
    }

    // Report status of loaded data
    for rev in &revisions {
        let fr = &rev.file_results[0];
        let n = fr.n_measurements();
        if n > 0 {
            if is_revision_finished(rev, args.min_measurements, args.rel_error) {
                eprintln!(
                    "{:.8}: {:<50} is already done, {} samples, median/error: {:.2}/{:.4}",
                    rev.oid,
                    rev.clipped_summary(50),
                    n,
                    fr.median.unwrap(),
                    fr.rel_error.unwrap()
                );
            } else {
                eprintln!(
                    "{:.8}: {:<50} loaded {} cached samples (needs more)",
                    rev.oid,
                    rev.clipped_summary(50),
                    n
                );
            }
        }
    }

    // RAII guard ensures repo is restored even on error/panic
    let mut repo_guard = RepoGuard::new(original_ref_name.clone());

    // Build binaries for revisions that need benchmarking
    for rev in revisions
        .iter_mut()
        .filter(|rev| !is_revision_finished(rev, args.min_measurements, args.rel_error))
    {
        checkout_revision(&repo, rev.oid)?;
        eprint!("Building {}: {}...", rev.oid, rev.summary);
        rev.build(binary_dir)?;
        eprintln!("done!");
    }

    // Restore repo (explicit for error handling, guard is safety net)
    repo_guard.restore()?;

    // RAII guard ensures processes are resumed even on error/panic
    let mut process_pauser = ProcessPauser::new(args.pause_processes.clone())?;

    // Benchmark loop: randomly sample unfinished (revision, file) pairs
    loop {
        let unfinished_pairs =
            find_unfinished_pairs(&revisions, args.min_measurements, args.rel_error);
        if unfinished_pairs.is_empty() {
            break;
        }

        let pair_idx = rng.random_range(0..unfinished_pairs.len());
        let (rev_idx, file_idx) = unfinished_pairs[pair_idx];

        let rev = &mut revisions[rev_idx];
        let fr = &rev.file_results[file_idx];
        if is_multi_file {
            eprint!(
                "Benchmarking {:.8} [{}]...",
                rev.oid,
                Path::new(&fr.file_path)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
            );
        } else {
            eprint!(
                "Benchmarking {:.8}: {:<50}...",
                rev.oid,
                rev.clipped_summary(50)
            );
        }

        rev.benchmark(file_idx)?;
        let fr = &mut rev.file_results[file_idx];
        let old_relative_error = fr.rel_error;

        if let Err(InsufficientSamples(_)) = fr.compute_median(&mut mi) {
            eprintln!("done!");
        } else {
            eprint!(
                "done! {} samples, median {:.2}",
                fr.n_measurements(),
                fr.median.unwrap()
            );
            match old_relative_error {
                None => eprint!(", error {:.4}", fr.rel_error.unwrap()),
                Some(old) => eprint!(", error {:.4} => {:.4}", old, fr.rel_error.unwrap()),
            }
            if is_file_result_finished(fr, args.min_measurements, args.rel_error) {
                eprintln!(" is *GOOD ENOUGH*");
                fr.close();
            } else {
                eprintln!();
            }
        }
    }

    // Resume paused processes (explicit for error handling, guard is safety net)
    process_pauser.resume()?;

    // Sort by ordinal for display
    revisions.sort_by(|a, b| a.ordinal.cmp(&b.ordinal));

    if is_multi_file {
        print_results_multifile(&revisions, &mut mi, &args, &noise_metrics);
    } else {
        print_results_single(&revisions, &mut mi, &args, &noise_metrics);
    }

    Ok(())
}

fn print_header(title: &str, noise_metrics: &NoiseMetrics, width: usize) {
    println!("\n{}", "=".repeat(width));
    println!("{}", title);
    println!("  https://github.com/zond/jxl-perfhistory");
    println!("  CPU architecture: {}", std::env::consts::ARCH);
    if let Some(warning) = noise_metrics.warning_message() {
        println!("  {}", warning);
    }
    println!("{}", "=".repeat(width));
}

/// Draw a CI bar: ├───│───┤
/// Returns the bar as a string
fn draw_ci_bar(
    pos_lower: usize,
    pos_median: usize,
    pos_upper: usize,
    bar_width: usize,
    baseline_pos: Option<usize>,
) -> String {
    let mut bar = vec![' '; bar_width];

    // Draw baseline marker first (if provided)
    if let Some(bp) = baseline_pos
        && bp < bar_width
    {
        bar[bp] = '┃';
    }

    let lo = pos_lower.min(bar_width - 1);
    let hi = pos_upper.min(bar_width - 1);
    let med = pos_median.min(bar_width - 1);

    bar[lo] = '├';
    bar[hi] = '┤';

    for c in bar.iter_mut().take(med).skip(lo + 1) {
        if *c == '┃' {
            *c = '╂'; // CI crosses baseline
        } else {
            *c = '─';
        }
    }
    bar[med] = '│';
    for c in bar.iter_mut().take(hi).skip(med + 1) {
        if *c == '┃' {
            *c = '╂';
        } else {
            *c = '─';
        }
    }

    bar.into_iter().collect()
}

fn print_results_single(
    results: &[Revision],
    mi: &mut MedianIndices,
    args: &Args,
    noise_metrics: &NoiseMetrics,
) {
    let file_name = args.jxl_file.as_deref().unwrap_or("unknown");
    print_header(
        &format!("BENCHMARK RESULTS using {}", file_name),
        noise_metrics,
        80,
    );

    // Calculate statistics (using first file result from each revision)
    let mut min = f64::MAX;
    let mut max = f64::MIN;
    let mut sum = 0f64;
    for rev in results.iter() {
        let m = rev.file_results[0].median.unwrap();
        min = min.min(m);
        max = max.max(m);
        sum += m;
    }
    let avg = sum / results.len() as f64;

    println!("\nStatistics:");
    println!("  Revisions:          {:>15}", results.len());
    println!("  Confidence:         {:>15.1}%", 100f64 * args.confidence);
    println!("  Max relative error: {:>15.1}%", 100f64 * args.rel_error);
    println!("  Min:                {:>15.2} pixels/s", min);
    println!("  Max:                {:>15.2} pixels/s", max);
    println!("  Average:            {:>15.2} pixels/s", avg);
    println!(
        "  Improvement:        {:>15.2}% (max vs min)",
        ((max - min) / min) * 100.0
    );

    // Show performance graph with absolute pixels/s credible intervals
    println!("\n{}", "=".repeat(160));
    println!("Performance Graph (absolute pixels/s, with credible intervals):");
    println!("{}", "-".repeat(160));

    const BAR_WIDTH: usize = 80;

    // Helper to compute CI bounds from measurements
    let compute_ci = |measurements: &[f64], mi: &mut MedianIndices| -> Option<(f64, f64, f64)> {
        let n = measurements.len();
        if n < 3 {
            return None;
        }
        let mut sorted = measurements.to_vec();
        sorted.sort_by(f64_cmp);
        let median = if n.is_multiple_of(2) {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };
        let (lo, hi, _) = mi.get(n);
        Some((sorted[lo], median, sorted[hi]))
    };

    // Determine scale from absolute performance values
    let mut scale_min = f64::MAX;
    let mut scale_max = f64::MIN;
    for result in results.iter() {
        let fr = &result.file_results[0];
        if let Some((ci_lo, _, ci_hi)) = compute_ci(&fr.measurement_file.measurements, mi) {
            scale_min = scale_min.min(ci_lo);
            scale_max = scale_max.max(ci_hi);
        }
    }
    // Add some padding
    let range = scale_max - scale_min;
    scale_min -= range * 0.05;
    scale_max += range * 0.05;
    let scale_range = scale_max - scale_min;

    // Precompute ratios for display
    let mut ratios: Vec<Option<f64>> = vec![None; results.len()];
    for i in 0..(results.len() - 1) {
        let current_median = results[i].file_results[0].median.unwrap();
        let prev_median = results[i + 1].file_results[0].median.unwrap();
        ratios[i] = Some(current_median / prev_median);
    }

    for (i, result) in results.iter().enumerate() {
        let fr = &result.file_results[0];
        let median = fr.median.unwrap();

        let marker = if median == max {
            format!("▲ MAX ({:.2} / min)", max / min)
        } else if median == min {
            format!("▼ MIN ({:.2} / max)", min / max)
        } else {
            String::new()
        };

        let bar_str = if let Some((ci_lo, _, ci_hi)) =
            compute_ci(&fr.measurement_file.measurements, mi)
        {
            let pos_lo = ((ci_lo - scale_min) / scale_range * (BAR_WIDTH - 1) as f64) as usize;
            let pos_median = ((median - scale_min) / scale_range * (BAR_WIDTH - 1) as f64) as usize;
            let pos_hi = ((ci_hi - scale_min) / scale_range * (BAR_WIDTH - 1) as f64) as usize;
            draw_ci_bar(pos_lo, pos_median, pos_hi, BAR_WIDTH, None)
        } else {
            " ".repeat(BAR_WIDTH)
        };

        let ratio_str = if let Some(ratio) = ratios[i] {
            format!("{:.2} / prev", ratio)
        } else {
            String::new()
        };

        println!(
            "[{:2}] {:.8} {:<40} | {:80} | {:>12.0} px/s | {:12} {}",
            i + 1,
            result.oid,
            result.clipped_summary(40),
            bar_str,
            median,
            ratio_str,
            marker
        );
    }

    println!("{}", "-".repeat(160));
    println!("Scale: {:.0} to {:.0} pixels/s", scale_min, scale_max);
}

/// Compute ratio statistics between two sets of measurements using the same
/// order statistics approach as for individual medians.
fn compute_ratio_stats(
    current_measurements: &[f64],
    prev_median: f64,
    mi: &mut MedianIndices,
) -> Option<(f64, f64, f64)> {
    // Compute ratios: each measurement divided by previous median
    let ratios: Vec<f64> = current_measurements
        .iter()
        .map(|&m| m / prev_median)
        .collect();

    let n = ratios.len();
    if n < 3 {
        return None;
    }

    let mut sorted = ratios.clone();
    sorted.sort_by(f64_cmp);

    let median = if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };

    let (lo, hi, _) = mi.get(n);
    let ci_lower = sorted[lo];
    let ci_upper = sorted[hi];

    Some((median, ci_lower, ci_upper))
}

fn print_results_multifile(
    results: &[Revision],
    mi: &mut MedianIndices,
    args: &Args,
    noise_metrics: &NoiseMetrics,
) {
    print_header(
        &format!(
            "MULTI-FILE BENCHMARK RESULTS ({} files, {} revisions)",
            results[0].file_results.len(),
            results.len()
        ),
        noise_metrics,
        100,
    );

    println!("\nStatistics:");
    println!("  Revisions:          {:>15}", results.len());
    println!("  Confidence:         {:>15.1}%", 100f64 * args.confidence);
    println!("  Max relative error: {:>15.1}%", 100f64 * args.rel_error);

    // Skip the first (oldest) revision - nothing to compare to
    if results.len() < 2 {
        println!("Need at least 2 revisions to show comparisons.");
        return;
    }

    const BAR_WIDTH: usize = 80;

    // For each revision (except oldest), show comparison to previous
    for i in 0..(results.len() - 1) {
        let current_rev = &results[i];
        let prev_rev = &results[i + 1]; // +1 because sorted by ordinal (newest first)

        println!(
            "\n[{:2}] {:.8} {}",
            i + 1,
            current_rev.oid,
            current_rev.clipped_summary(60)
        );
        println!(
            "     vs {:.8} {}",
            prev_rev.oid,
            prev_rev.clipped_summary(60)
        );
        println!("{}", "-".repeat(160));

        // Collect ratio stats for all files to determine scale
        // Tuple: (file_name, ratio, ci_lo, ci_hi, current_median)
        let mut all_ratios: Vec<(String, f64, f64, f64, f64)> = vec![];

        for (file_idx, current_fr) in current_rev.file_results.iter().enumerate() {
            let prev_fr = &prev_rev.file_results[file_idx];

            if let (Some(prev_median), Some(current_median)) = (prev_fr.median, current_fr.median)
                && let Some((ratio, ci_lo, ci_hi)) =
                    compute_ratio_stats(&current_fr.measurement_file.measurements, prev_median, mi)
            {
                let file_name = Path::new(&current_fr.file_path)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                all_ratios.push((file_name, ratio, ci_lo, ci_hi, current_median));
            }
        }

        if all_ratios.is_empty() {
            println!("     No valid comparisons available");
            continue;
        }

        // Determine scale: center at 1.0, expand to fit all CIs
        let mut scale_min = 1.0f64;
        let mut scale_max = 1.0f64;
        for (_, _, ci_lo, ci_hi, _) in &all_ratios {
            scale_min = scale_min.min(*ci_lo);
            scale_max = scale_max.max(*ci_hi);
        }
        // Ensure symmetric around 1.0 and reasonable range
        let max_deviation = (1.0 - scale_min).max(scale_max - 1.0).max(0.1);
        scale_min = 1.0 - max_deviation * 1.1;
        scale_max = 1.0 + max_deviation * 1.1;
        let scale_range = scale_max - scale_min;

        // Position of 1.0 (baseline) in the bar
        let baseline_pos = ((1.0 - scale_min) / scale_range * (BAR_WIDTH - 1) as f64) as usize;

        // Find max file name length for alignment
        let max_name_len = all_ratios
            .iter()
            .map(|(name, _, _, _, _)| name.len())
            .max()
            .unwrap_or(10)
            .min(30);

        for (file_name, ratio, ci_lo, ci_hi, median) in &all_ratios {
            let pos_lo = ((ci_lo - scale_min) / scale_range * (BAR_WIDTH - 1) as f64) as usize;
            let pos_median = ((ratio - scale_min) / scale_range * (BAR_WIDTH - 1) as f64) as usize;
            let pos_hi = ((ci_hi - scale_min) / scale_range * (BAR_WIDTH - 1) as f64) as usize;

            let bar_str = draw_ci_bar(pos_lo, pos_median, pos_hi, BAR_WIDTH, Some(baseline_pos));

            // Truncate or pad file name
            let display_name = if file_name.len() > max_name_len {
                format!("{}...", &file_name[..(max_name_len - 3)])
            } else {
                format!("{:width$}", file_name, width = max_name_len)
            };

            // Indicator for significant change
            let indicator = if *ci_lo > 1.0 {
                format!("▲ faster ({:.2} / prev)", ratio)
            } else if *ci_hi < 1.0 {
                format!("▼ slower ({:.2} / prev)", ratio)
            } else {
                String::new()
            };

            println!(
                "     {} | {} | {:>12.0} px/s | {:.3} / prev {}",
                display_name, bar_str, median, ratio, indicator
            );
        }

        println!(
            "     {:width$}   Scale: {:.2} to {:.2} (1.0 = same speed, '┃' marks 1.0)",
            "",
            scale_min,
            scale_max,
            width = max_name_len
        );
    }

    // Show the oldest revision as baseline reference
    let oldest_rev = &results[results.len() - 1];
    println!(
        "\n[{:2}] {:.8} {} (oldest, baseline for comparisons)",
        results.len(),
        oldest_rev.oid,
        oldest_rev.clipped_summary(60)
    );

    println!("\n{}", "=".repeat(160));
}
