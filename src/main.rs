// Copyright (c) 2025 the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use clap::Parser;
use color_eyre::eyre::{Result, eyre};
use git2::{Oid, Repository};
use memmap2::Mmap;
use rand::Rng;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::process::Command;
use std::result;
use tempfile::TempDir;
use thiserror::Error;

#[derive(Parser, Debug)]
#[command(name = "jxl-perfhistory")]
#[command(about = "Benchmark jxl_cli across git revisions", long_about = None)]
struct Args {
    /// Number of revisions to go back from HEAD
    #[arg(short = 'r', long = "revisions", default_value = "10")]
    revisions: usize,

    /// Path to the JXL file to decode
    #[arg(short = 'f', long = "file")]
    jxl_file: String,

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
}

macro_rules! print_flush {
    ($($arg:tt)*) => {{
        print!($($arg)*);
        io::stdout().flush().unwrap();
    }};
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

            // SAFETY: The file contains only f64 values written by append().
            // f64 has no alignment requirements beyond 8 bytes.
            // The mmap is read-only and we copy the data out before it's dropped.
            let mmap = unsafe { Mmap::map(&file)? };
            let bytes: &[f64] = bytemuck::cast_slice(&mmap);
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
            file.write_all(&value.to_le_bytes())?;
            file.flush()?;
        }
        self.measurements.push(value);
        Ok(())
    }

    /// Close the file handle explicitly.
    fn close(&mut self) {
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
        let mut coeffs = vec![0u128; n + 1];
        coeffs[0] = 1;
        for k in 1..=n {
            coeffs[k] = coeffs[k - 1] * (n - k + 1) as u128 / k as u128;
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

struct Revision {
    oid: Oid,
    summary: String,
    binary_path: Option<String>,
    measurement_file: MeasurementFile,
    median: Option<f64>,
    rel_error: Option<f64>,
    ordinal: usize,
}

impl Revision {
    /// Compute credible interval for median using order statistics.
    ///
    /// Uses the Bayesian interpretation: with an uninformative prior,
    /// the probability that the true median lies between X_(r) and X_(s)
    /// is given by the Binomial(n, 0.5) distribution. This is both a valid
    /// Bayesian credible interval and a frequentist confidence interval.
    pub fn compute_median(
        &mut self,
        mi: &mut MedianIndices,
    ) -> result::Result<(), InsufficientSamples> {
        let measurements = &self.measurement_file.measurements;
        let n = measurements.len();
        if n < 3 {
            return Err(InsufficientSamples(n));
        }

        // Sort measurements to get order statistics
        let mut sorted = measurements.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute median
        let median = if n.is_multiple_of(2) {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        // Get CI indices using exact binomial probabilities
        let (lo, hi, _actual_confidence) = mi.get(n);

        let ci_lower = sorted[lo];
        let ci_upper = sorted[hi];
        let half_width = (ci_upper - ci_lower) / 2.0;

        self.median = Some(median);
        self.rel_error = Some(half_width / median.abs());
        Ok(())
    }

    pub fn n_measurements(&self) -> usize {
        self.measurement_file.measurements.len()
    }

    pub fn benchmark(&mut self, jxl_file: &str) -> Result<()> {
        let output = Command::new(self.binary_path.as_ref().unwrap())
            .arg(jxl_file)
            .args(["--speedtest", "--num-reps", "1"])
            .output()?;
        if !output.status.success() {
            return Err(eyre!(
                "Benchmark failed for {:.8}!\n{}",
                self.oid,
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let pixels_per_sec: f64 = stdout
            .lines()
            .find(|line| line.contains("pixels/s") && line.contains("Decoded"))
            .and_then(|line| line.split_whitespace().rev().nth(1))
            .ok_or_else(|| eyre!("Can't find decoding speed in `{}`", stdout))?
            .parse()
            .map_err(|e| eyre!("Can't parse decoding speed: {}", e))?;
        self.measurement_file.append(pixels_per_sec)?;
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
) -> Result<Vec<Revision>> {
    let mut revwalk = repo.revwalk()?;
    revwalk.push_head()?;

    let mut ordinal = 0;
    revwalk
        .take(count)
        .map(|oid| {
            let oid = oid?;
            ordinal += 1;

            let measurement_file = match data_dir {
                Some(dir) => MeasurementFile::open(Path::new(&dir.join(oid.to_string())))?,
                None => MeasurementFile {
                    file: None,
                    measurements: Vec::new(),
                },
            };

            Ok(Revision {
                oid,
                summary: repo
                    .find_commit(oid)?
                    .summary()
                    .unwrap_or("(no message)")
                    .to_string(),
                binary_path: None,
                measurement_file,
                median: None,
                rel_error: None,
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

    // Save original HEAD state (branch or detached)
    let head = repo.head()?;
    if head.is_branch() {
        head.name()
            .ok_or(eyre!(
                "Working directory doesn't have a name, won't be able to restore it properly."
            ))
            .map(|s| s.into())
    } else {
        Err(eyre!(
            "Working directory isn't a checked out branch, won't be able to restore it properly."
        ))
    }
}

fn restore_repo(repo: &Repository, original_ref_name: String) -> Result<()> {
    repo.set_head(&original_ref_name)?;
    let mut opts = git2::build::CheckoutBuilder::new();
    opts.force();
    repo.checkout_head(Some(&mut opts))?;
    Ok(())
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Args::parse();

    let mut rng = rand::rng();
    let tmp_dir = TempDir::new()?;
    let binary_dir = match &args.binary_directory {
        Some(s) => Path::new(s),
        None => tmp_dir.path(),
    };

    let repo = Repository::open(".")?;
    let original_ref_name = verify_repo(&repo)?;
    let mut mi = MedianIndices::new(args.confidence)?;

    let result: Result<()> = (|| {
        let mut unfinished_revisions = vec![];
        let mut finished_revisions = vec![];
        for mut rev in collect_revisions(
            &repo,
            args.revisions,
            args.data_directory.as_deref().map(Path::new),
        )? {
            if rev.n_measurements() < args.min_measurements {
                unfinished_revisions.push(rev);
            } else if let Err(InsufficientSamples(_)) = rev.compute_median(&mut mi) {
                unfinished_revisions.push(rev);
            } else if let Some(current_error) = rev.rel_error
                && current_error <= args.rel_error
            {
                println!(
                    "{:.8}: {:<50} is already done, {} samples, median/error: {:.2}/{:.4}",
                    rev.oid,
                    rev.clipped_summary(50),
                    rev.n_measurements(),
                    rev.median.unwrap(),
                    rev.rel_error.unwrap()
                );
                finished_revisions.push(rev);
            } else {
                unfinished_revisions.push(rev);
            }
        }

        for rev in &mut unfinished_revisions {
            checkout_revision(&repo, rev.oid)?;
            print_flush!("Building {}: {}...", rev.oid, rev.summary);
            rev.build(binary_dir)?;
            println!("done!");
        }
        restore_repo(&repo, original_ref_name)?;

        while !unfinished_revisions.is_empty() {
            let idx = rng.random_range(0..unfinished_revisions.len());
            let done_with_revision = {
                let rev = &mut unfinished_revisions[idx];
                print_flush!(
                    "Benchmarking {:.8}: {:<50}...",
                    rev.oid,
                    rev.clipped_summary(50)
                );
                rev.benchmark(&args.jxl_file)?;
                let old_relative_error = rev.rel_error;

                if let Err(InsufficientSamples(_)) = rev.compute_median(&mut mi) {
                    println!("done!");
                    false
                } else if rev.n_measurements() >= args.min_measurements
                    && rev.rel_error.unwrap() <= args.rel_error
                {
                    println!(
                        "done! {} samples, median/error ({:.2}/{:.4} (from {:?})) is *GOOD ENOUGH*",
                        rev.n_measurements(),
                        rev.median.unwrap(),
                        rev.rel_error.unwrap(),
                        old_relative_error,
                    );
                    true
                } else {
                    println!(
                        "done! {} samples, median/error: {:.2}/{:.4} (from {:?})",
                        rev.n_measurements(),
                        rev.median.unwrap(),
                        rev.rel_error.unwrap(),
                        old_relative_error,
                    );
                    false
                }
            };
            if done_with_revision {
                let mut rev = unfinished_revisions.swap_remove(idx);
                rev.measurement_file.close();
                finished_revisions.push(rev);
            }
        }
        finished_revisions.sort_by(|a, b| a.ordinal.partial_cmp(&b.ordinal).unwrap());
        print_results(&finished_revisions, &args);
        Ok(())
    })();

    result
}

fn print_results(results: &[Revision], args: &Args) {
    println!("\n{}", "=".repeat(80));
    println!("BENCHMARK RESULTS using {}", args.jxl_file);
    println!("{}", "=".repeat(80));

    // Calculate statistics
    let mut min = f64::MAX;
    let mut max = f64::MIN;
    let mut sum = 0f64;
    for rev in results.iter() {
        let m = rev.median.unwrap();
        min = min.min(m);
        max = max.max(m);
        sum += m;
    }
    let avg = sum / results.len() as f64;

    println!("\nStatistics:");
    println!("  Samples:            {:>15}", results.len());
    println!("  Confidence:         {:>15.1}%", 100f64 * args.confidence);
    println!("  Max relative error: {:>15.1}%", 100f64 * args.rel_error);
    println!("  Min:                {:>15.2} pixels/s", min);
    println!("  Max:                {:>15.2} pixels/s", max);
    println!("  Average:            {:>15.2} pixels/s", avg);
    println!(
        "  Improvement:        {:>15.2}% (max vs min)",
        ((max - min) / min) * 100.0
    );

    // Show performance graph with credible intervals
    println!("\n{}", "=".repeat(80));
    println!("Performance Graph (with credible intervals):");
    println!("{}", "-".repeat(150));

    // Find the full range including all CI bounds
    let mut graph_min = min;
    let mut graph_max = max;
    for result in results.iter() {
        let median = result.median.unwrap();
        let half_width = result.rel_error.unwrap() * median;
        graph_min = graph_min.min(median - half_width);
        graph_max = graph_max.max(median + half_width);
    }
    let graph_range = graph_max - graph_min;

    const BAR_WIDTH: usize = 60;

    for (i, result) in results.iter().enumerate() {
        let median = result.median.unwrap();
        let half_width = result.rel_error.unwrap() * median;
        let ci_lower = median - half_width;
        let ci_upper = median + half_width;

        // Normalize positions to bar width
        let pos_lower = ((ci_lower - graph_min) / graph_range * (BAR_WIDTH - 1) as f64) as usize;
        let pos_median = ((median - graph_min) / graph_range * (BAR_WIDTH - 1) as f64) as usize;
        let pos_upper = ((ci_upper - graph_min) / graph_range * (BAR_WIDTH - 1) as f64) as usize;

        // Build the bar string
        let mut bar = vec![' '; BAR_WIDTH];
        bar[pos_lower] = '├';
        for char in bar[(pos_lower + 1)..pos_median].iter_mut() {
            *char = '─';
        }
        bar[pos_median] = '│';
        for char in bar[(pos_median + 1)..pos_upper].iter_mut() {
            *char = '─';
        }
        bar[pos_upper] = '┤';
        let bar_str: String = bar.into_iter().collect();

        let marker = if median == max {
            "▲ MAX"
        } else if median == min {
            "▼ MIN"
        } else {
            ""
        };

        println!(
            "[{:2}] {:.8} {:<50} | {:60} | {:.2} {}",
            i + 1,
            result.oid,
            result.clipped_summary(50),
            bar_str,
            median,
            marker
        );
    }

    println!("{}", "-".repeat(150));
    println!("Scale: {:.0} to {:.0} pixels/s", graph_min, graph_max);
}
