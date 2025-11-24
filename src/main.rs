// Copyright (c) 2025 the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use clap::Parser;
use color_eyre::eyre::{eyre, Result};
use git2::{Oid, Repository};
use memmap2::Mmap;
use rand::Rng;
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

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

struct Revision {
    oid: Oid,
    summary: String,
    binary_path: Option<String>,
    measurement_file: MeasurementFile,
    mean: Option<f64>,
    rel_error: Option<f64>,
    ordinal: usize,
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

impl Revision {
    /// Compute credible interval for a constant value measured with noise
    ///
    /// With uninformative priors, the Bayesian credible interval equals the
    /// frequentist t-interval. Each measurement = true_value + noise, with
    /// unknown true value and noise variance.
    pub fn compute(&mut self, confidence: f64) -> Result<()> {
        if !(0f64..=1f64).contains(&confidence) {
            return Err(eyre!("Can't compute with confidence {}", confidence));
        }
        let measurements = &self.measurement_file.measurements;
        let n = measurements.len() as f64;
        if n <= 2f64 {
            return Err(eyre!("Can't compute with {} measurements", n));
        }
        let mean = measurements.iter().sum::<f64>() / n;

        let variance = measurements.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_error = (variance / n).sqrt();

        let t_dist = StudentsT::new(0.0, 1.0, n - 1.0).unwrap();
        let t_critical = t_dist.inverse_cdf(1.0 - (1.0 - confidence) / 2.0);
        let margin = t_critical * std_error;

        self.mean = Some(mean);
        self.rel_error = Some(margin / mean.abs());
        Ok(())
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
                mean: None,
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

    let result: Result<()> = (|| {
        let mut unfinished_revisions = collect_revisions(
            &repo,
            args.revisions,
            args.data_directory.as_deref().map(Path::new),
        )?;
        for rev in &mut unfinished_revisions {
            checkout_revision(&repo, rev.oid)?;
            print_flush!("Building {}: {}...", rev.oid, rev.summary);
            rev.build(binary_dir)?;
            println!("done!");
        }
        restore_repo(&repo, original_ref_name)?;
        let mut finished_revisions = vec![];
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
                let n = rev.measurement_file.measurements.len();
                if n > 2 {
                    let old_relative_error = rev.rel_error;
                    rev.compute(args.confidence)?;
                    if n >= args.min_measurements && rev.rel_error.unwrap() <= args.rel_error {
                        println!(
                            "done! {} samples, mean/error ({:.2}/{:.4} (from {:?})) is *GOOD ENOUGH*",
                            n,
                            rev.mean.unwrap(),
                            rev.rel_error.unwrap(),
                            old_relative_error,
                        );
                        true
                    } else {
                        println!(
                            "done! {} samples, mean/error: {:.2}/{:.4} (from {:?})",
                            n,
                            rev.mean.unwrap(),
                            rev.rel_error.unwrap(),
                            old_relative_error,
                        );
                        false
                    }
                } else {
                    println!("done!");
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
        let m = rev.mean.unwrap();
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

    // Show performance graph
    println!("\n{}", "=".repeat(80));
    println!("Performance Graph (normalized to max):");
    println!("{}", "-".repeat(80));

    for (i, result) in results.iter().enumerate() {
        let speed = result.mean.unwrap();
        let normalized = ((speed - min) / (max - min) * 40.0) as usize;
        let bar = "\u{2588}".repeat(normalized.max(1));

        let marker = if speed == max {
            "\u{25b2} MAX"
        } else if speed == min {
            "\u{25bc} MIN"
        } else {
            ""
        };

        println!(
            "[{:2}] {:.8} {:<50} {:40} {:.2} {}",
            i + 1,
            result.oid,
            result.clipped_summary(50),
            bar,
            speed,
            marker
        );
    }

    println!("{}", "-".repeat(80));
    println!("Scale: Min={:.0} pixels/s, Max={:.0} pixels/s", min, max);

    println!("\nDetailed Results:");
    println!("{}", "-".repeat(80));
    println!(
        "{:<4} {:<10} {:<50} {:>20}",
        "#", "Commit", "Message", "Performance"
    );
    println!("{}", "-".repeat(80));

    for (i, result) in results.iter().enumerate() {
        println!(
            "[{:2}] {:.8} {:<50} {:>20.2} pixels/s",
            i + 1,
            result.oid,
            result.clipped_summary(50),
            result.mean.unwrap()
        );
    }
}
