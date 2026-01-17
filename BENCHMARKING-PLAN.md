# Plan: Add Reproduction Time Tracking to Method of Moderation

## Overview
Add benchmarking infrastructure following the HAFiscal-Latest pattern to track reproduction times across different hardware, operating systems, and configurations.

## Current State

### Method of Moderation
- Simple reproduce.sh script with 5 steps
- No timing or system information capture
- No benchmark tracking
- Simple reproduce_min.sh for quick validation

### HAFiscal-Latest Pattern (Reference)
- reproduce/benchmarks/ directory with benchmark.sh, capture_system_info.py, schema.json
- Benchmark files stored with timestamps and metadata
- latest.json symlink to most recent benchmark

## Implementation Plan - 7 Phases

### Phase 1: Directory Structure
Create reproduce/benchmarks/ with subdirectories for results

### Phase 2: System Info Capture Script
Create capture_system_info.py adapted for method-of-moderation packages (mystmd, etc.)

### Phase 3: Benchmark Wrapper Script
Create benchmark.sh that wraps reproduce.sh/reproduce_min.sh

### Phase 4: Enhance reproduce.sh
Add optional benchmark hooks (backward compatible)

### Phase 5: JSON Schema
Create schema.json for benchmark format validation

### Phase 6: Documentation
Create README.md and BENCHMARKING_GUIDE.md

### Phase 7: Git Configuration
Update .gitignore for benchmark results

## Key Adaptations from HAFiscal

- Focus on MyST/mystmd packages (not LaTeX)
- Two reproduction modes: full (reproduce.sh) and min (reproduce_min.sh)
- Simpler structure (no complex multi-stage reproduction)

See full plan details by examining HAFiscal-Latest/reproduce/benchmarks/ for reference.
