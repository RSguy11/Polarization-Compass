"""
Underwater Testing Module
=========================

Trains and evaluates polarization compass ML models on the
Capstone underwater dataset (Capstone_live_data/).

Components:
    UnderwaterDataLoader  – loads raw PNGs + solar_labels.parquet
    run_all_models        – full training pipeline (same models as Training_loops/)

Usage:
    python -m Underwater_testing.run_all_models
"""
