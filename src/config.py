"""
Configuration module for Graph Comparison Tool.

This module contains all configuration settings for the application,
including database connection parameters and file paths.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "neo4j"
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 100
    connection_acquisition_timeout: int = 60


@dataclass
class AppConfig:
    """Application configuration settings."""
    
    # File paths
    cypher_file_1: str = "data/macm_files/Ewelink_incorrect.macm"
    cypher_file_2: str = "data/macm_files/Ewelink_correct.macm"
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Output settings
    verbose: bool = False
    output_format: str = "text"  # text, json, csv


def get_config() -> tuple[DatabaseConfig, AppConfig]:
    """
    Get configuration from environment variables or defaults.
    
    Returns:
        Tuple of (DatabaseConfig, AppConfig)
    """
    db_config = DatabaseConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "neo4j"),
        max_connection_lifetime=int(os.getenv("NEO4J_MAX_CONNECTION_LIFETIME", "3600")),
        max_connection_pool_size=int(os.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE", "100")),
        connection_acquisition_timeout=int(os.getenv("NEO4J_CONNECTION_TIMEOUT", "60"))
    )
    
    app_config = AppConfig(
        cypher_file_1=os.getenv("CYPHER_FILE_1", "data/macm_files/Ewelink_incorrect.macm"),
        cypher_file_2=os.getenv("CYPHER_FILE_2", "data/macm_files/Ewelink_correct.macm"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE"),
        verbose=os.getenv("VERBOSE", "false").lower() == "true",
        output_format=os.getenv("OUTPUT_FORMAT", "text")
    )
    
    return db_config, app_config
