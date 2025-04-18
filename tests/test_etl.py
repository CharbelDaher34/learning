import pytest
import polars as pl
from app.etl.data_pipeline import AGNewsETLFlow
import re


def test_clean_text():
    """Test text cleaning function."""
    flow = AGNewsETLFlow()

    # Access the clean_text function from the transform_data step
    clean_text = flow.transform_data.__closure__[0].cell_contents

    # Test HTML removal
    text_with_html = "<p>This is a test</p>"
    assert clean_text(text_with_html) == "this is a test"

    # Test lowercase conversion
    text_with_caps = "THIS IS A TEST"
    assert clean_text(text_with_caps) == "this is a test"

    # Test special character removal
    text_with_special = "This is a test! With @#$ characters."
    cleaned = clean_text(text_with_special)
    assert re.match(r"^[a-z0-9\s]+$", cleaned) is not None


def test_data_transformation():
    """Test data transformation logic."""
    # Create sample data
    data = {
        "title": ["Test Title", "<p>HTML Title</p>"],
        "description": ["Test Description", "Another Description"],
        "label": [1, 2],
    }
    df = pl.DataFrame(data)

    # Create flow instance
    flow = AGNewsETLFlow()
    flow.df = df

    # Run transform step
    flow.transform_data()

    # Verify transformations
    assert "cleaned_title" in flow.df.columns
    assert "cleaned_description" in flow.df.columns
    assert "text" in flow.df.columns
    assert "processed_at" in flow.df.columns

    # Verify content
    transformed_data = flow.df.to_dicts()
    assert transformed_data[0]["cleaned_title"] == "test title"
    assert transformed_data[1]["cleaned_title"] == "html title"


def test_database_connection(mocker):
    """Test database connection and operations."""
    # Mock psycopg2 connection and cursor
    mock_cursor = mocker.MagicMock()
    mock_conn = mocker.MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    mocker.patch("psycopg2.connect", return_value=mock_conn)

    # Create flow instance with test data
    flow = AGNewsETLFlow()
    flow.df = pl.DataFrame(
        {"label": [1], "text": ["test text"], "processed_at": ["2024-01-01 00:00:00"]}
    )

    # Run load step
    flow.load_data()

    # Verify database operations
    mock_cursor.execute.assert_called()  # Verify table creation
    mock_conn.commit.assert_called_once()  # Verify transaction commit


def test_full_pipeline_integration(mocker):
    """Test full pipeline integration."""
    # Mock external dependencies
    mocker.patch(
        "polars.read_ndjson",
        return_value=pl.DataFrame(
            {"title": ["Test Title"], "description": ["Test Description"], "label": [1]}
        ),
    )

    # Mock database operations
    mock_cursor = mocker.MagicMock()
    mock_conn = mocker.MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mocker.patch("psycopg2.connect", return_value=mock_conn)

    # Run flow
    flow = AGNewsETLFlow()
    flow.start()
    flow.extract_data()
    flow.transform_data()
    flow.load_data()
    flow.end()

    # Verify pipeline completion
    assert mock_conn.commit.called
    assert isinstance(flow.df, pl.DataFrame)
