"""Tests for the fromdecay submodule."""
import pytest

# This makes it so that assert errors are more helpful for e.g., the check_norm helper function
pytest.register_assert_rewrite("fromdecay.test_fulldecay")
