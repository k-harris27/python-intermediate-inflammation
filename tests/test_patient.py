"""Tests for the Patient model."""

import pytest

try:
    from inflammation.models import Patient
except ImportError:
    Patient = None

@pytest.mark.skipif(Patient is None, reason="Patient class not yet implemented!")
def test_create_patient():

    name = 'Alice'
    p = Patient(name=name)

    assert p.name == name
