"""Tests for external.due module (duecredit stub)."""

from torchio.external.due import InactiveDueCreditCollector


class TestInactiveDueCreditCollector:
    """Tests for the duecredit stub collector."""

    def test_dcite_returns_identity_decorator(self):
        """dcite() returns a decorator that does not modify the function."""
        collector = InactiveDueCreditCollector()

        @collector.dcite(description='test')
        def my_function():
            return 42

        assert my_function() == 42

    def test_repr(self):
        """__repr__ returns class name with parens."""
        collector = InactiveDueCreditCollector()
        assert repr(collector) == 'InactiveDueCreditCollector()'

    def test_active_is_false(self):
        """Stub collector is not active."""
        collector = InactiveDueCreditCollector()
        assert collector.active is False

    def test_donothing_methods(self):
        """activate, add, cite, dump, load are all no-ops."""
        collector = InactiveDueCreditCollector()
        collector.activate()
        collector.add()
        collector.cite()
        collector.dump()
        collector.load()
