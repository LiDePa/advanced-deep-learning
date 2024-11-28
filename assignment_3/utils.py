class Counter:
    """A counter for counting. It can count.
    """

    def __init__(self):
        """Initializes the counter. The counter is initially set to 0.
        """
        self._counter = 0

    def __call__(self):
        """Increments the counter by 1."""
        self._counter += 1

    @property
    def state(self) -> int:
        """Current state of the counter."""
        return self._counter
