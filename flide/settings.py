"""Process-wide settings, fixed before an engine is built.

`batch_invariant`: matrix products, attention tiles and the final norm that keep each request's numbers the same
whatever else shares its round, so a request gets the same bits at any load (`Engine(batch_invariant=True)`,
`--batch-invariant`; slower)."""

batch_invariant = False
