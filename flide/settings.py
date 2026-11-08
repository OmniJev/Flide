"""Process-wide settings, fixed before an engine is built.

`batch_invariant`: matrix products, attention tiles and the final norm that keep each request's numbers the same
whatever else shares its round, so a request gets the same bits at any load (`Engine(batch_invariant=True)`,
`--batch-invariant`; slower).

`max_media_bytes`: the largest image or video a request may carry, once decoded (`--max-media-mb`, 64 by default); read
from FLIDE_MAX_MEDIA_MB, so the planning processes of `serve --frontends` hold the same."""
import os

batch_invariant = False
max_media_bytes = int(float(os.environ.get("FLIDE_MAX_MEDIA_MB", "64")) * 2 ** 20)
