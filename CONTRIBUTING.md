# Contributing to Flide

Thank you for helping. Please read this page before you open a pull request.

## How changes land

Accepted pull requests are merged by the maintainers and shipped in the next release. Maintainers review pull requests
in batches before each release, so a reply can take a few days.

We are a small team, so please keep one pull request open at a time. Maintainers may close those that do not fit our
plans.

## Open an issue first

Open an issue and wait for a maintainer to agree before you write code for any of these:

- a new model family
- a change to the engine, which is everything under `flide/` (model plugins live in `plugins/`)
- a new dependency, or a wider version range of an existing one

A bug fix with a test can go straight to a pull request.

## New model families

We add new model families together with the teams who make them or the people who really use them, since they know
the model best. When you ask for one, please say which you are.

A family you need for your own work can also live in its own package. Flide loads any plugin registered under the
`flide.formats` entry point, so it works with a plain `pip install flide`, and it does not have to wait for a release.

## What a pull request needs

- One change per pull request.
- You can use AI help. But please leave its Co-Authored-By lines out of your commit messages.
- A bug fix comes with a test, or the smallest command that shows the bug, and the GPU you ran it on.
- An engine change lists every engine file it touches and says why the change cannot live in a plugin. We check on our
  GPUs that every supported model keeps bit-identical answers and its speed within 2%. A change that moves another
  model's answers or speed does not land, unless it fixes them, in which case the new answers have to be closer to a
  float32 reference.
- A new model family runs on Flide's engine. Wrappers around a model's own runner and anything that needs
  `trust_remote_code` are not accepted. It has to pass our model checks against the engine its release ships. The
  prompts Flide builds match the release's token for token. Its top answers match the release's, except where the
  release itself is near a tie. Its median latency is no higher on the requests you test. Include the token comparison
  and your speed numbers on a named GPU; we run the full checks on ours.
- A dependency change comes alone, with the runs that show answers and speed unchanged.
- README, docs and benchmark numbers are written by the maintainers. We don't accept changes to them; open an issue
  instead.

## Style

Python 3.10 or newer. Keep comments short, in ASCII, and about why the code does what it does. Match the style of the
file you are editing.

## License

Flide is licensed under Apache 2.0. Your contribution is licensed under the same terms, as section 5 of the license
says.
