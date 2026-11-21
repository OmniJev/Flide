# Contributing to Flide

Thank you for helping. Please read this page before you open a pull request.

## How changes land

Accepted pull requests are merged by the maintainers and shipped in the next release.

## Open an issue first

Open an issue and wait for a maintainer to agree before you write code for any of these:

- a new model family
- a change to the engine, which is everything under `flide/` (model plugins live in `plugins/`)
- a new dependency, or a wider version range of an existing one

A bug fix with a test can go straight to a pull request.

## What a pull request needs

- One change per pull request.
- You can use AI help. But please leave its Co-Authored-By lines out of your commit messages.
- A bug fix comes with a test, or the smallest command that shows the bug, and the GPU you ran it on.
- An engine change lists every engine file it touches and says why the change cannot live in a plugin. We check on our
  GPUs that every supported model keeps bit-identical answers and its speed within 2%. A change that moves another
  model's answers or speed does not land, unless it fixes them, in which case the new answers have to be closer to a
  float32 reference.

## Style

Python 3.10 or newer. Keep comments short, in ASCII, and about why the code does what it does. Match the style of the
file you are editing.

## License

Flide is licensed under Apache 2.0. Your contribution is licensed under the same terms, as section 5 of the license
says.
