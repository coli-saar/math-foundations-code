# Code for "Foundations of Mathematics"

This is code for the course [Foundations of Mathematics](https://coli-saar.github.io/math20/) at the [Department of Language Science and Technology](https://www.lst.uni-saarland.de/en/) at Saarland University.

It requires Python 3 to run.

## What's in here?

* Generate random systems of linear equations (uniquely solvable, inconsistent, underconstrained). Start with `python web.py` and navigate to `/linear-equations.html`.


## Installation

Push to Heroku with `git push heroku master`. (Not `main` - this Github repository still calls the main branch "master".)

If you get the error "heroku does not appear to be a git repository", register Heroku as a Git remote with `heroku git:remote -a foundations-math`.

If the log says "no web processes running", spin up a dyno with `heroku ps:scale web=1`.



