# autosokoban
Automatic Sokoban level solving and generation

**(Still cleaning up this readme)**

:computer: Getting set up
- Requires python 3.12+
- requirements.txt for necessary packages

:package::package::package: [Sokoban](https://en.wikipedia.org/wiki/Sokoban)! It's a type of puzzle!

![image](https://github.com/user-attachments/assets/0785b872-0ebb-446e-a54c-7109d51062e2)

As far as puzzles go it's not that complicated, so I had the idea to try writing something to generate levels...and [it's already been done, really well](https://linusakesson.net/games/autosokoban/).

But it's still a cool project, so I made this repo with a few goals in mind
- Make a decent Sokoban solver
  - Do it in an organized way!
  - Like, make something *general* that just happens to work for Sokoban
- Get something generating puzzles
  - Ideally something on the same level as [Linus Akesson's](https://linusakesson.net/games/autosokoban/) generator
  - Eventually get mine onto a website
- Learn how to do python cleanly
  - Before this I hadn't used virtual environments or requirements.txt files!
  - Taking this as a chance to make sure I know misc "common" things: gitignore, jsons, unit testing, etc.

Notable literature
- "Automatic Making of Sokoban Problems" by Murase, Matsubara, and Hiraga
- "Procedural Generation of Sokoban Levels" by Taylor and Parberry
- "The FESS Algorithm: A Feature Based Approach to Single-Agent Search" by Shoham and Schaeffer

Existing work out there
- [This website](https://linusakesson.net/games/autosokoban/) (that's the third time I linked it, can you tell I think it's awesome?)
- [A nice web app with a solver](https://dangarfield.github.io/sokoban-solver/)
- [This defunct site](https://web.archive.org/web/20191002082058/http://www.erimsever.com/sokoban7.htm) which links to a bunch of solvers

Sokoban "community"
- [A wiki](http://sokobano.de/wiki/)
- [A standard test suite](https://sokoban-solver-statistics.sourceforge.io/statistics/XSokoban/XSokoban.html)
