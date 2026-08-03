# Research Notes — Machine Unlearning

A Quarto website used as a public research journal for my PhD on machine
unlearning. Every post follows the same structure — *Context / Idea /
Formalisation / Open questions / References* — so that notes can be lifted
into the thesis manuscript later.

**Author:** Virgile Dine
**Live site:** https://owl1996.github.io/blog

## Local development

```bash
quarto preview          # live-reloading server at localhost:xxxx
quarto render           # build the static site into _site/
```

## Adding a post

```bash
cp -r posts/_template posts/2026-08-01-my-idea
$EDITOR posts/2026-08-01-my-idea/index.qmd   # set title/date/categories, drop `draft: true`
git add . && git commit -m "post: my idea" && git push
```

The push triggers `.github/workflows/publish.yml`, which renders the site and
deploys it to GitHub Pages. No manual publish step.

## Layout

| Path | Purpose |
|---|---|
| `_quarto.yml` | Site config: nav, theme, MathJax macros, bibliography |
| `index.qmd` | Home page + listing of recent posts |
| `posts.qmd` | Full post index (table view) |
| `about.qmd` | About page |
| `posts/` | One folder per post |
| `posts/_template/` | Skeleton for new posts (ignored by Quarto — leading `_`) |
| `references.bib` | Shared BibTeX database |
| `styles.css` | Custom CSS |
| `_site/`, `.quarto/` | Build output — git-ignored |

## Maths

MathJax is enabled site-wide. Shorthand macros defined in `_quarto.yml`:
`\D` `\Df` `\Dr` `\A` `\U` `\E` `\R` `\argmin` `\argmax`.

## Citations

Add an entry to `references.bib`, cite it as `[@key]`, and end the post with:

```markdown
## References

::: {#refs}
:::
```
