# books.wazeem.com

Companion pages for the books by Waseem Khan, one folder per book,
served by GitHub Pages. `build.py` assembles it from each book's own
repository; do not edit the generated folders by hand.

```
python3 build.py
git add -A && git commit -m "Rebuild" && git push
```

Folders: `llm/` (Large Language Models from the Ground Up),
`claude-code/` (Claude Code from the Ground Up). To add a book, add a
`Book(...)` entry in `build.py`.
