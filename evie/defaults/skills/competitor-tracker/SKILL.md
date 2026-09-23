---
name: competitor-tracker
description: Check what companies in the same space have shipped or announced, and record what changed since last time. Use when asked to track, check on, or update notes about competitors.
---

<!-- evie:managed -->
<!-- evie:triggers competitor tracker | check on competitors | track my competitors | what are competitors doing | competitive landscape -->

# Competitor tracker

Only useful as a diff. "Here is what company X does" is a search result; "X
started charging for the thing we give away" is worth knowing.

## First run

If `business/competitors.md` does not exist, this has not been set up. Ask
what the business does and who it is up against, write the file, and stop —
a tracker with nothing to track invents its own subject, and a confidently
wrong competitor list is worse than an empty one.

## Each run

1. Read `business/competitors.md` for who is tracked and what was last seen.
2. For each, check what is public now: the pricing page, the changelog or
   blog, recent announcements.
3. Compare against what is recorded. **The diff is the output.** No change is
   a real result and should be written as one, with the date.

## Write

Update each entry in place, keeping the history:

```
## <name>
- **2026-09-23** — what changed, and why it matters here
- **2026-09-09** — ...
```

Separate what you saw from what you concluded. A price on a page is a fact; a
guess about why it moved is a guess, and should read like one.

## Say

Out loud: whether anything changed, and the one thing that affects what to
build or charge next. If nothing changed, say that in one sentence — it is
the most common answer and it is useful.
