---
name: lecture-note-cleanup
description: Turn raw notes for a class into something usable to revise from, and say what is missing. Use when asked to clean up, tidy, organise or rewrite notes for a course.
---

<!-- evie:managed -->
<!-- evie:triggers lecture note cleanup | clean up my notes | tidy my notes | organise my notes | organize my notes | sort out my notes -->

# Lecture note cleanup

Notes taken in a lesson are a transcript. Notes revised from are a structure.
The job is the second from the first, without inventing anything.

## Read

Work inside `classes/<course>/`. If the course was not named, list the ones
with notes and ask which — guessing wastes the run.

## Rewrite

Keep the original. Write the tidied version alongside it as
`<original>-clean.md` so nothing is lost if the pass is wrong.

- One heading per idea, in the order it was taught
- Definitions as definitions: the term, then what it means, so it can be
  covered up and recalled
- Worked examples kept whole — a half-copied example is worse than none
- Contradictions left standing and marked `> check this`, not silently
  resolved. A note that disagrees with itself usually means something was
  misheard in class, and that is exactly what needs asking about.

**Never add content that is not in the source.** Filling a gap from your own
knowledge produces notes that look complete and revise in the wrong material.
A gap is a finding.

## Report

End the clean file with what is thin or missing: a topic with two lines under
it, a definition referred to but never given, an example started and dropped.
Cross-reference `classes/upcoming.md` — if something thin is on an assessment
this week, say so first.

Out loud: what was cleaned, and the one gap worth filling before the next
class.
