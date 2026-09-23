---
name: weekly-deadline-sweep
description: Work out what is actually due and what to do first, from the Canvas deadlines in the vault and what was already said to be done. Use when asked to sweep, plan, or triage the week's deadlines, or asked what to work on next.
---

<!-- evie:managed -->
<!-- evie:triggers weekly deadline sweep | deadline sweep | sweep my deadlines | plan my week | what should i work on -->

# Weekly deadline sweep

The point is a decision, not a list. A list is already in the vault; reading
it back is not worth a Claude Code session.

## Read

1. `classes/upcoming.md` — everything Canvas knows, split into `## Late` and
   `## Coming up`. A `- [x]` box means submitted. `**missing**` means Canvas
   has marked it missing, which is worse than merely late.
2. The last week of `daily/*.md` — what was said out loud. Something named
   there as finished but still unticked in Canvas is worth flagging: it is
   either unsubmitted or ungraded, and those need different actions.
3. `classes/<course>/` for anything relevant — a syllabus with a weighting
   changes which item matters most.

Dates in those files are absolute on purpose. Work out "today" and "this week"
yourself; do not trust a relative phrase you find written down.

## Judge

Rank by what actually costs the most to get wrong, not by date alone:

- **missing** and heavily weighted beats merely late
- late-but-small beats on-time-and-large only if it is nearly done
- anything with a hard cutoff (a test, a timed quiz) outranks anything that
  accepts a late submission

Say when two things genuinely conflict rather than pretending a clean order
exists.

## Write

Update `classes/this-week.md`, keeping anything outside the markers:

```
<!-- evie:sweep -->
...your plan...
<!-- /evie:sweep -->
```

Lead with the single thing to do next and why. Then the rest, shortest path
first. Note anything that looks wrong in Canvas — a missing item that was
handed in, a date that cannot be right — so it can be checked.

## Say

Out loud, give the one next thing and how long it should take. Offer the rest.
The file is there for the detail; speech is not a screen.
