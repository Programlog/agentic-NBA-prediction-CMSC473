# H1 experiment log

**H1:** adding box-tracking features (shooter and defender measurements from BasketEvent's
per-frame player and ball boxes) improves shot-make prediction beyond tabular shot data.

**Current result (frozen design D0, `configs/basketevent_h1_final.json`, cohort v5):** on 7,726
unblocked jump shots from 184 train and validation games, 5 date-blocked folds, the box features
lower Brier by **0.0009** (95% game-bootstrap interval 0.0001 to 0.0017): Brier 0.1963 to 0.1954,
AUC 0.572 to 0.583. That is about a third of what all the tabular features add over the constant
prediction (0.1992 to 0.1963). The gain comes mainly from the shooter's release height. The test
split has not been scored.

Brier differences below are box model minus baseline; negative favours the boxes. Intervals are
95% paired bootstraps over whole games (1,000 resamples). "Significant" means the interval
excludes zero.

## Data and what the features are

The inputs are not video pixels. BasketEvent's authors ran detection models on broadcast video
and published per-frame `[x, y, w, h]` boxes for each player and the ball, plus the shot event.
Our 9 box features are computed from those boxes at the detected release, in units of the
shooter's standing box height (bh), which cancels camera zoom:

- shooter: release height, jump, ball rise speed, shooter speed;
- nearest defender: distance, defenders within 1 bh, closing speed, on the basket side, arms up.

Raw video exists for only 778 cohort shots (13 games in the dataset's own valid/test folders), all
in our training split.

---

## 1. Original comparison on one validation split (runs 1 and 2)

- **Why:** the simplest fair test: the same XGBoost with and without the 9 box features, scored on
  the 2,138 shots of the 32 validation games.
- **What happened:** no significant difference. Run 1: -0.0005 (-0.0037 to +0.0027). Run 2, after
  the defender corrections in 3: +0.0007 (-0.0024 to +0.0038).
- **Why it didn't work:** three problems found later. The validation set could only detect large
  effects; about 1% of shots had wildly wrong box features (11); and the baseline was weaker than
  it looked, so the comparison was easier to misread (5).

## 2. Dropping shots with no detected release (commit 40cd4fd)

- **Why:** shots whose clip had no detectable release, or held several shots, were kept with blank
  box features. A multi-shot clip is usually a miss and its putback (21% made vs 37%), so "all box
  features blank" told the model "miss".
- **What happened:** an early, apparent gain for the box model disappeared. Only shots with a
  release enter the cohort, for every model, and the audit reports the make rate of each dropped
  group.
- **Intuition:** missingness can carry the outcome; a feature set that is blank for a
  non-random group of shots leaks. This fix is what makes every later result trustworthy.

## 3. Camera perspective and two more defender features (commit a682b75)

- **Why:** distances were measured flat on the screen; the broadcast camera looks down, so a
  defender deeper on the court looks closer than they are. Vertical gaps were stretched by a depth
  factor, and defender-on-basket-side and defender-arms-up were added.
- **What happened:** no measurable change. Depth factors 1.0 to 4.0 differed by under 0.0006 Brier
  in grouped CV on training games.
- **Intuition:** perspective was a real error but a small one next to the glitches in 11.

## 4. Blind review of the labels (25 clips)

- **Why:** to check that the automatic boxes and release frames are right before trusting
  features built on them.
- **What happened:** shooter box correct 25/25, defender box 25/25; release within 1 frame of the
  reviewer 40%, within 3 frames 84%, median error 2 frames (0.08 s). One clip (item 20) is
  unresolved: the reviewer marked frame 179, outside its strip.
- **Intuition:** the labels are good enough to build on; release timing is usually a frame or two
  off.

## 5. Checking which baseline features are filled

- **Why:** to understand why 9 extra features changed nothing.
- **What happened:** 4 of the 9 baseline features (defender distance, shot clock, dribbles, touch
  time) are empty for every 2025-26 shot; they come from a tracking source with no public per-shot
  data this season. The baseline really had 5 features: shot distance, 2 vs 3, period, game clock,
  home or away (`location` is home/away, not court zone).
- **Intuition:** the boxes were the only defender information in either model, which made the
  null result a sign of noise rather than redundancy with real tracking stats.

## 6. Box features alone

- **Why:** to tell "the boxes carry no information" from "their information overlaps with the
  baseline".
- **What happened:** boxes alone AUC 0.605; shot distance alone 0.678; the 5-feature baseline
  0.684; baseline plus boxes 0.683.
- **Intuition:** the boxes predict, but mostly by re-describing the shot: a high release with an
  open defender is a long jump shot, a low release among defenders is a shot at the rim, and the
  model already knows that from distance.

## 7. Dataset composition

- **Why:** a check on the cohort's make rate, which looked low.
- **What happened:** BasketEvent's clips over-represent misses. In the same games, 46.8% of all
  shots went in but 37.4% of cohort shots; for jump shots 45% vs 27%. Makes and misses match to
  the play-by-play at the same 99.99% rate, so this is the dataset's clip selection, not ours.
- **Intuition:** the comparison stays fair because both models score the same shots, but results
  describe BasketEvent's shots, not all NBA shots.

## 8. Stronger tabular baseline (run 3)

- **Why:** a convincing result has to beat the best tabular baseline available; otherwise the boxes
  might only be recovering shot type. Added from the NBA shot chart we already had: shot type
  (e.g. "Pullup Jump Shot"), zone, area, court x/y, and the shooter's earlier-game make rate. Kept
  outside the LLM agents' feature list so their prompts did not change.
- **What happened:** the baseline improved (Brier 0.2079 to 0.2065). Boxes on all shots: +0.0016
  (-0.0008 to +0.0037), not significant.
- **Intuition:** this made the test fairer, not easier; the problems in 1 were still there.

## 9. Jump shots only (run 3)

- **Why:** body mechanics should matter most where distance explains least; mixing in layups and
  dunks lets rim versus perimeter dominate.
- **What happened:** on 1,300 validation jump shots, boxes -0.0003 (-0.0037 to +0.0033). Both
  default XGBoost models were worse than always predicting 27% (Brier 0.1978 and 0.1975 vs
  0.1964).
- **Why it didn't work yet:** the default XGBoost overfit a small, hard subset, the test was too
  small, and the features were still glitchy. The idea was right: this is where the final effect
  shows.

## 10. Open-shot validity check

- **Why:** open shots go in more often; if our defender distance cannot reproduce that, it is not
  measuring what it claims.
- **What happened:** within train and validation jump shots, holding shot distance fixed, the
  effect of log defender distance was 0.04 (-0.10 to +0.15): nothing. Values like defenders 320 bh
  away and jumps of 500 bh were present. Leaving out defender distances over 3 bh gave 0.35 (0.11 to
  0.54); open threes went in 28% vs 22% for the most tightly guarded quarter.
- **Intuition:** the feature measured something real, but a few impossible values flattened the
  relationship. This exposed the glitch problem.

## 11. Glitch filter for player boxes (cohort v4, run 4)

- **Why:** on about 1% of shots the shooter's box near release was about 20 px tall against 149 px
  for the same player elsewhere in the clip: the detector had locked onto a hand or a fragment,
  and every body-height unit blew up. Player boxes more than 2x shorter or taller than the player's
  median in the clip are now treated as missing.
- **What happened:** the largest defender distance fell from 320 to 14 bh and the largest jump
  from 500 to 6; 19 shots lost their release, 403 changed release frame, splits unchanged. The
  open-shot effect became visible with no cutoff: 0.25 (0.04 to 0.42). H1 on one split: all shots
  +0.0013 (-0.0011 to +0.0039), jump shots -0.0008 (-0.0045 to +0.0029).
- **Intuition:** cleaning fixed the features; the single validation split was still too small to
  show a small gain.

## 12. Date-blocked cross-validation and a constrained model (primary test)

- **Why:** to fix the two remaining problems together: too few scored shots, and overfitting.
  Five folds of contiguous dates over the train and validation games (test never read), XGBoost
  limited to depth 2 and 50 shots per leaf. The primary test was written down before running.
- **What happened:** H1 significant. Jump shots -0.0013 (-0.0021 to -0.0004), Brier 0.1951 to
  0.1939, AUC 0.569 to 0.583. All shots, constrained: -0.0013 (-0.0019 to -0.0007). Jump shots with
  default XGBoost: -0.0017 (-0.0036 to -0.0001).
- **Caveat:** the constrained settings were chosen after seeing overfitting on validation games,
  which are inside the cross-validation, so the design is not fully pre-registered.

## 13. Defender features only

- **Why:** the open-shot effect is real; does it add predictive value?
- **What happened:** defender distance and arms up only: -0.0001 (-0.0007 to +0.0005); all 5
  defender features: -0.0002 (-0.0008 to +0.0004).
- **Intuition:** contest matters but is small, and much of it is already implied by shot type,
  distance and the shooter features (a contested shot is released differently).

## 14. Feature ablation

- **Why:** to find where the gain comes from.
- **What happened:** shooter features alone -0.0011 (-0.0018 to -0.0004); all shots -0.0010
  (-0.0016 to -0.0005). Dropping release height alone gave -0.0006 (-0.0013 to +0.0003); dropping
  any other single feature left -0.0011 to -0.0013. Low releases go in least: on 2-point jumpers
  27% in the lowest fifth vs about 37% otherwise; on threes 22% vs about 25%.
- **Intuition:** the boxes help through one thing: how high the ball is released relative to the
  shooter's body, a coarse summary of a rushed, off-balance or contested release.

## 15. Robustness checks

- **Why:** to rule out luck, fold choice, and the objection that early folds train on later games.
- **What happened:** five other seeds -0.0012 to -0.0017, all significant; 10 folds -0.0012
  (-0.0020 to -0.0003); forward-only folds (each block fit on earlier blocks only) on all shots
  -0.0010 (-0.0019 to -0.0002); forward-only on jump shots -0.0007 (-0.0018 to +0.0003).
- **Intuition:** the one miss has the least power: forward-only skips the first block and fits
  early blocks on little data. It points the same way.

## 16. Blocked shots

- **Why:** blocked shots (5.4%, all misses) were concentrated among the lowest releases (3.6% of
  the lowest fifth vs 0.4 to 1.2% elsewhere). Release detection and ball rise look at frames after
  the release, where a block can land, so blocks could leak the outcome into release height.
- **What happened:** blocked shots are flagged from the play-by-play (a BLOCK at the same period
  and clock) and excluded by `exclude_blocked`. H1 became -0.0009 (-0.0017 to -0.0001), about 25%
  smaller and still significant. This is the frozen design D0.
- **Intuition:** part of the release-height signal was "this shot was blocked"; most of it was not.

## 17. CoTracker3 point tracking (pilot, 25 reviewed clips)

- **Why:** to measure the release more precisely from the video pixels and validate the box
  release.
- **What happened:** worse than the boxes. First try: release found on 18 of 25 clips, 6% within
  3 frames of the reviewer. With a tighter crop and several ball seeds: 10 of 25, 30% within 3.
  Boxes: 84%.
- **Intuition:** a point tracker follows surface texture; a small, plain-coloured, motion-blurred
  ball partly hidden by hands is its worst case, and it kept losing the ball.

## 18. YOLO pose wrists (pilot, 25 reviewed clips)

- **Why:** a more direct definition of release: the ball box leaving the shooter's wrists.
- **What happened:** better than the boxes: 64% within 1 frame (boxes 40%), 88% within 3, median
  error 1 frame, at a reach of one ball width chosen before scoring.
- **Limit:** only the 778 video shots (486 unblocked jump shots, all training) can have pose, so
  it can validate the box features but cannot enter the main test.

## 19. Looking for more data

- **Why:** more shots would tighten the interval, whose upper end is close to zero.
- **What happened:** none available. All 218 games with shot events in the latest BasketEvent
  release (last updated 2025-11-04) are collected. The dataset's `valid/` folder (2 games, 557
  clips, 556 videos) has player and ball boxes but no shot events, so its clips cannot be matched
  to make or miss.
- **Still possible within the data:** about 1,200 shots in multi-shot clips (needs the leak audit
  in 2) and about 950 shots with no detected release.

## 20. Improvement round 1: measuring release height better

Plan and decision rule written down and committed before running (`h1_improvement_round.md`):
each step kept only if its box model beats the last kept design by more than one bootstrap
standard error. Summary: `reports/basketevent_h1_round1/summary.csv`.

| Step | H1 | Gain over last kept | 1 SE | Kept |
|---|---|---|---|---|
| D0 | -0.00093 (-0.00169 to -0.00008) | | | |
| E1 sharper release rule | -0.00096 (-0.00170 to -0.00020) | 0.00000 | 0.00028 | no |
| E2 smoothing over 3 frames | -0.00086 (-0.00165 to -0.00005) | -0.00007 | 0.00011 | no |
| E3 relative to the player's earlier shots | -0.00101 (-0.00181 to -0.00022) | 0.00008 | 0.00014 | no |

- **Why:** release height drives the gain, so measuring it better seemed the most direct lever.
- **Intuition:** E1 really did find the release better (15 of 24 reviewed clips within 1 frame,
  up from 10) and the predictions did not change. The signal is posture at release, not its exact
  timing, and single-frame jitter is not what limits it.

## 21. Improvement round 2: more posture from the boxes

Same rule, fixed before running. Summary: `reports/basketevent_h1_round2/summary.csv`.

| Step | H1 | Gain over last kept | 1 SE | Kept |
|---|---|---|---|---|
| D0 | -0.00093 (-0.00169 to -0.00008) | | | |
| F1 arm extension (box height at release / standing) | -0.00100 (-0.00182 to -0.00017) | 0.00007 | 0.00017 | no |
| F2 release vs jump peak (seconds) | -0.00106 (-0.00183 to -0.00028) | 0.00014 | 0.00016 | no |
| F3 lean toward the basket (bh) | -0.00078 (-0.00155 to +0.00002) | -0.00015 | 0.00018 | no |

- **Why:** if release height works because it captures posture, other posture measures from the
  same boxes might add to it.
- **Intuition:** with release height and the stronger baseline in place, further 2D box
  measurements carry no new information about the outcome. Six reasonable additions across two
  rounds all moved the result by noise-sized amounts while the H1 effect stayed between -0.0008 and
  -0.0011. The boxes look used up; F2 came closest and is worth mentioning only as exploratory.

## Not tried

- **Other model types** (planned, then dropped): the box features alone reach only AUC 0.605, so
  the limit is the information, not the learner.
- **Pose or CoTracker3 features in the H1 model:** 486 video jump shots are too few to detect an
  effect of this size.

## Caveats

- Many analyses were run on the same train and validation games. Each was reasonable, and the two
  improvement rounds followed written rules, but a result this close to zero (upper end -0.0001)
  should be read as moderate evidence until the test split confirms it.
- The constrained XGBoost settings were chosen after seeing overfitting on validation games.
- Early cross-validation folds are fit on later games; forward-only folds keep the effect on all
  shots and its direction on jump shots.
- Results describe BasketEvent's miss-heavy shots.

## What is left

- The one-time test-split confirmation of D0, only when the user says so.
- Optional: recover the multi-shot clips with the leak audit, to tighten the interval.
- Optional: validate box release height against YOLO pose on the 778 video shots.
