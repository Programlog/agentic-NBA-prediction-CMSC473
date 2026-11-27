# BasketEvent human release-review checkpoint (blind-safe)

## Scope and limitations

This report summarizes a **partial manual-review checkpoint**, not a finalized
ground-truth dataset or a detector-accuracy evaluation. The reviewer examined
basketball videos and recorded candidate ball-hand separation intervals.
Some frame interpretations and note wording were assisted by an AI assistant.
Identity decisions marked `confirmed` are the reviewing user's judgments
about matches to the annotated shot events, not independent adjudications.
A `provisional` release interval is **not** a verified release frame.
No detector predictions or timing residuals were accessed for this report.

## Coverage

- Sample clips: **20**
- Provisional release intervals: **8**
- Unreviewed release clips: **12**
- Uncertain release clips: **0**
- User-confirmed event identities: **5**
- Pending event identities: **15**
- Uncertain event identities: **0**
- Mismatched event identities: **0**
- Provisional intervals with a user-confirmed event identity: **5**

## Recorded provisional intervals

| Video path | Release interval (frames, inclusive) | Annotated player | Identity decision | Reviewer observation |
|---|---:|---|---|---|
| test/0022500003/4566b81e-7ff7-fd11-fbcb-ab34bf99d242_1280x720.mp4 | 119-120 | Allen | pending | Possible release around frames 119-120. Ball appears near Allen and moves upward. Another apparent shooting attempt occurs later in the video. Exact release frame is uncertain. |
| test/0022500003/ce979297-2c6d-a6df-2455-eac10a268ed4_1280x720.mp4 | 32-34 | Bridges | pending | Possible Bridges reverse-layup release around frames 32-34. Ball overlaps the tracked shooter at frame 32 and moves sharply upward by frame 33. Exact release uncertain. |
| test/0022500003/ea680af0-fed1-ddc3-7e6b-3c23c7cc9e42_1280x720.mp4 | 143-144 | McBride | confirmed | A Knicks player in white attempts a shot near the right side of the paint, with several Cavaliers defenders in red nearby. The ball appears to leave the shooter's raised hands around frames 143–144. I estimated this as the release interval, although the exact separation point is somewhat difficult to distinguish. |
| test/0022500004/1392898e-bf37-912a-3c7f-f9e79726ca65_1280x720.mp4 | 79-80 | Washington | confirmed | White #25 attempts a shot near the left baseline. Between frames 79 and 80, the ball appears to separate from his raised hand. I estimated the release interval as 79–80 because the exact separation point is difficult to determine from the footage. |
| test/0022500025/06287e20-f527-7a6f-e1dc-0acd87a4aa31_1280x720.mp4 | 82-83 | Allen | confirmed | The Suns player in black (#8) attempts a jump shot near the free-throw line while a defender in white contests with his arm raised. At frame 82, the ball appears close to the shooter's hands, and by frame 83, it is visibly separated and moving upward. I estimated the release interval as 82–83. |
| test/0022500025/3d4503fb-0abb-76b6-8bdf-03685e0e6ca0_1280x720.mp4 | 128-129 | O'Neale | pending | Phoenix player #00 attempts a contested jump shot. The ball appears to separate from his raised hands around frames 128-129, with a Sacramento defender jumping to contest. Exact release timing is uncertain. |
| test/0022500025/91cb73ec-1afe-b2e2-6ffc-c74a5fd14409_1280x720.mp4 | 121-122 | Anderson | confirmed | The player in white attempts a close-range shot near the left side of the basket while a defender in black (#15) contests the attempt. The ball appears to separate from the shooter's raised hand around frames 121–122, although the defender's arm and the overlapping players make the exact release frame difficult to determine. |
| test/0022500032/0bdc7db6-a49d-3e51-fe02-148448987c3f_1280x720.mp4 | 84-85 | Sengun | confirmed | The player in red attempts a close-range shot near the basket while a defender in gray contests the attempt with his arm raised. Between frames 84 and 85, the ball appears to separate from the shooter's hand. I estimated the release interval as 84–85, although the defender's arm makes the exact separation point somewhat difficult to identify. |

## Reviewed event-identity judgments

| Video path | Annotated event | Identity decision | Observed shooter | Observed action | Review rationale |
|---|---|---|---|---|---|
| test/0022500003/ea680af0-fed1-ddc3-7e6b-3c23c7cc9e42_1280x720.mp4 | MISS McBride 23' Step Back Jump Shot | confirmed | McBride | Step Back Jump shot | Manual video review supports McBride as the shooter. The observed release near frames 143-144 is consistent with the annotated missed step-back jump shot. |
| test/0022500004/1392898e-bf37-912a-3c7f-f9e79726ca65_1280x720.mp4 | MISS Washington 24' 3PT Jump Shot | confirmed | Washington | Jump Shot | Manual video review supports Washington as the shooter. The observed release near frames 79-80 is consistent with the annotated missed jump shot. |
| test/0022500025/06287e20-f527-7a6f-e1dc-0acd87a4aa31_1280x720.mp4 | MISS Allen 28' 3PT Jump Shot | confirmed | Allen | Jump Shot | Manual video review supports Allen as the shooter. The observed release near frames 82-83 is consistent with the annotated missed jump shot. |
| test/0022500025/91cb73ec-1afe-b2e2-6ffc-c74a5fd14409_1280x720.mp4 | MISS Anderson 1' Driving Hook Shot | confirmed | Anderson | Driving Hook Shot | Manual video review supports Anderson as the shooter. The observed release near frames 121-122 is consistent with the annotated missed driving hook shot. |
| test/0022500032/0bdc7db6-a49d-3e51-fe02-148448987c3f_1280x720.mp4 | MISS Sengun 5' Hook Shot | confirmed | Sengun | Hook Shot | Manual video review supports Sengun as the shooter. The observed release near frames 84-85 is consistent with the annotated missed hook shot. |

## Reproduction and integrity

Regenerate this Markdown with `python -m tools.report_release_human_review`
using the two local review worksheets. The CSVs may be ignored locally;
therefore this Markdown preserves a readable snapshot, not the original
source worksheets. Keep the originals securely for later adjudication.

- Release CSV SHA-256: `d6371c656d72ed4a63f734776d043864d5d3553a0a7d5247719b2c105c0e9844`
- Identity CSV SHA-256: `1c3d6d895839fb7385e98ae4f40fab9cd603cb73f72559c34270a36b2de71892`

**Not evaluated here:** detector accuracy, absolute timing error, inter-rater
agreement, frame-ground-truth validity, or statistical generalization.
