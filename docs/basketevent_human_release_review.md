# BasketEvent human release-review checkpoint (blind-safe)

## Scope and limitations

This report summarizes a **first-pass manual-review checkpoint**, not a finalized
ground-truth dataset or a detector-accuracy evaluation. The reviewer examined
basketball videos and recorded candidate ball-hand separation intervals.
Some frame interpretations and note wording were assisted by an AI assistant.
Identity decisions marked `confirmed` are the reviewing user's judgments
about matches to the annotated shot events, not independent adjudications.
A `provisional` release interval is **not** a verified release frame.
No detector predictions or timing residuals were accessed for this report.

## Coverage

- Sample clips: **20**
- Provisional release intervals: **19**
- Unreviewed release clips: **0**
- Uncertain release clips: **1**
- User-confirmed event identities: **20**
- Pending event identities: **0**
- Uncertain event identities: **0**
- Mismatched event identities: **0**
- Provisional intervals with a user-confirmed event identity: **19**

## Recorded provisional intervals

| Video path | Release interval (frames, inclusive) | Annotated player | Identity decision | Reviewer observation |
|---|---:|---|---|---|
| test/0022500003/265f9cef-a1fd-a9c2-8324-88b10b547a6f_1280x720.mp4 | 128-129 | Kolek | confirmed | A Knicks player in white (#13) attempts a jump shot while a Cavaliers defender in red contests the attempt with his arm raised. At frame 129, the ball appears to be leaving or has just left the shooter's hands. By frame 130, the ball is clearly separated and moving upward toward the basket. I estimated the release interval as 128–129 based on the change in the ball's position. |
| test/0022500003/4566b81e-7ff7-fd11-fbcb-ab34bf99d242_1280x720.mp4 | 119-120 | Allen | confirmed | Possible release around frames 119-120. Ball appears near Allen and moves upward. Another apparent shooting attempt occurs later in the video. Exact release frame is uncertain. |
| test/0022500003/b629385f-3591-f012-9201-3e00653269ed_1280x720.mp4 | 87-88 | Nance Jr. | confirmed | A Cavaliers player in reddish-brown attempts a close-range layup near the left side of the basket while Knicks defenders contest the shot. Around frame 87, the ball appears to still be in the shooter's hand before a defender makes contact and knocks it away. By frame 88, the ball appears separated from his hand. I estimated the ball-hand separation interval as 87–88, although the separation may have resulted from the block rather than a normal shooting release. |
| test/0022500003/ce979297-2c6d-a6df-2455-eac10a268ed4_1280x720.mp4 | 32-34 | Bridges | confirmed | Possible Bridges reverse-layup release around frames 32-34. Ball overlaps the tracked shooter at frame 32 and moves sharply upward by frame 33. Exact release uncertain. |
| test/0022500003/ea680af0-fed1-ddc3-7e6b-3c23c7cc9e42_1280x720.mp4 | 143-144 | McBride | confirmed | A Knicks player in white attempts a shot near the right side of the paint, with several Cavaliers defenders in red nearby. The ball appears to leave the shooter's raised hands around frames 143–144. I estimated this as the release interval, although the exact separation point is somewhat difficult to distinguish. |
| test/0022500004/1392898e-bf37-912a-3c7f-f9e79726ca65_1280x720.mp4 | 79-80 | Washington | confirmed | White #25 attempts a shot near the left baseline. Between frames 79 and 80, the ball appears to separate from his raised hand. I estimated the release interval as 79–80 because the exact separation point is difficult to determine from the footage. |
| test/0022500004/510033bd-05af-44d5-e82e-f2587c332f92_1280x720.mp4 | 85-86 | Johnson | confirmed | A Spurs player in black attempts a close-range shot near the basket while multiple Mavericks defenders in white contest the attempt. Around frame 85, the ball is difficult to distinguish because the players' raised arms overlap. By frame 86, the ball appears to have separated and moved upward toward the basket. I estimated the release interval as 85–86, although the exact moment of separation is somewhat difficult to identify. |
| test/0022500004/a34de8a4-2edc-a386-b5b4-4eb44888a4d8_1280x720.mp4 | 131-132 | Johnson | confirmed | A Spurs player in black (#3) attempts a three-point jump shot from the left corner while a Mavericks defender in white contests with his arm raised. At frame 131, the ball appears to be touching or just leaving the shooter's fingertips. By frame 132, it is clearly separated and moving toward the basket. I estimated the release interval as 131–132. |
| test/0022500004/eb842d48-bf4f-61ae-3f70-519d0dd2fa96_1280x720.mp4 | 75-76 | Davis | confirmed | A Mavericks player in white (#3) attempts a three-point jump shot from the right wing while a Spurs defender in black contests with his arm extended. Around frames 75–76, the ball appears to separate from the shooter's raised hands and move upward. I estimated the release interval as 75–76, although the exact separation point is slightly difficult to distinguish. |
| test/0022500004/ed77e57c-3bfa-9803-e2c5-98d4cf1b135d_1280x720.mp4 | 172-173 | Jones Garcia | confirmed | A Spurs player in black attempts a close-range, one-handed floating shot inside the paint while several Mavericks defenders in white are nearby. At frame 172, the ball appears close to the shooter's raised hand, and by frame 173, it appears to have separated and moved upward toward the basket. I estimated the release interval as 172–173 because the exact moment of separation is somewhat difficult to distinguish. |
| test/0022500004/f8a7eb6a-ab5a-0f86-c2f3-ea856226b33a_1280x720.mp4 | 165-166 | Thompson | confirmed | A Mavericks player in white (#31) attempts a three-point jump shot from the left wing. At frame 164, the ball is close to his raised hands. Around frame 165, it appears to be leaving his fingertips, and by frame 166, it is visibly separated and moving upward toward the basket. I estimated the release interval as 165–166 because the exact separation point is slightly difficult to determine. |
| test/0022500025/06287e20-f527-7a6f-e1dc-0acd87a4aa31_1280x720.mp4 | 82-83 | Allen | confirmed | The Suns player in black (#8) attempts a jump shot near the free-throw line while a defender in white contests with his arm raised. At frame 82, the ball appears close to the shooter's hands, and by frame 83, it is visibly separated and moving upward. I estimated the release interval as 82–83. |
| test/0022500025/3d4503fb-0abb-76b6-8bdf-03685e0e6ca0_1280x720.mp4 | 128-129 | O'Neale | confirmed | Phoenix player #00 attempts a contested jump shot. The ball appears to separate from his raised hands around frames 128-129, with a Sacramento defender jumping to contest. Exact release timing is uncertain. |
| test/0022500025/68189737-b1c8-0caf-4af1-73ab5f074c2b_1280x720.mp4 | 136-137 | Williams | confirmed | A Jazz player in white attempts a three-point jump shot from the right corner. The ball appears to be near the shooter's hands at frame 136, separated by frame 137, and moving farther away by frame 138. I estimated the release interval as 136–137. |
| test/0022500025/7c447ee2-3fa8-a6cd-fb0e-89a20bb22755_1280x720.mp4 | 124-125 | Dunn | confirmed | A Suns player in black jumps toward the basket for an apparent alley-oop dunk attempt. Around frames 124–125, the ball is near the rim and appears airborne while the player reaches toward it. I estimated the release interval as 124–125, but I could not clearly confirm the exact moment the ball separated from his hand. |
| test/0022500025/91cb73ec-1afe-b2e2-6ffc-c74a5fd14409_1280x720.mp4 | 121-122 | Anderson | confirmed | The player in white attempts a close-range shot near the left side of the basket while a defender in black (#15) contests the attempt. The ball appears to separate from the shooter's raised hand around frames 121–122, although the defender's arm and the overlapping players make the exact release frame difficult to determine. |
| test/0022500025/d1f90faf-a181-55bb-5cfb-98db41f75bff_1280x720.mp4 | 57-58 | Bailey | confirmed | A Jazz player in white attempts a contested three-point pull-up jump shot from the right wing around 8:31 in the second quarter. Between frames 57 and 58, the ball appears to separate from his raised hands while a Suns defender in black contests with an outstretched arm. I estimated the release interval as 57–58, although the exact separation point is difficult to distinguish. The clip also shows a separate close-range shooting attempt around frames 199–201, which does not appear to be the annotated three-pointer. |
| test/0022500032/0bdc7db6-a49d-3e51-fe02-148448987c3f_1280x720.mp4 | 84-85 | Sengun | confirmed | The player in red attempts a close-range shot near the basket while a defender in gray contests the attempt with his arm raised. Between frames 84 and 85, the ball appears to separate from the shooter's hand. I estimated the release interval as 84–85, although the defender's arm makes the exact separation point somewhat difficult to identify. |
| test/0022500032/bb91a081-0ed1-8c46-a8e9-8c95102fc685_1280x720.mp4 | 102-103 | Wembanyama | confirmed | A Spurs player in gray attempts a jump shot from around the left elbow while a Rockets defender in red contests with his arm raised. By frame 103, the ball appears to have separated from the shooter's hands, and at frame 104, it moves farther toward the basket. I estimated the release interval as 102–103 based on the ball's position across these frames. |

## Reviewed event-identity judgments

| Video path | Annotated event | Identity decision | Observed shooter | Observed action | Review rationale |
|---|---|---|---|---|---|
| test/0022500003/265f9cef-a1fd-a9c2-8324-88b10b547a6f_1280x720.mp4 | MISS Kolek 26' 3PT Jump Shot | confirmed | Kolek | Jump Shot | Reviewer confirmed the Knicks missed three-point jump shot shown in the video. |
| test/0022500003/4566b81e-7ff7-fd11-fbcb-ab34bf99d242_1280x720.mp4 | MISS Allen 1' Layup | confirmed | Allen | Layup Shot | Reviewer confirmed Allen's annotated layup despite another apparent shooting action in the clip. |
| test/0022500003/b629385f-3591-f012-9201-3e00653269ed_1280x720.mp4 | MISS Nance Jr. 4' Layup | confirmed | Nance Jr. | Layup Shot | Reviewer confirmed the Cavaliers blocked layup matches the annotated event. The observed ball-hand separation may be block-induced rather than a normal shot release. |
| test/0022500003/ce979297-2c6d-a6df-2455-eac10a268ed4_1280x720.mp4 | MISS Bridges 1' Running Reverse Layup | confirmed | Bridges | Running Reverse Layup Shot | Reviewer confirmed the Knicks running reverse layup matches the annotated shot event. |
| test/0022500003/ea680af0-fed1-ddc3-7e6b-3c23c7cc9e42_1280x720.mp4 | MISS McBride 23' Step Back Jump Shot | confirmed | McBride | Step Back Jump shot | Manual video review supports McBride as the shooter. The observed release near frames 143-144 is consistent with the annotated missed step-back jump shot. |
| test/0022500004/1392898e-bf37-912a-3c7f-f9e79726ca65_1280x720.mp4 | MISS Washington 24' 3PT Jump Shot | confirmed | Washington | Jump Shot | Manual video review supports Washington as the shooter. The observed release near frames 79-80 is consistent with the annotated missed jump shot. |
| test/0022500004/510033bd-05af-44d5-e82e-f2587c332f92_1280x720.mp4 | Johnson 5' Cutting Layup Shot (2 PTS) (Kornet 3 AST) | confirmed | Johnson | Cutting Layup Shot | Reviewer confirmed the Spurs made cutting layup matches the annotated shot event. |
| test/0022500004/a34de8a4-2edc-a386-b5b4-4eb44888a4d8_1280x720.mp4 | Johnson 24' 3PT Jump Shot (5 PTS) (Castle 6 AST) | confirmed | Johnson | Jump Shot | Reviewer confirmed the Spurs made three-point jump shot shown in the video. |
| test/0022500004/eb842d48-bf4f-61ae-3f70-519d0dd2fa96_1280x720.mp4 | MISS Davis 26' 3PT Jump Shot | confirmed | Davis | Jump Shot | Reviewer confirmed the Mavericks missed three-point jump shot shown in the video. |
| test/0022500004/ed77e57c-3bfa-9803-e2c5-98d4cf1b135d_1280x720.mp4 | Jones Garcia 8' Driving Floating Jump Shot (4 PTS) | confirmed | Jones Garcia | Driving Floating Jump Shot | Reviewer confirmed the Spurs close-range floating shot matches the annotated event. |
| test/0022500004/f8a7eb6a-ab5a-0f86-c2f3-ea856226b33a_1280x720.mp4 | MISS Thompson 26' 3PT Jump Shot | confirmed | Thompson | Jump Shot | Reviewer confirmed the Mavericks missed three-point jump shot shown in the video. |
| test/0022500025/06287e20-f527-7a6f-e1dc-0acd87a4aa31_1280x720.mp4 | MISS Allen 28' 3PT Jump Shot | confirmed | Allen | Jump Shot | Manual video review supports Allen as the shooter. The observed release near frames 82-83 is consistent with the annotated missed jump shot. |
| test/0022500025/3d4503fb-0abb-76b6-8bdf-03685e0e6ca0_1280x720.mp4 | O'Neale 29' 3PT Pullup Jump Shot (10 PTS) (Gillespie 6 AST) | confirmed | O'Neale | Pullup Jump shot | Reviewer confirmed the annotated Phoenix three-point shot matches the shooting event visible in the video. |
| test/0022500025/68189737-b1c8-0caf-4af1-73ab5f074c2b_1280x720.mp4 | Williams 24' 3PT Jump Shot (3 PTS) (George 5 AST) | confirmed | Williams | Jump Shot | Reviewer confirmed the Jazz made three-point jump shot shown in the video. |
| test/0022500025/7c447ee2-3fa8-a6cd-fb0e-89a20bb22755_1280x720.mp4 | MISS Dunn 1' Running Alley Oop Dunk Shot | confirmed | Dunn | Running Alley Oop Dunk Shot | Reviewer confirmed the annotated alley-oop dunk attempt. The exact ball-hand separation remains ambiguous; event identity confirmation does not validate release timing. |
| test/0022500025/91cb73ec-1afe-b2e2-6ffc-c74a5fd14409_1280x720.mp4 | MISS Anderson 1' Driving Hook Shot | confirmed | Anderson | Driving Hook Shot | Manual video review supports Anderson as the shooter. The observed release near frames 121-122 is consistent with the annotated missed driving hook shot. |
| test/0022500025/d1f90faf-a181-55bb-5cfb-98db41f75bff_1280x720.mp4 | MISS Bailey 25' 3PT Pullup Jump Shot | confirmed | Bailey | Pullup Jump shot | Reviewer confirmed the earlier Jazz missed three-pointer matches the annotated event. A later close-range attempt is a separate shooting action. |
| test/0022500032/0bdc7db6-a49d-3e51-fe02-148448987c3f_1280x720.mp4 | MISS Sengun 5' Hook Shot | confirmed | Sengun | Hook Shot | Manual video review supports Sengun as the shooter. The observed release near frames 84-85 is consistent with the annotated missed hook shot. |
| test/0022500032/695e04af-940c-1f7e-fa3b-46726eac136c_1280x720.mp4 | Champagnie 26' 3PT Jump Shot (16 PTS) (Barnes 2 AST) | confirmed | Champagnie | Jump Shot | Reviewer confirmed the annotated made three-point shot. The clip also contains another apparent shooting motion. The release timing remains uncertain. |
| test/0022500032/bb91a081-0ed1-8c46-a8e9-8c95102fc685_1280x720.mp4 | MISS Wembanyama 27' 3PT Jump Shot | confirmed | Wembanyama | Jump Shot | Reviewer confirmed the Spurs missed three-point jump shot matches the annotated event. |

## Reproduction and integrity

Regenerate this Markdown with `python -m tools.report_release_human_review`
using the two local review worksheets. The CSVs may be ignored locally;
therefore this Markdown preserves a readable snapshot, not the original
source worksheets. Keep the originals securely for later adjudication.

- Release CSV SHA-256: `85b08caacfe64e18477fbe836a5f1d4726aabc65825c2035cb5d537cb3d172a9`
- Identity CSV SHA-256: `db869f9e458941dd6ab2867996218352b17f9d8c9ccdacbc6bcdd157fd9f673e`

**Not evaluated here:** detector accuracy, absolute timing error, inter-rater
agreement, frame-ground-truth validity, or statistical generalization.

## Uncertain release observations

| Video path | Annotated player | Identity decision | Reviewer observation |
|---|---|---|---|
| test/0022500032/695e04af-940c-1f7e-fa3b-46726eac136c_1280x720.mp4 | Champagnie | confirmed | The video appears to show two separate shooting motions, one around frames 12–13 and another around frames 174–175. The later sequence occurs closer to the annotated game clock of 11:40 in the fourth quarter. However, I cannot confidently determine which shooting motion corresponds to Champagnie's recorded three-pointer, so I marked the release timing as uncertain. |
