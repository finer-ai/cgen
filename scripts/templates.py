"""
画像生成に使用するテンプレート設定

このファイルには、画像生成プロセスで使用される各種テンプレートのデフォルト値が含まれています。
"""

# タグ候補生成テンプレート
TAG_CANDIDATE_GENERATION_TEMPLATE = """
Generate Danbooru tags from the prompt. The prompt may either be a question or instruction requesting a scene to be drawn, or a direct description of the desired scene without a question.
First, extract the scene. Then, generate up to 20 Danbooru tags, separated by commas, detailing all relevant aspects including character count, group tags, and other necessary details.

Important character count rules:
- Unless explicitly specified otherwise in the prompt, assume there is only one character.
- For a single character, use exactly one character tag (e.g., '1girl', '1boy', or '1other') along with 'solo'.
- For multiple characters, use tags that match the exact number of characters (e.g., '2girls', '2boys', 'multiple_girls').
- When gender is not explicitly specified in the prompt, prioritize using '1girl' over '1boy'.

Additional rules:
- If there is a single human or humanoid character (including orcs, elves, etc.), use 'solo' along with '1boy' or '1girl'.
- If there is a single non-humanoid character, use 'solo' along with '1other'.
- If applicable, combine species tags (e.g., 'orc', 'elf') with character count tags.

Prompt: {prompt}
Tags (up to 20):
"""

# タグ正規化テンプレート
TAG_NORMALIZATION_TEMPLATE = """
You are a Danbooru tag matcher. Find exact matches in the context for the input.
Rules:
1. Return only exact matches from the context.
2. Check main tags and "Other names".
3. If no match, return input as is.
4. Provide one tag output per input.
5. Match tag names or other names, not descriptions.
6. For character counts and groups:
- Use 'solo' for a single character (human, humanoid, or non-humanoid)
- Use '1boy' for a single male character or humanoid male-presenting character (including orcs, elves, etc.)
- Use '1girl' for a single female character or humanoid female-presenting character
- Use '1other' for a single non-humanoid character
- For other character counts, use appropriate tags like '2boys', 'multiple_girls', etc.
7. Prioritize character count and group tags over other matches if applicable.
8. When using '1boy', '1girl', or '1other', also include the 'solo' tag.
9. Do not use square brackets or double square brackets around tags.
10. Match all relevant tags, including those for background, clothing, expressions, and actions.
11. For humanoid characters like orcs, elves, etc., use appropriate character count tags ('1boy', '1girl', etc.) along with their species tag.

Context: {context}
Input: {query}
Output:
"""

# タグフィルターテンプレート
TAG_FILTER_TEMPLATE = """
From the tag list below:
1. Remove tags that significantly deviate from the given context
2. Keep tags that are relevant to the context
3. Add missing pose-related tags by analyzing the context and comparing with the pose reference list below
Please output the filtered and enhanced tags as a comma-separated list.

# Pose Reference List ####
tag: "butterfly sitting", desc: "Sitting in a butterfly position / 蝶の座り方"
tag: "indian style", desc: "Sitting cross-legged in an Indian style / あぐら"
tag: "knees apart feet together", desc: "Sitting with knees apart and feet together / 膝を広げて足を合わせる"
tag: "leg lock", desc: "Locking legs together / 足を絡める"
tag: "lotus position", desc: "Sitting in a lotus position / 蓮華座"
tag: "spread legs", desc: "Sitting with legs spread apart / 脚を広げる"
tag: "arm support", desc: "Supporting oneself with arms / 腕を支える"
tag: "crossed legs", desc: "Sitting with legs crossed / 足を組む"
tag: "fetal position", desc: "Curled up in a fetal position / 胎児のような姿勢"
tag: "figure four sitting", desc: "Sitting in a figure-four position / 四の字座り"
tag: "knees together feet apart", desc: "Sitting with knees together and feet apart / 膝を合わせる"
tag: "sitting", desc: "General sitting position / 座る"
tag: "slouching", desc: "Slouching while sitting / 猫背になる"
tag: "head rest", desc: "Resting head on something / ヘッドレスト"
tag: "hugging own legs", desc: "Hugging one's own legs / 自分の足をハグ"
tag: "hugging own legs", desc: "Hugging one's own legs / 自分の足を抱きしめる"
tag: "knee up", desc: "Sitting with one knee up / 膝を上げる"
tag: "knees to chest", desc: "Bringing knees close to the chest / 膝を胸につける"
tag: "knees up", desc: "Sitting with both knees up / 膝を上げる"
tag: "over the knee", desc: "Resting something over the knee / 膝の上"
tag: "reclining", desc: "Leaning back while sitting / リクライニング"
tag: "sitting on lap", desc: "Sitting on someone's lap / 膝の上に座る"
tag: "sitting on shoulder", desc: "Sitting on someone's shoulder / 肩に座る"
tag: "all fours", desc: "Being on all fours / 四つん這い"
tag: "crawling", desc: "Crawling on the ground / 這う"
tag: "kneeling", desc: "Kneeling on the ground / ひざをつく"
tag: "on one knee", desc: "Resting on one knee / 片膝をつく"
tag: "pigeon pose", desc: "Performing the pigeon yoga pose / 鳩のポーズ"
tag: "prostration", desc: "Lying flat in prostration / 平伏"
tag: "seiza", desc: "Sitting in the traditional Japanese seiza position / 正座"
tag: "squatting", desc: "Squatting down / しゃがむ"
tag: "superhero landing", desc: "Doing a superhero landing / スーパーヒーローの着地"
tag: "top-down bottom-up", desc: "Moving from top to bottom and back up / 上から下から上へ"
tag: "wariza", desc: "Sitting in wariza (split-leg seiza) position / 割り座"
tag: "yokozuwari", desc: "Sitting in yokozuwari (side-sitting) position / 横割り"
tag: "sitting on person", desc: "Sitting on top of another person / 人の上に座る"
tag: "straddling", desc: "Straddling something or someone / またがる"
tag: "thigh straddling", desc: "Straddling someone's thigh / 太ももをまたぐ"
tag: "upright straddle", desc: "Straddling while standing upright / 直立してまたがる"
tag: "legs over head", desc: "Lifting legs over the head / 脚を頭の上に上げる"
tag: "lying", desc: "Lying down / 横になる"
tag: "on back", desc: "Lying on one's back / 仰向けになる"
tag: "on side", desc: "Lying on one's side / 横向きになる"
tag: "on stomach", desc: "Lying face down / うつ伏せになる"
tag: "chest stand handstand", desc: "Performing a chest stand or handstand / 胸立ち逆立ち_cleanup"
tag: "handstand", desc: "Balancing on hands / 逆立ち"
tag: "upside-down", desc: "Being in an upside-down position / 逆さま"
tag: "a-pose", desc: "Standing in an A-pose / A ポーズ"
tag: "animal hug", desc: "Hugging an animal / 動物ハグ"
tag: "animal pose", desc: "Posing like an animal / 動物のポーズ"
tag: "arm behind back", desc: "Placing arms behind the back / 腕を背中の後ろで組む"
tag: "arm behind head", desc: "Placing arms behind the head / 腕を頭の後ろで組む"
tag: "arm up", desc: "Raising an arm / 腕を上げる"
tag: "arms behind head", desc: "Both arms placed behind the head / 腕を頭の後ろで組む"
tag: "arms up", desc: "Raising both arms / 腕を上げる"
tag: "baby carry", desc: "Carrying a baby / 赤ちゃん抱っこ"
tag: "balancing", desc: "Balancing the body / バランスを取る_1"
tag: "bent over", desc: "Bending over / 身をかがめる"
tag: "carried breast rest", desc: "Carried while resting on someone's chest / 胸を乗せて抱っこ"
tag: "carrying over shoulder", desc: "Carrying someone over the shoulder / 肩にかける"
tag: "child carry", desc: "Carrying a child / 子供抱っこ"
tag: "claw pose", desc: "Doing a claw-like pose / 爪のポーズ"
tag: "contrapposto", desc: "Standing in a contrapposto pose / コントラポスト"
tag: "cowering", desc: "Cowering in fear / 身をすくめる"
tag: "crossed ankles", desc: "Crossing ankles / 足首を組む"
tag: "crossed arms", desc: "Crossing arms / 腕を組む"
tag: "crucifixion", desc: "Assuming a crucifixion pose / 磔"
tag: "dojikko pose", desc: "Performing a dojikko pose / どじっこポーズ"
tag: "fighting stance", desc: "Standing in a fighting stance / ファイティング スタンス"
tag: "flexing", desc: "Flexing muscles / 曲げる"
tag: "full scorpion", desc: "Performing a full scorpion yoga pose / フル スコーピオン"
tag: "head back", desc: "Tilting head backward / 頭を後ろに"
tag: "head down", desc: "Tilting head downward / 頭を下げる"
tag: "head tilt", desc: "Tilting head to the side / 頭を傾ける"
tag: "horns pose", desc: "Doing a horns pose / 角のポーズ"
tag: "hugging object", desc: "Hugging an object / 物をハグ"
tag: "hugging tail", desc: "Hugging one's own tail / 尻尾をハグ"
tag: "incoming hug", desc: "Approaching with arms open for a hug / 迫りくるハグ"
tag: "JoJo pose", desc: "Performing a JoJo-style pose / ジョジョのポーズ"
tag: "Jonathan Joestars pose", desc: "Jonathan Joestar's iconic pose / ジョナサンジョースターのポーズ"
tag: "jumping", desc: "Jumping in the air / ジャンプする"
tag: "leaning forward", desc: "Leaning forward / 前かがみになる"
tag: "leg lift", desc: "Lifting one leg / 脚を上げる"
tag: "leg up", desc: "Raising a leg / 脚を上げる"
tag: "legs apart", desc: "Standing with legs apart / 足を広げる"
tag: "ojou-sama pose", desc: "Performing an ojou-sama pose / お嬢様ポーズ"
tag: "outstretched arm", desc: "Extending an arm outward / 伸ばした腕"
tag: "outstretched hand", desc: "Extending a hand outward / 伸ばした手"
tag: "outstretched leg", desc: "Extending a leg outward / 伸ばした脚"
tag: "own hands clasped", desc: "Clasping one's own hands / 自分の手を握る"
tag: "own hands together", desc: "Joining one's own hands together / 自分の手を合わせる"
tag: "paw pose", desc: "Performing a paw-like pose / 足のポーズ"
tag: "pigeon-toed", desc: "Sitting with pigeon-toed legs / 内股"
tag: "plantar flexion", desc: "Performing plantar flexion / 底屈"
tag: "rabbit pose", desc: "Performing a rabbit pose / ウサギのポーズ"
tag: "reaching", desc: "Reaching out with a hand / 手を伸ばす"
tag: "running", desc: "Running / 走る"
tag: "scorpion pose", desc: "Performing a scorpion yoga pose / サソリのポーズ"
tag: "shoulder carry", desc: "Carrying someone on the shoulder / 肩に乗せる"
tag: "shrugging", desc: "Shrugging shoulders / 肩をすくめる"
tag: "spread arms", desc: "Spreading arms / 腕を広げる"
tag: "spread eagle position", desc: "Adopting a spread eagle position / 大股開きの鷲の姿勢"
tag: "standing on one leg", desc: "Standing on one leg / 片足で立つ"
tag: "standing on shoulder", desc: "Standing on someone's shoulder / 肩に立つ"
tag: "standing split", desc: "Performing a standing split / 立って開脚する"
tag: "standing", desc: "Standing / 立つ"
tag: "stretching", desc: "Performing a stretch / ストレッチ"
tag: "stroking own chin", desc: "Stroking one's own chin / 自分のあごをなでる"
tag: "symmetrical hand pose", desc: "Adopting a symmetrical hand pose / 左右対称の手のポーズ"
tag: "t-pose", desc: "Adopting a T-pose / T ポーズ"
tag: "tiptoes", desc: "Standing on tiptoes / つま先立ち"
tag: "v arms", desc: "Adopting a V-shaped arm pose / V 字腕"
tag: "victory pose", desc: "Adopting a victory pose / 勝利のポーズ"
tag: "w arms", desc: "Adopting a W-shaped arm pose / W 字腕"
tag: "walk cycle", desc: "Performing a walk cycle / 歩行サイクル"
tag: "walking", desc: "Walking / 歩く"
tag: "zombie pose", desc: "Adopting a zombie pose / ゾンビのポーズ"
tag: "arm hug", desc: "Giving an arm hug / 腕ハグ"
tag: "ass-to-ass", desc: "An intimate pose with buttocks together / お尻同士"
tag: "back-to-back", desc: "Standing back-to-back / 背中合わせ"
tag: "belly-to-belly", desc: "With bellies together / 腹同士"
tag: "cheek-to-breast", desc: "Placing a cheek on the breast / 頬を胸につける"
tag: "cheek-to-cheek", desc: "Cheeks touching / 頬同士"
tag: "circle formation", desc: "Forming a circle / 輪になる"
tag: "eye contact", desc: "Making eye contact / アイコンタクト"
tag: "face-to-face", desc: "Face-to-face / 対面"
tag: "forced hug", desc: "Forcing a hug / 無理矢理ハグ"
tag: "forehead-to-forehead", desc: "Foreheads touching / 額同士"
tag: "group hug", desc: "A group hug / グループハグ"
tag: "head on chest", desc: "Resting one's head on someone's chest / 頭を胸につける"
tag: "heads together", desc: "Heads together / 頭を合わせる"
tag: "holding hands", desc: "Holding hands / 手を繋ぐ"
tag: "hug from behind", desc: "Hugging from behind / 後ろからハグ"
tag: "hug", desc: "Giving a hug / ハグ"
tag: "hugging anothers leg", desc: "Hugging another's leg / 相手の足をハグ"
tag: "imminent hug", desc: "An imminent hug / 差し迫ったハグ"
tag: "locked arms", desc: "Locking arms together / 腕を組む"
tag: "mutual hug", desc: "Mutual hug / お互いのハグ"
tag: "noses touching", desc: "Noses touching / 鼻を合わせる"
tag: "princess carry", desc: "A princess carry / お姫様抱っこ"
tag: "shoulder-to-shoulder", desc: "Shoulder-to-shoulder / 肩同士"
tag: "tiptoe kiss", desc: "Kissing while standing on tiptoes / つま先立ちキス"

# Example Start ####
Example Context: 女の子がジャンプしているポーズを描いてください。セーラー服ではなくブレザーを着ている。
Example Tag List: original, 1girl, embarrassed, jacket, solo, outdoors, day, long hair, skirt, kneehighs, smile, looking at viewer, miniskirt, monochrome, monster, open clothes, open jacket, open mouth, pleated skirt, shoes, socks, teeth, upper teeth only, wide shot, wind, wind lift
Example Output: original, 1girl, jumping, jacket, solo, long hair, skirt, smile, looking at viewer, miniskirt, open jacket, shoes, socks
Note: Added 'jumping' based on context description. Removed irrelevant tags.

Context: {context_prompt}

Tag List: {tags_str}

Output (filtered and enhanced tags):"""

# タグ重み付けテンプレート
TAG_WEIGHT_TEMPLATE = """
Please analyze the context and add weights ONLY when there are explicit or strongly implied modifiers in the context.
Be aggressive in applying weights when modifiers are present, but DO NOT add weights when there are no modifiers.

Important rules for weight application:
1. NO WEIGHTS by default:
   - If a tag has no direct or implied modifier, leave it as is without weights
   - Do not add weights based on assumptions or general knowledge
   - Only add weights when there is clear evidence in the context

2. POSE-RELATED weights (:1.3):
   - Apply :1.3 to ALL pose-contributing tags when the context specifically requests a pose
   - This includes:
     a) Direct pose tags: standing, sitting, jumping, running, lying, kneeling, etc.
     b) Body part position tags: arms_up, crossed_legs, spread_arms, hands_in_pockets, etc.
     c) Action-implying tags: looking_back, hair_flip, stretching, twirling, etc.
     d) Pose-affecting clothing tags: skirt_flip, wind_lift, floating_hair, etc.
     e) Dynamic elements: wind, motion_blur, action_lines, etc.

3. Direct modifiers - Apply to ALL related tags when present:
   - Low intensity (:0.3): ちょっと (a little), 少し (slightly), やや (somewhat), 微妙に (subtly), 控えめに (moderately)
   - High intensity (:1.7): すごく (very), とても (extremely), 非常に (highly), かなり (considerably), めちゃくちゃ (incredibly)

4. Mood/Atmosphere modifiers - Apply ONLY when explicitly mentioned or strongly implied:
   When the context explicitly mentions these moods, apply weights to ALL related tags:
   
   a) Suggestive/Erotic mood:
      - If words like エッチ, セクシー, 色気 appear:
        * Apply :0.3 to: sexually_suggestive, revealing_clothes, suggestive_pose, etc.
        * Also apply :0.3 to related tags: thighhighs, miniskirt, bare_shoulders, etc.
   
   b) Cute/Innocent mood:
      - If words like 可愛い, innocent appear:
        * Apply :1.7 to: cute, innocent, pure, etc.
        * Also apply :1.7 to related tags: smile, flower, pastel_colors, etc.
   
   c) Intensity modifiers stack with mood:
      - "ちょっとエッチ" = (sexually_suggestive:0.3)
      - "すごくエッチ" = (sexually_suggestive:1.7)

5. Context Analysis - Strict Rules:
   - Only consider explicit modifiers or very strong contextual implications
   - Do not add weights based on subtle implications or assumptions
   - When in doubt, leave the tag without weights
   - Pose weights (:1.3) can stack with mood/modifier weights

# Examples ####
Example 1 (Pose with modifiers):
Context: ちょっとエッチな感じで女の子がジャンプしているポーズを描いてください。スカートがふわっとなっています。
Tags: original, 1girl, jumping, solo, sexually_suggestive, skirt, smile, thighhighs, wind_lift
Output: original, 1girl, (jumping:1.3), solo, (sexually_suggestive:0.3), (skirt:0.3), smile, (thighhighs:0.3), (wind_lift:1.3)

Example 2 (Simple pose):
Context: 女の子が両手を広げてバランスを取っているポーズを描いてください。
Tags: original, 1girl, standing, spread_arms, balancing, skirt
Output: original, 1girl, (standing:1.3), (spread_arms:1.3), (balancing:1.3), skirt

Example 3 (Mixed modifiers with pose):
Context: すごく可愛らしい女の子が、ちょっとセクシーな感じで後ろを振り返るポーズ。
Tags: original, 1girl, looking_back, turning, cute, sexually_suggestive, smile
Output: original, 1girl, (looking_back:1.3), (turning:1.3), (cute:1.7), (sexually_suggestive:0.3), (smile:1.7)

Example 4 (Dynamic pose):
Context: 風で制服がなびいている中、ジャンプしている女の子を描いてください。
Tags: original, 1girl, jumping, school_uniform, wind, floating_hair, skirt_flip
Output: original, 1girl, (jumping:1.3), school_uniform, (wind:1.3), (floating_hair:1.3), (skirt_flip:1.3)

Example 5 (No clear modifiers):
Context: 放課後の教室で本を読んでいる女の子を描いてください。
Tags: original, 1girl, classroom, reading, book, school_uniform, afternoon
Output: original, 1girl, classroom, (reading:1.3), book, school_uniform, afternoon

Context: {context_prompt}

Tags: {tags_str}

Output:""" 