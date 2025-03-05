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
From the tag list below, please remove tags that significantly deviate from the given context,
and keep only the tags that are relevant.
Please output the filtered tags as a comma-separated list.
Note that the context represents the content we want to include in the prompt, so please keep the tags that align with the context.

# Example Start ####
Example Context: 女の子がジャンプしているポーズを描いてください。セーラー服ではなくブレザーを着ている。
Example Tag List: original, 1girl, jumping, embarrassed, jacket, solo, outdoors, day, long hair, skirt, kneehighs, smile, looking at viewer, miniskirt, monochrome, monster, open clothes, open jacket, open mouth, pleated skirt, shoes, socks, teeth, upper teeth only, wide shot, wind, wind lift
Example Output: original, 1girl, jumping, jacket, solo, long hair, skirt, smile, looking at viewer, miniskirt, open jacket, shoes, socks
# Example End ####
Note: (deleted tags: embarrassed, outdoors, day, kneehighs, monochrome, monster, open clothes, open mouth, pleated skirt, teeth, upper teeth only, wide shot, wind, wind lift)

Context: {context_prompt}

Tag List: {tags_str}

Output:"""

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